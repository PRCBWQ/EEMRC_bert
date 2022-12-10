import random
import numpy as np

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from config import Config, log, BaseArgs
import torch
from model.MRC_model import BertForQA, BertForEventTriggerAndType
from train_eval import train

from dataProcess.dataset import DuEEEventDataset, collate_fn_type_task, collate_fn_trigger_task, collate_fn_role_task

logger = log.logger
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from getQuestion import Question
from dataProcess.dataset import tokenoffset2exoffset
import json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class EEMRCpredict():
    def __init__(self, config: Config, model_trigger, model_type, model_role):
        self.model_trigger = model_trigger
        self.model_type = model_type
        self.model_role = model_role
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(config.model_dir, config.model_name))
        self.que = Question()

        # event schema
        self.id2type = []  # id to event type
        self.type2id = {}
        self.typeNum = 0
        # 加载事件schema
        with open(config.schema_path, 'r', encoding='utf-8') as fs:
            lines = fs.readlines()
            id = 0
            for line in lines:
                data = json.loads(line.strip())
                event_type = data["event_type"]
                self.id2type.append(event_type)
                self.type2id[event_type] = {}
                self.type2id[event_type]["id"] = id
                self.type2id[event_type]["roles"] = []
                for role in data["role_list"]:
                    self.type2id[event_type]["roles"].append(role["role"])
                id = id + 1
            self.typeNum = id

    def model_predict(self, question: str, sentence: str, model, task_type="type_task"):
        model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(
                question,  # 融入问题
                sentence,
                max_length=self.config.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt",
                return_offsets_mapping=True
            )
            offset_map = inputs['offset_mapping']
            inputs.pop('offset_mapping')
            inputs.to(config.device)
            outputs = model(**inputs, task_type=task_type)
            outputs["token_type_ids"] = inputs.token_type_ids
            outputs["offsets_mapping"] = offset_map
            return outputs

    def qa_span_decode_simple(self, sentence: str, start_logits, end_logits, token_type_ids,
                              offsets_mapping):  # 简单解码算法(复杂解码算法其实是绣花提升不了多少,嘿嘿，嘘嘘)
        with torch.no_grad():
            answer_start_index = start_logits.argmax()
            answer_end_index = end_logits.argmax()
            if answer_start_index > answer_end_index:
                return False
            # print("ans token start:{} end:{}".format(answer_start_index, answer_end_index))
            org_start = tokenoffset2exoffset(answer_start_index, token_type_ids, offsets_mapping, isLeft=True)
            org_end = tokenoffset2exoffset(answer_end_index, token_type_ids, offsets_mapping, isLeft=False)  # 不包含
            if org_start >= org_end or org_start == -1 or org_end == -1:
                return False
            print("start{} end{}".format(org_start, org_end))
            # print("ans org start:{} end:{}".format(org_start, org_end))
            # print("ans word is {}".format(sentence[org_start, org_end]))
            return sentence[org_start:org_end + 1]

    def EE_predict(self, sentence: str):  # 三个任务总入口
        outputs = self.model_predict(self.que.getTriggerQ(), sentence, self.model_trigger, task_type="trigger_task")
        trigger = self.qa_span_decode_simple(sentence, outputs["start_logits"], outputs["end_logits"],
                                             outputs["token_type_ids"], outputs["offsets_mapping"])
        print("trigger is {}".format(trigger))
        if trigger is False:
            return trigger
        type_outputs = self.model_predict(self.que.getTypeQ(trigger), sentence, self.model_type, task_type="type_task")
        typeid = type_outputs["type_logits"].argmax()
        event_type = self.id2type[typeid]
        print("type is {}".format(event_type))

        rolesList = self.type2id[event_type]["roles"].copy()
        print("type {} schema is {}".format(event_type, rolesList))
        roledict = {}
        for role in rolesList:
            question = self.que.getRoleQ(trigger, event_type, role)
            outputs = self.model_predict(question, sentence, self.model_role, task_type="role_task")
            roleword = self.qa_span_decode_simple(sentence, outputs["start_logits"], outputs["end_logits"],
                                                  outputs["token_type_ids"], outputs["offsets_mapping"])
            print("role {} is {}".format(role, roleword))
            if trigger is False:
                continue
            roledict[role] = roleword

        event = dict()
        event["trigger"] = trigger
        event["type"] = event_type
        event["roles"] = roledict
        return event


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python run py
    config = Config()
    set_seed(config.seed)
    args = BaseArgs().get_parser()
    # '''
    # 训练示例
    # 构造tokenizer和dataset
    torch.cuda.set_device(args.gpu_ids)
    if args.mode == "train":
        config.max_seq_length = args.max_seq_len
        config.nums_epochs = args.epoch
        config.batch_size = args.batch_size
        config.model_name = args.bert_type
        tokenizer = BertTokenizerFast.from_pretrained(os.path.join(config.model_dir, config.model_name))

        train_dataset = DuEEEventDataset(config,
                                         data_path=config.train_path,
                                         schema_path=config.schema_path,
                                         tokenizer=tokenizer)
        trigger_DataLoader = DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=config.batch_size,
                                        collate_fn=collate_fn_trigger_task,
                                        num_workers=20)
        type_DataLoader = DataLoader(train_dataset,
                                     shuffle=True,
                                     batch_size=config.batch_size,
                                     collate_fn=collate_fn_type_task,
                                     num_workers=20)
        role_DataLoader = DataLoader(train_dataset,
                                     shuffle=True,
                                     batch_size=config.batch_size // 2,  # role需要减小batch节省显存
                                     collate_fn=collate_fn_role_task,
                                     num_workers=20)

        # model = BertForQA(config)
        model = BertForEventTriggerAndType(config)
        # 如果需要加载已经训练的模型，则如下
        # path = "/home/BWQ/EEMRC_bert/model/checkpoints/role_task/checkpoint-role_task-step-920.pkl"
        # model.load_state_dict(torch.load(path))
        model.to(config.device)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", config.nums_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", config.batch_size)
        if args.task_type == "trigger":
            print("Trigger Task")
            train(config, model, trigger_DataLoader, tokenizer, task="trigger_task")

        elif args.task_type == "type":
            print("Type Task")
            train(config, model, type_DataLoader, tokenizer, task="type_task")
        elif args.task_type == "role":
            print("Role Task")
            train(config, model, role_DataLoader, tokenizer, task="role_task")
        else:
            print("arg task_type is wrong")
        print("done")
    else:
    # predict 示例
    # 加载trigger model、 type model、 role model
    # 简单提一下，本项目trigger_type联合模型数据并无突出优势，理论上本项目模型的结构可以直接玩端到端模型，
    # 对了，我论文数据故意写低了，毕竟写低不算不端，写高了算......
    # 但是炼丹玄学太多了，师兄精力有限，师弟师妹加油成功了效果不错的话 还能搞一篇
    # 题目就叫《基于预训练模型的MRC式端到端事件抽取》，端到端玄学训练再加点UDA当小料，发个中等会或国内期刊应该没问题（2022.4.25）)
        model_trigger = BertForEventTriggerAndType(config)
        model_trigger_path = os.path.join(config.output_dir, "trigger_task/checkpoint-trigger_task-step-11960.pkl")
        model_trigger.load_state_dict(torch.load(model_trigger_path))
        model_trigger.to(config.device)

        model_type = BertForEventTriggerAndType(config)
        model_type_path = os.path.join(config.output_dir, "type_task/checkpoint-type_task-step-11960.pkl")
        model_type.load_state_dict(torch.load(model_type_path))
        model_type.to(config.device)

        model_role = BertForEventTriggerAndType(config)
        model_role_path = os.path.join(config.output_dir, "role_task/checkpoint-role_task-step-1840.pkl")
        model_role.load_state_dict(torch.load(model_role_path))
        model_role.to(config.device)

        EEMRCpredModel = EEMRCpredict(config, model_trigger, model_type, model_role)
        EEMRCpredModel.EE_predict("雀巢裁员4000人：时代抛弃你时，连招呼都不会打！")
