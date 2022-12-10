import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 进度条
from transformers import BertTokenizerFast
from config import log, Config
from getQuestion import Question
import json
import torch.nn as nn

logger = log.logger
max_len = 512
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def getStartoffsset(token_type_ids):
    start = 0
    if isinstance(token_type_ids, torch.Tensor):
        length = token_type_ids[0].size()[0]
        for i in range(length):
            if token_type_ids[0][i] == 1:
                start = i
                break
    else:
        length = len(token_type_ids)
        for i in range(length):
            if token_type_ids[i] == 1:
                start = i
                break
    return start


def exoffset2tokenoffset(start_offset: int, end_offset: int, token_type_ids, offset_mapping):
    length = -1
    start = getStartoffsset(token_type_ids)
    ans_start = start - 1
    ans_end = start - 1
    if start_offset <= -1 or end_offset <= -1:  # 如果无答案，则返回原句前一个token对应offset
        return start - 1
    if isinstance(offset_mapping, torch.Tensor):
        for i in range(start, offset_mapping[0].size()[0]):
            if offset_mapping[0][i][1] == 0:
                break
            if offset_mapping[0][i][1] > start_offset >= offset_mapping[0][i][0]:
                ans_start = i
            if offset_mapping[0][i][1] > end_offset >= offset_mapping[0][i][0]:
                ans_end = i
                break
    else:
        for i in range(start, len(offset_mapping)):
            if offset_mapping[i][1] == 0:
                break
            if offset_mapping[i][1] > start_offset >= offset_mapping[i][0]:
                ans_start = i
            if offset_mapping[i][1] > end_offset >= offset_mapping[i][0]:
                ans_end = i
                break
    return ans_start, ans_end


def tokenoffset2exoffset(offset: int, token_type_ids, offset_mapping, isLeft=True):
    # 将tokenoffset转换为原句offset
    # isleft表明取范围区间的左值还是右值
    if isinstance(token_type_ids, torch.Tensor):
        length = token_type_ids[0].size()[0]
        jud = token_type_ids[0][offset]
    else:
        length = len(token_type_ids)
        jud = token_type_ids[offset]
    if offset >= length:
        return -1
    index = 0 if isLeft else 1
    ans = -1
    if jud == 1:  # 在原句范围内，如果在之外表明无答案 返回-1
        if isinstance(offset_mapping, torch.Tensor):
            ans = offset_mapping[0][offset][index].item() - index
        else:
            ans = offset_mapping[offset][index] - index
    return ans


class DuEEEventDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, config: Config, data_path, schema_path, tokenizer: BertTokenizerFast):
        # 加载id2entity
        self.id2type = []  # id to event type
        self.type2id = {}
        self.typeNum = 0
        self.examples = []
        self.tokenized_examples = []  # dict trigger_task type_task role_task
        # 加载事件schema
        with open(schema_path, 'r', encoding='utf-8') as fs:
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

        # 加载准备好的训练集
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                self.examples.append(data)
        # print(self.examples)
        # 将训练集转换为token张量，
        with tqdm(enumerate(self.examples), total=len(self.examples), desc="tokenizing...ing") as pbar:
            # tokenizer = BertTokenizerFast.from_pretrained(os.path.join(config.model_dir))
            que = Question()
            for i, example in pbar:
                TokenInput = {}
                # if len(example["event_list"]) == 0:
                #     continue

                # trigger_task
                # print(event_dict)
                tokenized_trigger = tokenizer.encode_plus(
                    que.getTriggerQ(),  # 融入问题
                    example["text"],
                    max_length=config.max_seq_length,
                    truncation=True,
                    padding='max_length',
                    # return_tensors="pt",
                    return_offsets_mapping=True
                )
                event_dict = {}
                if len(example["event_list"]) == 0:  # 无事件时
                    token_offset = getStartoffsset(tokenized_trigger.token_type_ids)
                    tokenized_trigger["start_positions"] = token_offset - 1
                    tokenized_trigger["end_positions"] = token_offset - 1
                    print("example no event :{}".format(example["text"]))
                    print("token_offset :{}".format(token_offset))
                else:  # 存在事件时
                    event_dict = example["event_list"][0]
                    trigger = event_dict["trigger"]
                    start_offset = event_dict["trigger_start_index"]
                    end_offset = event_dict["trigger_start_index"] + len(trigger) - 1
                    token_start_offset, token_end_offset = exoffset2tokenoffset(start_offset, end_offset,
                                                                                tokenized_trigger.token_type_ids,
                                                                                tokenized_trigger.offset_mapping)
                    tokenized_trigger["start_positions"] = token_start_offset
                    tokenized_trigger["end_positions"] = token_end_offset
                TokenInput["trigger_task"] = tokenized_trigger
                # type_task
                if len(example["event_list"]) == 0:
                    continue
                tokenized_type = tokenizer.encode_plus(
                    que.getTypeQ(event_dict["trigger"]),  # 融入问题
                    example["text"],
                    max_length=config.max_seq_length,
                    truncation=True,
                    padding='max_length',
                    # return_tensors="pt",
                    return_offsets_mapping=True
                )
                event_type = event_dict["event_type"]
                tokenized_type["type"] = self.type2id[event_type]['id']
                TokenInput["type_task"] = tokenized_type
                # role task
                RoleTokenList = []
                if self.type2id.__contains__(event_type) is not True:
                    continue
                rolesList = self.type2id[event_type]["roles"].copy()
                for role in event_dict["arguments"]:
                    if role["role"] not in rolesList:
                        continue
                    tokenized_Role = tokenizer.encode_plus(
                        que.getRoleQ(event_dict["trigger"], event_dict["event_type"], role["role"]),  # 融入问题
                        example["text"],
                        max_length=config.max_seq_length,
                        truncation=True,
                        padding='max_length',
                        # return_tensors="pt",
                        return_offsets_mapping=True
                    )
                    rolesList.remove(role["role"])  # 删除提及角色
                    argument = role["argument"]
                    start_offset = role["argument_start_index"]
                    end_offset = role["argument_start_index"] + len(argument) - 1
                    token_start_offset, token_end_offset = exoffset2tokenoffset(start_offset, end_offset,
                                                                                tokenized_Role.token_type_ids,
                                                                                tokenized_Role.offset_mapping)
                    tokenized_Role["start_positions"] = token_start_offset
                    tokenized_Role["end_positions"] = token_end_offset
                    RoleTokenList.append(tokenized_Role)
                # 对于所有在schema但是实际未提及的角色
                for roleName in rolesList:
                    tokenized_Role = tokenizer.encode_plus(
                        que.getRoleQ(event_dict["trigger"], event_dict["event_type"], roleName),  # 融入问题
                        example["text"],
                        max_length=config.max_seq_length,
                        truncation=True,
                        padding='max_length',
                        # return_tensors="pt",
                        return_offsets_mapping=True
                    )
                    token_offset = getStartoffsset(tokenized_Role.token_type_ids)
                    tokenized_Role["start_positions"] = token_offset - 1
                    tokenized_Role["end_positions"] = token_offset - 1
                    RoleTokenList.append(tokenized_Role)
                TokenInput["role_task"] = RoleTokenList
                self.tokenized_examples.append(TokenInput)

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


def collate_fn_trigger_type_join_tassk(batch):  # 联合训练数据集构造
    max_len_trig = max([sum(x['trigger_task']['attention_mask']) for x in batch])
    # seq_lens = torch.tensor([sum(x['trigger_task']["attention_mask"]) for x in batch])
    trig_input_ids = torch.tensor([x['trigger_task']['input_ids'][:max_len_trig] for x in batch])
    trig_token_type_ids = torch.tensor([x['trigger_task']['token_type_ids'][:max_len_trig] for x in batch])
    trig_attention_mask = torch.tensor([x['trigger_task']['attention_mask'][:max_len_trig] for x in batch])
    trig_start_positions = torch.tensor([x['trigger_task']["start_positions"] for x in batch])
    trig_end_positions = torch.tensor([x['trigger_task']["end_positions"] for x in batch])

    max_len_type = max([sum(x['type_task']['attention_mask']) for x in batch])
    # seq_lens = torch.tensor([sum(x['type_task']["attention_mask"]) for x in batch])
    type_input_ids = torch.tensor([x['type_task']['input_ids'][:max_len_type] for x in batch])
    type_token_type_ids = torch.tensor([x['type_task']['token_type_ids'][:max_len_type] for x in batch])
    type_attention_mask = torch.tensor([x['type_task']['attention_mask'][:max_len_type] for x in batch])
    type = torch.tensor([x['type_task']["type"] for x in batch])
    return {
        "trigger_task": {
            "input_ids": trig_input_ids,
            "token_type_ids": trig_token_type_ids,
            "attention_mask": trig_attention_mask,
            "start_positions": trig_start_positions,
            "end_positions": trig_end_positions
        },
        "type_task": {
            "input_ids": type_input_ids,
            "token_type_ids": type_token_type_ids,
            "attention_mask": type_attention_mask,
            "type": type
        }
        # "all_seq_lens": seq_lens
    }


def collate_fn_trigger_task(batch):
    max_len = max([sum(x['trigger_task']['attention_mask']) for x in batch])
    # seq_lens = torch.tensor([sum(x['trigger_task']["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['trigger_task']['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['trigger_task']['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['trigger_task']['attention_mask'][:max_len] for x in batch])
    all_start_positions = torch.tensor([x['trigger_task']["start_positions"] for x in batch])
    all_end_positions = torch.tensor([x['trigger_task']["end_positions"] for x in batch])
    return {
        "input_ids": all_input_ids,
        "token_type_ids": all_token_type_ids,
        "attention_mask": all_attention_mask,
        "start_positions": all_start_positions,
        "end_positions": all_end_positions
        # "all_seq_lens": seq_lens
    }


def collate_fn_type_task(batch):
    max_len = max([sum(x['type_task']['attention_mask']) for x in batch])
    # seq_lens = torch.tensor([sum(x['type_task']["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['type_task']['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['type_task']['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['type_task']['attention_mask'][:max_len] for x in batch])
    all_type = torch.tensor([x['type_task']["type"] for x in batch])

    return {
        "input_ids": all_input_ids,
        "token_type_ids": all_token_type_ids,
        "attention_mask": all_attention_mask,
        "type": all_type
    }


def collate_fn_role_task(batch):
    totalRoleList = []
    for x in batch:
        for y in x['role_task']:
            totalRoleList.append(y)
    max_len = max([sum(x['attention_mask']) for x in totalRoleList])
    # seq_lens = torch.tensor([sum(x["attention_mask"]) for x in totalRoleList])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in totalRoleList])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in totalRoleList])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in totalRoleList])
    all_start_positions = torch.tensor([x["start_positions"] for x in totalRoleList])
    all_end_positions = torch.tensor([x["end_positions"] for x in totalRoleList])
    return {
        "input_ids": all_input_ids,
        "token_type_ids": all_token_type_ids,
        "attention_mask": all_attention_mask,
        "start_positions": all_start_positions,
        "end_positions": all_end_positions
        # "all_seq_lens": seq_lens
    }


if __name__ == '__main__':
    from model.MRC_model import BertForQA

    config = Config()
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(config.model_dir, config.model_name))
    path = os.path.join(config.RootPath, "data/test.json")
    print(path)
    dataset = DuEEEventDataset(config,
                               # data_path=config.dev_path,
                               data_path=path,
                               schema_path=config.schema_path,
                               tokenizer=tokenizer)
    print("len of dataset {}".format(len(dataset)))
    for data  in  dataset:
        print(data["trigger_task"]["offset_mapping"])
    test_iter = DataLoader(dataset,
                           shuffle=True,
                           batch_size=10,
                           collate_fn=collate_fn_role_task,
                           num_workers=20)
    print("len of loader {}".format(len(test_iter)))
    for index, batch in enumerate(test_iter):
        for i in range(len(batch["input_ids"])):
            print(batch["input_ids"][i])
            print(tokenizer.decode(batch["input_ids"][i]))
            print(batch["token_type_ids"][i])
            print(batch["start_positions"][i])
            print(batch["end_positions"][i])
