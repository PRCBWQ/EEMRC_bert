from config import Config
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, BertConfig, BertTokenizerFast
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import os
import torch


class BertForQA(nn.Module):
    def __init__(self, config):  # 注意这里config是自定义的 不是bert config
        super(BertForQA, self).__init__()
        self.BertModule = BertForQuestionAnswering.from_pretrained(
            os.path.join(config.model_dir, config.model_name), output_hidden_states=True)
        self.type_linear = nn.Linear(config.hidden_size, config.num_type)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        outputs = self.BertModule(**inputs)

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        hidden_states = outputs['hidden_states'][0]
        type_logits = self.type_linear(hidden_states[:, 0, :])
        # type_prob = self.softmax(type_logits)
        final = {}
        if outputs.__contains__("loss"):
            loss = outputs.get('loss', torch.tensor([]))
            final["loss"] = loss
        final["start_logits"] = start_logits
        final["end_logits"] = end_logits
        final["type_logits"] = type_logits
        # print(type_prob.shape)
        return final


class BertForEventTriggerAndType(nn.Module):
    def __init__(self, config: Config):
        super(BertForEventTriggerAndType, self).__init__()
        self.bert = BertModel.from_pretrained(os.path.join(config.model_dir, config.model_name))
        self.type_outputs = nn.Linear(self.bert.config.hidden_size, config.num_type)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None,
                type=None,
                task_type="trigger_task"):
        final = {}
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        # 分开trigger_task和type_task
        if task_type != "type_task":
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            final["start_logits"] = start_logits
            final["end_logits"] = end_logits
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                final["loss"] = total_loss

        elif task_type == "type_task":
            type_logits = self.type_outputs(pooled_output)
            final["type_logits"] = type_logits
            if type is not None:
                loss_fct = CrossEntropyLoss()
                total_loss = loss_fct(type_logits, type)
                final["loss"] = total_loss
        return final


config = Config()

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(config.model_dir, config.model_name))
    print(os.path.join(config.model_dir, config.model_name, 'config.json'))
    # modelConfig = BertConfig.from_pretrained(os.path.join(config.model_dir, config.model_name, 'config.json'))
    # model = BertForQuestionAnswering.from_pretrained(os.path.join(config.model_dir, config.model_name), output_hidden_states=True)
    # model = BertForQuestionAnswering.from_pretrained(os.path.join(config.model_dir, config.model_name))
    # model = BertForQA(config)
    model = BertForEventTriggerAndType(config)
    question, text = "小王的职业是什么?", "小王是个学生"
    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    inputs = tokenizer.encode_plus(question, text, return_tensors='pt', return_offsets_mapping=True)
    inputs = tokenizer.encode_plus(question, text, max_length=30, truncation=True, padding='max_length',
                                   return_tensors='pt', return_offsets_mapping=True)
    # inputs = tokenizer.encode_plus(question, text, max_length=30, truncation=True, padding='max_length',
    # return_offsets_mapping=True)
    print(inputs)

    # print("inputsize ", inputs.input_ids[0].size())
    from dataProcess.dataset import exoffset2tokenoffset, tokenoffset2exoffset

    print("token offset %d \n" % exoffset2tokenoffset(4, inputs.token_type_ids, inputs.offset_mapping))

    '''
    for i in range(inputs.input_ids[0].size()[0]):
        print(inputs.input_ids[0][i])
        if inputs.input_ids[0][i] == 0:
            break
            '''
    with torch.no_grad():
        # outputs = model(**inputs)# 自建模型直接inputs， BertForQuestionAnswering需要**inputs来展开
        offset_map = inputs.offset_mapping
        inputs.pop('offset_mapping')
        outputs = model(**inputs, task_type="trigger_task")
        print(outputs)
        # print(outputs.hidden_states[1])
        # print(outputs.hidden_states[12])
        answer_start_index = outputs["start_logits"].argmax()
        answer_end_index = outputs["end_logits"].argmax()
        print(answer_start_index, answer_end_index)
        # print(inputs.input_ids)
        org_start = tokenoffset2exoffset(15, inputs.token_type_ids, offset_map,isLeft=True)
        org_end = tokenoffset2exoffset(16, inputs.token_type_ids, offset_map, isLeft=False)
        print("原始解码{}".format(text[org_start:org_end]))
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        print(predict_answer_tokens)
        print(tokenizer.decode(predict_answer_tokens))

        target_start_index = torch.tensor([11])
        target_end_index = torch.tensor([14])
        inputs["start_positions"] = target_start_index
        inputs["end_positions"] = target_end_index
        # outputs = model(inputs, start_positions=target_start_index,end_positions=target_end_index)  # BertForQuestionAnswering需要
        outputs = model(**inputs)
        loss = outputs["loss"]
        print(round(loss.item(), 2))
        answer_start_index = outputs["start_logits"].argmax()
        answer_end_index = outputs["end_logits"].argmax()
        print(answer_start_index, answer_end_index)
