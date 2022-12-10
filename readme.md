# EEMRC_bert 基于阅读理解的事件抽取（2022.6.5整理版）
《阅读理解式事件抽取》 
过去手工训练写的太乱了，现在整理并统一写了入口函数，添加了一点命令行参数，现在介绍如下， 
## 主要项目依赖 
```angular2
jsonnet	0.17.0	
jsonschema	2.6.0	
lmdb	1.2.1	
numpy	1.19.5	
packaging	20.9	
pandas	0.25.2	
pip	21.1.1	
six	1.16.0	
sklearn	0.0	
tokenizers	0.10.3	
toml	0.10.2	
toolz	0.11.1	
torch	1.8.1	
torchvision	0.9.1	
tqdm	4.61.0	
transformers	4.5.1	
```
项目目录  
在10.99.4.36机器上的`/home/BWQ/EEMRC_bert/`下  
环境为EEQA 在dell账号下执行`conda info -e`即可搜索到
目录介绍
```
    data
        dev.json    模型评估的样本
        train.json  训练文件 来自DUEE1.0(目前官方取消开源，不要轻易向外传播以免存在法律问题)
        test.json   示例文件
    dataProcess
        dataset.py  dataset的构造和dataloader的调用函数
    DuEE            DUEE1.0数据
    model
        checkpoints                 分为三个子文件夹，分别保存各自任务训练的模型
        chinese-roberta-wwm-ext     哈工大开源Roberta
        RoBERTa_zh_L12_PyTorch      某热心网友训练并开源的RoberTa,居然似乎比哈工好点
        MRC_model.py                模型结构定义、前向计算定义
    config.py       各参数的配置项，注意非路径参数会被命令行参数覆盖
    getQuestion.py  问题定义
    run.py          项目训练、示例入口
    train_eval.py   训练，评估过程定义
    readme.md       本介绍文件
```
## 快速开始
首先在huggingface官网上下载某Bert模型或RoberTa模型至model目录下  
保证以下目录结构
```
EEMRC_bert
    model
        RoBERTa_zh_L12_PyTorch 
            config.json
            pytorch_model.bin
            vocab.txt

```
hugginface模型官网 https://huggingface.co/models  
作者使用的模型链接   
热心网友：https://github.com/brightmart/roberta_zh  
哈工Roberta: https://github.com/ymcui/Chinese-BERT-wwm

训练命令
```
conda activate EEQA #切换到指定环境下
在10.99.4.36的/home/BWQ/EEMRC_bert目录下执行下面命令
python run.py 即可在默认参数下进行trigger_task训练
```
命令行可选参数和解释如下
```
optional arguments:
  -h, --help            show this help message and exit
  --bert_type           BERT_TYPE
                        RoBERTa_zh_L12_PyTorch / chinese-roberta-wwm-ext
  --gpu_ids             GPU_IDS     which gpu ids to use, "1, 2, 3" for multi gpu
  --mode MODE           train / example

  --task_type           TASK_TYPE
                        trigger / type / role
  --max_seq_len         MAX_SEQ_LEN 默认256
  --batch_size          BATCH_SIZE 默认10 
  --epoch               EPOCH 默认10
```
比如希望在GPU2上训练role_task任务
```
python run.py --gpu_ids 2 --task_type role
```
同理trigger、type、任务
```
python run.py --gpu_ids 2 --task_type trigger
python run.py --gpu_ids 2 --task_type type
```

训练完成后模型完整事件抽取示例
```
python run.py --gpu_ids 2 --mode example
```
执行后效果如下
```
start2 end3
trigger is 裁员
type is 组织关系-裁员
type 组织关系-裁员 schema is ['时间', '裁员方', '裁员人数']
role 时间 is False
start0 end1
role 裁员方 is 雀巢
start4 end8
role 裁员人数 is 4000人
```
由上可见对“雀巢裁员4000人：时代抛弃你时，连招呼都不会打！”这句话完成了完整的事件抽取
其示例代码在run.py下 先加载各任务模型，再进行全流程抽取
```python
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
```
简单提一下，其实本项目trigger_type联合模型数据并无突出优势，但是理论上本项目模型的结构可以直接玩端到端模型，  

但是炼丹玄学太多了，师兄精力有限，师弟师妹加油成功了效果不错的话 还能搞一篇  
题目就叫《基于预训练模型的MRC式端到端事件抽取》，端到端玄学训练再加点UDA当小料，发个中等会或国内期刊应该没问题（2022.4.25）)  
