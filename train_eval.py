import torch
import time
import torch.nn as nn
from sklearn import metrics
import numpy as np
from config import log, Config
from tqdm import tqdm, trange
import os
from dataProcess.dataset import DuEEEventDataset, DataLoader, collate_fn_role_task, collate_fn_trigger_task, \
    collate_fn_type_task
import timeit
# from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from model.MRC_model import BertForQA, BertForEventTriggerAndType

logger = log.logger


def to_list(tensor):
    return tensor.detach().cpu().tolist() if not tensor == torch.Size([0]) else None


def saveModel(model, path, name):
    output_path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), output_path)
    logger.info("Saving %s model to %s", name, output_path)


def train(config: Config, model, train_loader, tokenizer, task="type_task"):
    global_step = 0
    best_acc = -1
    tr_loss, logging_loss = 0.0, 0.0
    t_total = len(train_loader) * config.nums_epochs // config.gradient_accumulation_steps
    # if task == "type_task":
    #     opt_list = list(model.type_linear.parameters())
    #     lr = 1e-3
    # else:
    opt_list = list(model.parameters())
    # print(opt_list)
    lr = config.learning_rate
    optimizer = torch.optim.AdamW(opt_list, lr=lr, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)

    criterion = nn.CrossEntropyLoss().to(config.device)
    # tb_writer = SummaryWriter()
    model.zero_grad()
    print("load dev set")
    dev_dataset = DuEEEventDataset(config,
                                   data_path=config.dev_path,
                                   schema_path=config.schema_path,
                                   tokenizer=tokenizer)
    for train_iterator in range(int(config.nums_epochs)):  # 每个轮次
        epoch_iterator = tqdm(train_loader, desc="Epoch:{} Iteration".format(train_iterator), position=0)
        # with tqdm(enumerate(train_loader), total=len(train_loader),desc="Epoch:{} Iteration".format(train_iterator), position=0) as pbar:
        # for step, batch in pbar:
        for step, batch in enumerate(epoch_iterator):
            global_step += 1
            model.train()
            # print(len(batch))
            for key in batch.keys():
                batch[key] = batch[key].to(config.device)
            if task == "type_task":
                # types = batch["type"]
                output = model(**batch, task_type="type_task")
                # loss = criterion(output["type_logits"], types)
            else:
                output = model(**batch)
            loss = output["loss"]
            tr_loss += loss.item()
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                epoch_iterator.write(
                    "step:{} learn rate:{} loss:{}".format(step + 1, scheduler.get_last_lr(), loss.item()))
                model.zero_grad()

            # Log metrics
            if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                results, lossV = evaluate(config, model, dev_dataset, task=task)
                epoch_iterator.write('F1 {}:loss {}'.format(results, lossV))
                acc = results
                if acc > best_acc and acc > 0.8:
                    output_dir = os.path.join(config.output_dir, task)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    saveModel(model, output_dir,
                              'best_model-{}-F1-{}-loss-{}.pkl'.format(task, round(results, 2), round(lossV, 3)))
                    # logger.info("Saving best model to %s", output_path)
                    best_acc = acc
        if config.max_steps > 0 and global_step > config.max_steps:
            train_iterator.close()
            epoch_iterator.close()
            break
    results, lossV = evaluate(config, model, dev_dataset, task=task)
    output_dir = os.path.join(config.output_dir, task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('F1 {}:loss {}'.format(results, lossV))
    acc = results
    if acc > best_acc and acc > 0.8:
        saveModel(model, output_dir,
                  'best_model-{}-F1-{}-loss-{}.pkl'.format(task, round(results, 2), round(lossV, 3)))
        # logger.info("Saving best model to %s", output_path)
    saveModel(model, output_dir, 'checkpoint-{}-step-{}.pkl'.format(task, global_step))


def test(config, model, test_iter):
    pass


def same(a, b):
    if isinstance(a, list):
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True
    else:
        return a == b


def compute_f1(num_same: int, pred_toks: int, gold_toks: int):
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / pred_toks
    recall = 1.0 * num_same / gold_toks
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(config, model, dataset, prefix="", task="type_task"):
    # eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    if task == "trigger_task":
        eval_dataloader = DataLoader(dataset,
                                     shuffle=False,
                                     batch_size=config.batch_size,
                                     collate_fn=collate_fn_trigger_task,
                                     num_workers=20)
    elif task == "type_task":
        eval_dataloader = DataLoader(dataset,
                                     shuffle=False,
                                     batch_size=config.batch_size,
                                     collate_fn=collate_fn_type_task,
                                     num_workers=20)
    else:
        eval_dataloader = DataLoader(dataset,
                                     shuffle=False,
                                     batch_size=config.batch_size,
                                     collate_fn=collate_fn_role_task,
                                     num_workers=20)
    # Eval!
    logger.info("***** Running evaluation {} {}*****".format(prefix, task))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", config.batch_size)

    all_results = []
    all_pred = []
    all_aim = []
    start_time = timeit.default_timer()
    numTrue = 0
    Totalnum = 0
    TotalLoss = 0.0
    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0):
        model.eval()

        for key in batch.keys():
            batch[key] = batch[key].to(config.device)
        with torch.no_grad():
            if task == "type_task":
                outputs = model(**batch, task_type="type_task")
                pred_type = outputs["type_logits"].argmax(dim=1)
                type = batch["type"].tolist()
                pred_type = pred_type.tolist()
                for i in range(len(type)):
                    # print("type:{} pred_type:{}".format(type[i], pred_type[i]))
                    if pred_type == type:
                        numTrue = numTrue + 1
                    Totalnum = Totalnum + 1

            else:
                outputs = model(**batch)
                start = batch["start_positions"].tolist()
                end = batch["end_positions"].tolist()
                pred_start = outputs["start_logits"].argmax(dim=1).tolist()
                pred_end = outputs["end_logits"].argmax(dim=1).tolist()
                if len(start) != len(pred_start):
                    print("error! check the word")
                # print("len in batch is {}".format(len(start)))
                for i in range(len(start)):
                    # print("start:{} pred:{} end:{} pred:{}".format(start[i], pred_start[i], end[i], pred_end[i]))
                    if start[i] == pred_start[i] and end[i] == pred_end[i]: # 严格匹配标准
                        numTrue = numTrue + 1
                        # print("True!:{}".format(numTrue))
                    Totalnum = Totalnum + 1
                TotalLoss = TotalLoss + outputs["loss"].item()

    TotalLoss = TotalLoss / len(eval_dataloader)
    acc = 1.0 * numTrue / Totalnum
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    logger.info("  acc is %f loss is %f ", acc, TotalLoss)
    return acc, TotalLoss
