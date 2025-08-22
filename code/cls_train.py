import os
import sys
import json
import math
import copy
import nltk
import copy
import random
import pickle
import pymysql
import sklearn
import datetime
import argparse
import logging
import pandas as pd
import scipy.stats as stats
import prettytable as pt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from clickhouse_driver import Client

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", default='pubmed', type=str)
parser.add_argument("-m", "--model_name", default="Qwen2.5-14B-Instruct", type=str)
parser.add_argument("-mp", "--model_name_parent", default="Qwen-Instruct", type=str)
parser.add_argument("-r", "--lora_rank", default=32, type=int)
parser.add_argument("-a", "--lora_alpha", default=32, type=int)
parser.add_argument("-l", "--limit_num", default=500, type=int)
args = parser.parse_args()

import torch
import transformers
from torch import nn
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import  TrainingArguments, Trainer, GenerationConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from adapters import AutoAdapterModel

from datasets import load_dataset, Dataset, load_from_disk
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate

import utils


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


DOMAIN = args.domain
DOMAIN_DICT = {
    "pubmed": "biomedical",
    "physics": "physics",
    "chemistry": "chemistry",
    "mathematics": "mathematics",
}
DOMAIN_NAME = DOMAIN_DICT[DOMAIN]

"""数据存储路径"""
BASE_DIR = "/public/home/lab8/project/PaperOriginality/code/hsz"
data_dir = os.path.join(BASE_DIR, "data")
BALANCED_DATA_PATH = os.path.join(data_dir, f"DI_SCORE_PREDICTION_DATA/{DOMAIN.upper()}_FUTURE")  # 训练数据源路径
MODEL_SAVE_PATH = f"./outputs/{DOMAIN.upper()}_FUTURE" # 模型存储路径
RESULT_SAVE_PATH = f"./results/{DOMAIN.upper()}_FUTURE" # 测试结果储存路径
metric = evaluate.load("./evaluate-main/metrics/accuracy")

""" 模型路径 """
model_hub_dir = os.path.join(BASE_DIR, "models")
model_base_path = os.path.join(model_hub_dir, f"{args.model_name_parent}/{args.model_name}")
model_save_path = os.path.join(MODEL_SAVE_PATH, os.path.basename(model_base_path))
result_save_path = os.path.join(RESULT_SAVE_PATH, os.path.basename(model_base_path))
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(result_save_path, exist_ok=True)
logger.info(f"正在使用模型基座: {model_base_path}")

"""Tokenizer"""
if not os.path.exists(model_base_path):
    logger.info(f"模型基座不存在: {model_base_path}")
tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
max_length = 768 * (1 + 1) + 100

"""分类设置"""
num_labels = 3  # 三分类 (>, <, =)
quantify = True


def make_prompt(title_A, abstract_A, title_C, abstract_C, max_abstract_words=768):
    """ 生成指令 """
    abstract_A = " ".join(abstract_A.split(" ")[: max_abstract_words]).strip()
    abstract_C = " ".join(abstract_C.split(" ")[: max_abstract_words]).strip()

    prompt_task = (
        f"As an expert in {DOMAIN_NAME} field, "
        "you are required to meticulously scrutinize the titles and abstracts of both Paper A and Paper B in order to ascertain whether Paper A is superior in quality to Paper B, comparable in quality to Paper B, or inferior in quality to Paper B."
        "\n\n"
    )
    prompt_A = (
        "Title of Paper A:\n"
        f"{title_A}\n"
        "Abstract of Paper A:\n"
        f"{abstract_A}\n"
    )
    prompt_C = (
        "Title of Paper B:\n"
        f"{title_C}\n"
        "Abstract of Paper B:\n"
        f"{abstract_C}\n"
    )
    prompt = prompt_task + prompt_A + prompt_C
    return prompt.strip()


def tokenize(batch):
    titles_A = batch['titles_A']
    abstracts_A = batch["abstracts_A"]
    titles_C = batch["titles_C"]
    abstracts_C = batch["abstracts_C"]
    prompts = list()
    for title_A, abstract_A, title_C, abstract_C in zip(titles_A, abstracts_A, titles_C, abstracts_C):
        prompt = make_prompt(title_A, abstract_A, title_C, abstract_C)
        # chat = [{"role": "user", "content": "{}".format(prompt)}]
        # prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(prompt)
    return tokenizer(prompts, max_length=max_length, truncation=True)


def split_dataset():
    """ 本地路径读取数据, 生成训练集, 验证集, 验证集 """
    logger.info(f"随机划分数据: {BALANCED_DATA_PATH}")
    balanced_set = utils.read_json(os.path.join(BALANCED_DATA_PATH, "data_v2/balanced_set2.json"))

    total_size = len(balanced_set)
    train_size = int(total_size * 0.9)
    valid_size = int(total_size * 0.05)
    test_size  = total_size - train_size - valid_size
    
    if type(balanced_set) == list:
        index = np.arange(total_size)
    elif type(balanced_set) == dict:
        index = list(balanced_set.keys())
    random.shuffle(index)

    train_index = index[: train_size]
    valid_index = index[train_size: train_size + valid_size]
    test_index  = index[train_size + valid_size: ]

    train_dataset = {
        'pids_A': list(), 
        'pids_C': list(),
        'titles_A': list(), 
        'abstracts_A': list(), 
        'titles_C': list(), 
        'abstracts_C': list(), 
        'labels': list(), 
        'level_A': list(),
        'level_C': list(),
    }
    valid_dataset = {
        'pids_A': list(), 
        'pids_C': list(),
        'titles_A': list(), 
        'abstracts_A': list(), 
        'titles_C': list(), 
        'abstracts_C': list(), 
        'labels': list(), 
        'level_A': list(),
        'level_C': list(),
    }
    test_dataset = {
        'pids_A': list(), 
        'pids_C': list(),
        'titles_A': list(), 
        'abstracts_A': list(), 
        'titles_C': list(), 
        'abstracts_C': list(), 
        'labels': list(), 
        'level_A': list(),
        'level_C': list(),
    }
    dataset = {
        "train_dataset": train_dataset, 
        "valid_dataset": valid_dataset, 
        "test_dataset": test_dataset
    }
    relation_dict = {">": 2, "=":1, "<": 0}
    tmp_label_counts = list()
    for tmp_idx, tmp_dataset in zip([train_index, valid_index, test_index], [train_dataset, valid_dataset, test_dataset]):
        tmp_label_count = dict()
        for idx in tmp_idx:
            paper = balanced_set[idx]
            #
            tmp_dataset['pids_A'].append(paper["pid1"])
            tmp_dataset['titles_A'].append(paper['title1'])
            tmp_dataset['abstracts_A'].append(paper['abstract1'])
            #
            tmp_dataset['pids_C'].append(paper["pid2"])
            tmp_dataset['titles_C'].append(paper['title2'])
            tmp_dataset['abstracts_C'].append(paper['abstract2'])
            #
            label = relation_dict[paper["relation"]]
            tmp_dataset['labels'].append(label)
            tmp_dataset['level_A'].append(paper['level1'])
            tmp_dataset['level_C'].append(paper['level2'])

            label_count = tmp_label_count.get(label, 0)
            label_count += 1
            tmp_label_count[label] = label_count
        tmp_label_counts.append(tmp_label_count)

    # 样本量统计
    tb = pt.PrettyTable()
    tb.title = "数据集类别分布"
    tb.field_names = ["类别", "A<C(0)", "A=C(1)", "A>C(2)"]
    tb.add_rows([
        ["训练集", tmp_label_counts[0][0], tmp_label_counts[0][1], tmp_label_counts[0][2]],
        ["验证集", tmp_label_counts[1][0], tmp_label_counts[1][1], tmp_label_counts[1][2]],
        ["测试集", tmp_label_counts[2][0], tmp_label_counts[2][1], tmp_label_counts[2][2]],
    ])
    logger.info(tb)
    # 数据存储
    for data_name in dataset:
        tmp = Dataset.from_dict(dataset[data_name])
        tmp.save_to_disk(os.path.join(BALANCED_DATA_PATH, f"classification/{data_name}"))


def load_dataset():
    """ 读取训练数据 """
    train_dataset_path = os.path.join(BALANCED_DATA_PATH, "classification/train_dataset")
    valid_dataset_path = os.path.join(BALANCED_DATA_PATH, "classification/valid_dataset")
    test_dataset_path  = os.path.join(BALANCED_DATA_PATH, "classification/test_dataset")
    if not os.path.exists(train_dataset_path) or not os.path.exists(valid_dataset_path) or not os.path.exists(test_dataset_path):
        split_dataset()

    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    test_dataset  = test_dataset.map(tokenize, batched=True)

    # Token of Input Statistics
    tb = pt.PrettyTable()
    tb.title = "Token Count Statistics"
    tb.field_names = ["Class", "Mean", "Std", "Medium", "25th per", "50th per", "75th per", "Min", "Max"]
    for name_i, set_j in zip(["Valid", "Test"], [valid_dataset, test_dataset]):
        set_input_ids = set_j['input_ids']
        text_token_num = np.array([len(set_input_ids[i]) for i in range(len(set_input_ids))])
        tb.add_row([
            name_i,
            int(np.mean(text_token_num)), 
            int(np.std(text_token_num)), 
            np.median(text_token_num), 
            np.percentile(text_token_num, 25),
            np.percentile(text_token_num, 50),
            np.percentile(text_token_num, 75),
            min(text_token_num),
            max(text_token_num)
        ])
    logger.info(tb)
    return train_dataset, valid_dataset, test_dataset


def load_model_with_qlora():
    """ https://huggingface.co/docs/peft/quicktour """
    if quantify:
        # 模型量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels)

    # model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 添加LoRA配置
    # target_modules = ["q_proj", 'k_proj', "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",]
    target_modules = ["q_proj", 'k_proj', "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS", # task_type = "SEQ_CLS", "CAUSAL_LM"
    ) 
    model = get_peft_model(model, config)

    logger.info(model)
    model.print_trainable_parameters()
    return model


def load_model_with_qlora_continue(peft_path=""):
    """ 继续训练qlora # https://github.com/huggingface/peft/issues/184 """
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)
    logger.info("载入模型路径 (继续训练):", peft_path)

    if quantify:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 载入lora权重
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch.float16, is_trainable=True)
    # 载入分类头权重
    score_weights = torch.load(os.path.join(peft_path, "score.original_module.pt"), map_location='cpu')
    model.score.original_module.load_state_dict(score_weights)
    # model.cuda() # 影响qlora模型报错
    logger.info(model.score.original_module)
    
    model.train()
    logger.info(model)
    model.print_trainable_parameters()
    return model, final_ck


def load_trained_model(model_base_path, model_save_path, num_labels, quantify=True, peft_path=''):
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)
    logger.info(f"load_trained_model 载入测试模型的路径: {peft_path}")

    if quantify:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    
    tokenizer_score_model = AutoTokenizer.from_pretrained(peft_path)
    # model.resize_token_embeddings(len(tokenizer_score_model))
    model.config.pad_token_id = tokenizer_score_model.pad_token_id
    model.config.use_cache = False

    # 载入lora权重
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch.float16, )
    # 载入分类头权重
    score_weights = torch.load(os.path.join(peft_path, "score.original_module.pt"), map_location='cpu')
    model.score.original_module.load_state_dict(score_weights)
    # model.cuda() # 影响qlora模型报错
    model.eval()
    return model, tokenizer_score_model


class SaveScoreCallback(TrainerCallback):
    """ 模型库bug, 未能正确存储分类头"""
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
        ):
        fname = f"{model_save_path}/checkpoint-{state.global_step}/score.original_module.pt"
        torch.save(self.model.model.score.original_module.state_dict(), fname)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(batch_size, gradient_accumulation_steps, num_train_epochs):
    """ https://www.e2enetworks.com/blog/a-step-by-step-guide-to-fine-tuning-the-mistral-7b-llm """
    # 载入训练数据
    train_dataset, valid_dataset, test_dataset = load_dataset()
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")
    exclude_keys = ['pids_A', 'pids_C', 'titles_A', 'abstracts_A', 'titles_C', 'abstracts_C', "level_A", "level_C"]
    train_dataset = train_dataset.remove_columns(exclude_keys)
    valid_dataset = valid_dataset.remove_columns(exclude_keys)
    test_dataset = test_dataset.remove_columns(exclude_keys)

    model = load_model_with_qlora()

    training_args = TrainingArguments(
        output_dir=model_save_path,
        fp16=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        learning_rate=2e-5,
        load_best_model_at_end=False,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        optim="paged_adamw_32bit",
        max_grad_norm=50,
        weight_decay=0.01,
        disable_tqdm=False,
        report_to='tensorboard',
        seed=9527,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.add_callback(SaveScoreCallback(model))
    trainer.train()


def train_model_continue(batch_size, gradient_accumulation_steps):
    """训练断开后, 继续训练"""
    train_dataset, valid_dataset, test_dataset = load_dataset()
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")
    exclude_keys = ['pids_A', 'pids_C', 'titles_A', 'abstracts_A', 'titles_C', 'abstracts_C']
    train_dataset = train_dataset.remove_columns(exclude_keys)
    valid_dataset = valid_dataset.remove_columns(exclude_keys)
    test_dataset = test_dataset.remove_columns(exclude_keys)
    
    model, final_ck = load_model_with_qlora_continue()
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        fp16=True,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=2000,
        learning_rate=2e-5,
        load_best_model_at_end=False,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        optim="paged_adamw_32bit",
        max_grad_norm=50,
        weight_decay=0.01,
        disable_tqdm=False,
        report_to='tensorboard',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.add_callback(SaveScoreCallback(model))

    logger.info("继续训练:", os.path.join(model_save_path, f"{final_ck}"))
    # trainer.train(resume_from_checkpoint=True)
    trainer.train(resume_from_checkpoint=os.path.join(model_save_path, f"{final_ck}"))


def compute_score(output_softmax):
    """ 根据分类的logits, 经过softmax, 计算0-100的得分""" 
    batch_size, num_cls = output_softmax.shape

    pred_scores = list()
    for probs in output_softmax:
        pred_score = 0
        p0, p1, p2 = probs
        base_score = 60 * (p1 + p2)
        #
        if p1 + p2 > 0:
            p1_normal = (p1 / (p1 + p2))
            p2_normal = (p2 / (p1 + p2))
        else:
            p1_normal = 0
            p2_normal = 0
        # 
        consolidate_score  = 30 * p1_normal
        disrupt_score = 40 * p2_normal
        plus_score =  p1_normal * consolidate_score + p2_normal * disrupt_score
        # 
        pred_score = base_score + plus_score
        pred_scores.append(pred_score)
    return pred_scores


def test_model(eval_set, batch_size=1, peft_path=''):
    """"""
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    """Load Model"""
    model, tokenizer = load_trained_model(model_base_path, model_save_path, num_labels, quantify, peft_path=peft_path)

    """Load Dataset"""
    train_dataset, valid_dataset, test_dataset = load_dataset()
    llama_input_features = [feature for feature in train_dataset.features if feature not in ['input_ids', 'attention_mask', 'labels']]

    checkpoint_name = os.path.basename(peft_path)
    eval_save_path = os.path.join(result_save_path, checkpoint_name)
    os.makedirs(eval_save_path, exist_ok=True)

    if eval_set == "train":
        dataloader = DataLoader(train_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        eval_save_path = os.path.join(eval_save_path, "eval_on_train.json")
        dataset = train_dataset
    elif eval_set == "valid":
        dataloader = DataLoader(valid_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        eval_save_path = os.path.join(eval_save_path, "eval_on_valid.json")
        dataset = valid_dataset
    elif eval_set == "test":
        dataloader = DataLoader(test_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        eval_save_path = os.path.join(eval_save_path, "eval_on_test.json")
        dataset = test_dataset
    else:
        print("test_model: 测试数据集为空")
        return -1

    examine = dict()
    pred_labels = list()
    true_labels = list()
    pred_scores = list()
    count = 0
    for batch in tqdm(dataloader):
        count += 1
        labels = batch['labels']
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        # 分类
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # 分类标签
        labels = labels.detach().cpu().numpy()
        logits = output['logits'].detach().cpu().numpy()
        true_labels += list(labels)
        pred_labels += list(np.argmax(logits, axis=-1))
        
        # 预测概率转换成得分
        output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
        scores = compute_score(output_softmax)
        pred_scores += scores

        # if count >= 500:
        #     matrix_test = confusion_matrix(true_labels, pred_labels)
        #     report_on_test = classification_report(true_labels, pred_labels, digits=4)
        #     logger.info(matrix_test)
        #     logger.info(report_on_test)
        #     break

    evaluate_result = dict()
    pids_A = dataset['pids_A']
    pids_C = dataset['pids_C']
    for i, (pid_A, pid_C) in enumerate(zip(pids_A, pids_C)):
        if i >= len(pred_labels):
            logger.info("test_model: 提前结束模型测试过程")
            break
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        pred_score = pred_scores[i]
        pair = pid_A + "\n" + pid_C
        evaluate_result[pair] = dict()
        evaluate_result[pair]["pred_label"] = str(pred_label)
        evaluate_result[pair]["true_label"] = str(true_label)
        evaluate_result[pair]["pred_score"] = str(int(pred_score))
    utils.save_json(evaluate_result, eval_save_path)


def print_test_model_results(eval_set='test', peft_path=''):
    """打印分类精度"""
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)
    
    logger.info(f"Checkpoint path: {peft_path}")

    checkpoint_name = os.path.basename(peft_path)
    eval_save_path  = os.path.join(result_save_path, checkpoint_name)

    if eval_set == "test":
        eval_res = utils.read_json(os.path.join(eval_save_path, "eval_on_test.json"))
    elif eval_set == "valid":
        eval_res = utils.read_json(os.path.join(eval_save_path, "eval_on_valid.json"))
    else:
        eval_res = utils.read_json(os.path.join(eval_save_path, "eval_on_train.json"))

    true_labels = list()
    pred_labels = list()
    pred_scores = dict()
    for pid in eval_res:
        true_label = eval_res[pid]['true_label']
        pred_label = eval_res[pid]['pred_label']
        pred_score = int(eval_res[pid]['pred_score'])
        if true_label not in pred_scores:
            pred_scores[true_label] = [pred_score]
        else:
            pred_scores[true_label].append(pred_score)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    # 混淆矩阵
    matrix = confusion_matrix(true_labels, pred_labels)
    # 分类精度
    report = classification_report(true_labels, pred_labels, digits=4)
    report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    logger.info(matrix)
    logger.info(report)

    tb = pt.PrettyTable()
    tb.title = "原创性得分分布"
    tb.field_names = ["类别", "一般类(A<B)", "巩固类(A=B)", "颠覆类(A>B)"]
    tb.add_rows([
        ["均值", round(np.mean(pred_scores['0']), 2), round(np.mean(pred_scores['1']), 2), round(np.mean(pred_scores['2']), 2)],
        ["标准差", round(np.std(pred_scores['0']), 2), round(np.std(pred_scores['1']), 2), round(np.std(pred_scores['2']), 2)],
    ])
    # logger.info(tb)
    results = {
        "classification_report": report_dict["macro avg"]
    }
    return results


def statistics_evaluate_main(samples_dir):
    """
    统计测试结果:
    (1) 随机论文测评
    (2) 获奖论文测评
    (3) 推荐论文测评
    """
    # (1) 随机论文测评
    if DOMAIN == "mathematics":
        DOMAIN_NORMAL = "math"
    else:
        DOMAIN_NORMAL = DOMAIN
    random_papers = utils.read_json(os.path.join(samples_dir, f"collect_papers_by_random/random_samples_{DOMAIN_NORMAL}2.json"))
    random_res_dict = utils.read_json(os.path.join(samples_dir, f"collect_papers_by_random/random_samples_results_{DOMAIN_NORMAL}2.json"))

    score_rate = 35
    score_plus = 1

    tb_random_dict = {
        "Pubt": list(),
        "isTop5%": list(),
        "Q": list(),
        "IF": list(),
        "CC": list(),
        "average_score":list(),
        "average_score_before":list(),
        "average_score_after":list(),
        "average_std":list()
    }
    for year in random_papers:
        pids_info = random_papers[year]
        for pid in pids_info:
            pid_info = pids_info[pid]
            if pid in random_res_dict:
                pid_res = random_res_dict[pid]
                # 论文基础信息
                journal = pid_info['journal']
                Q_prefix = pid_info['isTop5%']
                Q = pid_info['Q']
                IF = pid_info['IF']
                CC = pid_info['citations']
                # 论文评价信息
                average_score_mean = np.mean([float(pid_res[t]["Average label"]) for t in pid_res])
                average_score_mean_before = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if t < year])
                average_score_mean_after = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if t >= year])
                averate_score_std = np.mean([float(pid_res[t]["Std"]) for t in pid_res])
                
                tb_random_dict["Pubt"].append(year)
                tb_random_dict["isTop5%"].append(Q_prefix)
                tb_random_dict["Q"].append(Q)
                tb_random_dict["IF"].append(IF)
                tb_random_dict["CC"].append(CC)
                tb_random_dict["average_score"].append((average_score_mean + score_plus) * score_rate)
                tb_random_dict["average_score_before"].append((average_score_mean_before + score_plus) * score_rate)
                tb_random_dict["average_score_after"].append((average_score_mean_after + score_plus) * score_rate)
                tb_random_dict["average_std"].append(averate_score_std)
    tb_random_df = pd.DataFrame(tb_random_dict)

    # (2) 获奖论文测评
    tb_prize_dict = {
        "Pubt": list(),
        "average_score":list(),
        "average_score_before":list(),
        "average_score_after":list(),
        "average_std":list()
    }
    prize_papers = utils.read_json(os.path.join(samples_dir, f"collect_papers_by_prize/PrizeSet_{DOMAIN}.json"))
    prize_res_dict = utils.read_json(os.path.join(samples_dir, f"collect_papers_by_prize/prize_samples_results_{DOMAIN}.json"))
    for year in prize_papers:
        pids_info = prize_papers[year]
        for pid in pids_info:
            pid_info = pids_info[pid]
            if pid in prize_res_dict:
                pid_res = prize_res_dict[pid]
                # 论文评价信息
                average_score_mean = np.mean([float(pid_res[t]["Average label"]) for t in pid_res])
                average_score_mean_before = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if t < year])
                average_score_mean_after = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if t >= year])
                averate_score_std = np.mean([float(pid_res[t]["Std"]) for t in pid_res])

                tb_prize_dict["Pubt"].append(year)
                tb_prize_dict["average_score"].append((average_score_mean + score_plus) * score_rate)
                tb_prize_dict["average_score_before"].append((average_score_mean_before + score_plus) * score_rate)
                tb_prize_dict["average_score_after"].append((average_score_mean_after + score_plus) * score_rate)
                tb_prize_dict["average_std"].append(averate_score_std)
    tb_prize_df = pd.DataFrame(tb_prize_dict)

    # (3) 推荐论文测评
    DOMAIN_EN_CH_DICT = {
        "mathematics": "数学",
        "physics": "物理",
        "chemistry": "化学",
        "pubmed": "生物",
    }
    tb_recommend_dict = {
        "Pubt": list(),
        "average_score":list(),
        "average_score_before":list(),
        "average_score_after":list(),
        "average_std":list()
    }
    recommend_papers = pd.read_excel(os.path.join(samples_dir, f"collect_papers_by_expert/专家推荐论文-{DOMAIN_EN_CH_DICT[DOMAIN]}.xlsx"))
    recommend_res_dict = utils.read_json(os.path.join(samples_dir, f"collect_papers_by_expert/expert_samples_results_{DOMAIN}.json"))
    for i in range(len(recommend_papers)):
        # 论文基础信息
        field = recommend_papers.iloc[i]['field']
        title = recommend_papers.iloc[i]['title']
        abstract = recommend_papers.iloc[i]['abstract']
        journal = recommend_papers.iloc[i]['journal']
        pubt = str(recommend_papers.iloc[i]['pubt'])
        year = int(pubt)
        # 论文评价信息
        if title not in recommend_res_dict:
            continue
        pid_res = recommend_res_dict[title]
        average_score_mean = np.mean([float(pid_res[t]["Average label"]) for t in pid_res])
        average_score_mean_before = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if int(t) < year])
        average_score_mean_after = np.mean([float(pid_res[t]["Average label"]) for t in pid_res if int(t) >= year])
        averate_score_std = np.mean([float(pid_res[t]["Std"]) for t in pid_res])

        tb_recommend_dict["Pubt"].append(year)
        tb_recommend_dict["average_score"].append((average_score_mean + score_plus) * score_rate)
        tb_recommend_dict["average_score_before"].append((average_score_mean_before + score_plus) * score_rate)
        tb_recommend_dict["average_score_after"].append((average_score_mean_after + score_plus) * score_rate)
        tb_recommend_dict["average_std"].append(averate_score_std)
    tb_recommend_df = pd.DataFrame(tb_recommend_dict)

    print(f"(1) 随机论文测评 (论文数目: {len(tb_random_df)})")
    print(tb_random_df.groupby('isTop5%')[['average_score_before', 'average_score', 'average_score_after']].mean())
    print("\n")
    print(f"(2) 获奖论文测评 (论文数目: {len(tb_prize_df)})")
    print(tb_prize_df[['average_score_before', 'average_score', 'average_score_after']].mean())
    print("\n")
    print(f"(3) 推荐论文测评 (论文数目: {len(tb_recommend_df)})")
    print(tb_recommend_df[['average_score_before', 'average_score', 'average_score_after']].mean())

    # tb = pt.PrettyTable()
    # tb.field_names = ["期刊", "分区", "IF", "引用"]
    # tb.add_row([journal, Q, IF, CC])
    # print(tb)


if __name__ == "__main__":
    # 模型训练
    # train_model(batch_size=16, gradient_accumulation_steps=4, num_train_epochs=3)
    # train_model_continue(16, 4)
    
    # 模型测试: 测试集上评价
    peft_path = f"./outputs/PUBMED_FUTURE/{args.model_name}/checkpoint-5439"
    # test_model("valid", batch_size=4, peft_path=peft_path)
    # test_model("test", batch_size=4, peft_path=peft_path)

   
