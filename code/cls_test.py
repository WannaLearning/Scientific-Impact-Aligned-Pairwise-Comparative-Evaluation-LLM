import os
import sys
import json
import math
import copy
import nltk
import copy
import time
import random
import humanize
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
from datetime import timedelta
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from clickhouse_driver import Client

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", default='pubmed', type=str)
parser.add_argument("-m", "--model_name", default="Qwen2.5-14B-Instruct", type=str)
parser.add_argument("-m_p", "--model_name_parent", default="Qwen-Instruct", type=str)
parser.add_argument("-r", "--lora_rank", default=32, type=int)
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
import extra_eval_data
from cls_train_papereval import make_prompt, load_trained_model, BASE_DIR


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

DOMAIN = args.domain
MODEL_SAVE_PATH = f"./outputs/{DOMAIN.upper()}_FUTURE" # 模型存储路径
RESULT_SAVE_PATH = f"./results/{DOMAIN.upper()}_FUTURE" # 测试结果储存路径

model_hub_dir = os.path.join(BASE_DIR, "models")
model_base_path = os.path.join(model_hub_dir, f"{args.model_name_parent}/{args.model_name}")
model_save_path = os.path.join(MODEL_SAVE_PATH, os.path.basename(model_base_path))


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


def generate_originality_score_func(
    model, tokenizer, 
    titles_batch, abstracts_batch, 
    titles_sim_batch, abstracts_sim_batch, 
    batch_size,
    ):
    """调用模型生成原创得分:"""
    # logger.info(f"{len(titles_batch)} vs {len(abstracts_batch)} vs {len(titles_sim_batch)} vs {len(abstracts_sim_batch)}")

    assert len(titles_batch) == len(abstracts_batch)
    assert len(titles_batch) == len(titles_sim_batch)
    assert len(abstracts_batch) == len(abstracts_sim_batch)

    prompts = list()
    for title_A, abstract_A, title_C, abstract_C in zip(titles_batch, abstracts_batch, titles_sim_batch, abstracts_sim_batch):
        prompt = make_prompt(title_A, abstract_A, title_C, abstract_C)
        prompts.append(prompt)

    output_softmax = list()
    scores = list()
    labels = list()
    batch_prompt = list()
    for i, prompt in enumerate(prompts):
        batch_prompt.append(prompt)
        if len(batch_prompt) >= batch_size or i == len(prompts)-1:
            batch = tokenizer(batch_prompt, max_length=600 * (10 + 1) + 100, truncation=True, return_tensors='pt', padding=True)
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
            batch_scores = compute_score(batch_output_softmax)
            batch_labels = np.argmax(batch_output_softmax, axis=-1)

            output_softmax += list(batch_output_softmax)
            scores += list(batch_scores)
            labels += list(batch_labels)
            batch_prompt = list()
    return output_softmax, scores, labels


def evaluate_func(
    model_score, tokenizer_score,
    titles_batch, 
    abstracts_batch, 
    pubts_batch, 
    similars_batch,
    ):
    pred_res_batch = list()
    for eval_idx in range(len(titles_batch)):
        tb_score_yearly = pt.PrettyTable()
        tb_score_yearly.title = '论文评价的逐年结果'
        tb_score_yearly.field_names = ["时间", "是否出版", "对照数量", "标签", "均分", "标准差"]

        title_core = titles_batch[eval_idx]
        abstract_core = abstracts_batch[eval_idx]

        time_span = sorted(list(similars_batch[eval_idx].keys()))
        pred_res = dict()

        # 统计所有输入信息
        titles_batch_for_score = list()
        abstracts_batch_for_score = list()
        titles_sim_batch_for_score = list()
        abstracts_sim_batch_for_score = list()
        t_batch_for_score = list()
        for t in time_span:
            for info_sim in similars_batch[eval_idx][t]:
                title_sim = info_sim['title_sim']
                abstract_sim = info_sim['abstract_sim']
                journal_sim = info_sim['journal_sim']
                titles_batch_for_score.append(title_core)
                abstracts_batch_for_score.append(abstract_core)
                titles_sim_batch_for_score.append(title_sim)
                abstracts_sim_batch_for_score.append(abstract_sim)
                t_batch_for_score.append(t)

        output_softmax, scores, labels = generate_originality_score_func(
            model_score, tokenizer_score,
            titles_batch_for_score, abstracts_batch_for_score,
            titles_sim_batch_for_score, abstracts_sim_batch_for_score,
            batch_size=32,
        )

        # 按年份分配结果
        for t in time_span:
            output_softmax = np.array(output_softmax)
            scores = np.array(scores)
            labels = np.array(labels)
            t_batch_for_score = np.array(t_batch_for_score)
            # 
            output_softmax_t = output_softmax[t_batch_for_score == t]
            scores_t = scores[t_batch_for_score == t]
            labels_t = labels[t_batch_for_score == t]

            if t == pubts_batch[eval_idx]:
                isFlag = 1
            else:
                isFlag = 0
            tb_score_yearly.add_row([
                t,
                isFlag,
                len(scores_t),
                round(np.mean(labels_t), 2), 
                round(np.mean(scores_t), 2), 
                round(np.std(scores_t), 2),
            ])
            pred_res[t] = {
                "isPubt": str(isFlag),
                "Number of comparative samples": len(scores_t),
                "scores": scores_t,
                "Average label": round(np.mean(labels_t), 2).item(),
                "Average score": round(np.mean(scores_t), 2).item(),
                "Percentile label": round(np.percentile(labels_t, 50), 2).item(),
                "Percentile score": round(np.percentile(scores_t, 50), 2).item(),
                "Average softmax": np.mean(output_softmax_t, axis=0),
                "softmax": output_softmax_t,
                "Std": round(np.std(scores_t), 2).item(),
            }
        logger.info(tb_score_yearly)
        pred_res_batch.append(pred_res)
    return pred_res_batch


def main(data_name, meta_name="meta.json", match_name="match.json", results_name="results_cls.json"):
    """在定制数据集上开展论文评价"""
    model_score, tokenizer_score = model, tokenizer = load_trained_model(
        model_base_path, model_save_path, num_labels=3, quantify=True, peft_path=""
    )
    if data_name == "expert_recommend_set":
        logger.info("***专家推荐论文集***")
        data_tmp_dir = extra_eval_data.data_expert_dir
    elif data_name == "prize_winning_set":
        logger.info("***获奖论文集***")
        data_tmp_dir = extra_eval_data.data_prize_dir
    elif data_name == "random_sample_set":
        logger.info("***2024年根据JCR分区随机采样集***")
        data_tmp_dir = extra_eval_data.data_random_dir
    elif data_name == "decay_sample_set":
        logger.info("***2010年根据JCR分区衰减采样集***")
        data_tmp_dir = extra_eval_data.data_decay_dir
    else:
        logger.info("***数据集异常***")
        return 

    # 读取元数据 & 检索对照论文
    data_meta_json = utils.read_json(os.path.join(data_tmp_dir, meta_name))
    data_match_json = utils.read_json(os.path.join(data_tmp_dir, match_name))

    # # 结果储存路径
    # res_save_path = os.path.join(data_tmp_dir, results_name)
    # if os.path.exists(res_save_path):
    #     pred_res_dict = utils.read_json(res_save_path)
    # else:
    #     pred_res_dict = dict()
    pred_res_dict = dict()

    start_t = time.perf_counter()
    for i in data_meta_json:
        if i in pred_res_dict:
            continue
        if i not in data_match_json:
            continue

        start_t_i = time.perf_counter()
        pred_res_batch = evaluate_func(
            model_score, tokenizer_score,
            titles_batch=[data_meta_json[i]['title']], 
            abstracts_batch=[data_meta_json[i]['abstract']], 
            pubts_batch=[data_meta_json[i]['pubt']], 
            similars_batch=[data_match_json[i]],
        )
        end_t_i = time.perf_counter()
        logger.info(f"运行时间: {timedelta(seconds=end_t_i-start_t_i)} / 条")

        pred_res = pred_res_batch[0]
        pred_res_str = dict()
        for t in pred_res:
            pred_res_str[t] = dict()
            pred_res_str[t]['Average softmax'] = pred_res[t]['Average softmax'].tolist()
            pred_res_str[t]['softmax'] = pred_res[t]['softmax'].tolist()
        pred_res_str = json.dumps(pred_res_str, indent=4)

        pred_res_dict[i] = pred_res_str
        # utils.save_json(pred_res_dict, res_save_path)
    end_t = time.perf_counter()
    logger.info(f"运行时间: {timedelta(seconds=end_t-start_t)} / {len(data_meta_json)}条")


if __name__ == "__main__":
    main(
        data_name="expert_recommend_set", 
        meta_name="meta.json", 
        match_name="match.json", 
        results_name="results_cls.json"
    )











