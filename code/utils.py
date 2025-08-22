import json
import pickle
import requests
import re
import nltk
import numpy as np
import prettytable as pt
import pandas as pd


def save_pickle(file, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(file, f)


def read_pickle(read_path):
    with open(read_path, 'rb') as f:
        file = pickle.load(f)
    return file


def save_json(json_file, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_file, f)


def read_json(json_path):
    with open(json_path, 'r') as f:
        file = json.load(f)
    return file


def save_jsonl(json_file, json_path):
    # 打开文件并逐行写入 JSON 对象
    with open(json_path, 'w', encoding='utf-8') as f:
        for entry in json_file:
            json_record = json.dumps(entry, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            f.write(json_record + '\n')  # 写入文件并添加换行符

def read_jsonl(json_path):
    # 打开文件并逐行读取 JSON 对象
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data

def trancate_abstract(abstract: str, abstract_len_limit: int=512):
    """截断摘要"""
    abstract = " ".join(abstract.split(" ")[: abstract_len_limit]).strip()
    return abstract
