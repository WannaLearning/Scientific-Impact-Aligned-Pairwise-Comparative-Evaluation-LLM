import os
import re
import sys
import time
import json
import torch
import argparse
import random
import logging
import numpy as np
from prettytable import PrettyTable
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset, load_from_disk, Dataset
from safetensors import safe_open
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer
from vllm import SamplingParams
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from component.utils import ModelUtils, save_jsonl, read_jsonl
from component.template import template_dict
from component import utils

import train_llm


base_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/models"
data_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/data"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", default='pubmed', type=str)
parser.add_argument("-m", "--model_dir", default='qwen', type=str)
args = parser.parse_args()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    total_trained_params = 0
    total_untrained_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            total_trained_params += params
        else:
            total_untrained_params += params
        total_params += params
        table.add_row([name, params])
    # print(table)
    print(f"Total Trainable Params: {total_trained_params}")
    print(f"Total Untrainable Params: {total_untrained_params}")
    print(f"Total Params: {total_params}")


def main(model_name, ck_num):
    if model_name.lower().startswith("qwen"):
        model_dir = os.path.join(base_dir, "Qwen-Instruct")
    elif model_name.lower().startswith("llama") or model_name.lower().startswith("meta-llama"):
        model_dir = os.path.join(base_dir, "Llama-Instruct")
    elif model_name.lower().startswith("mistral"):
        model_dir = os.path.join(base_dir, "Mistral-Instruct")
    else:
        model_dir = ""
    
    model_save_name = f"{model_name}-merged-{args.domain}"
    # 储存Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))
    tokenizer.save_pretrained(os.path.join(model_dir, model_save_name))

    # 储存Merge Model
    lora_adapter = f"save_adapter/qlora/{model_name}/originalityeval_{args.domain}/checkpoint-{ck_num}"
    print("***Load Adapter***")
    print(f"Adapter path: {lora_adapter}")

    base_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, model_name),
        device_map=None,
        trust_remote_code=True,
    )
    base_model = base_model.to('cuda')
    print("***Base Model***")
    print(base_model)
    count_parameters(base_model)
    print("\n\n")

    model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter)
    print("***With LoRA Model***")
    print(model_to_merge)
    count_parameters(model_to_merge)
    print("\n\n")

    merged_model = model_to_merge.merge_and_unload()
    print("***Merged Model***")
    print(merged_model)
    count_parameters(merged_model)

    merged_model.save_pretrained(os.path.join(model_dir, model_save_name))


if __name__ == "__main__":
    model_name = f"Qwen2.5-{14}B-Instruct"
    main(model_name, ck_num=5436)