import os
import re
import sys
import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk, Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from safetensors import safe_open
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer
from vllm import SamplingParams
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import utils
import sft_train_papereval


logger = sft_train_papereval.logger
models_dir = sft_train_papereval.models_dir
system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def load_model_vllm(model_id):
    model_path = os.path.join(models_dir, model_id)
    logger.info(f"load_model_vllm (以VLLM载入模型): {model_path}")
    model_vllm = LLM(
        model=model_path, 
        task="generate", 
        gpu_memory_utilization=0.9, 
        trust_remote_code=True, 
        enable_lora=True, 
        max_lora_rank=32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model_vllm, tokenizer


def infer_vllm(model_vllm, adapter_path, messages, temperature=0.0):
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = 0.95,
        max_tokens = 64,
    )
    if adapter_path:
        output = model_vllm.chat(
            messages,
            sampling_params = sampling_params,
            lora_request = LoRARequest("test", 1, adapter_path),
            use_tqdm = False
        )[0].outputs[0].text
    else:
        output = model_vllm.chat(
            messages,
            sampling_params = sampling_params,
            use_tqdm = False
        )[0].outputs[0].text
    return output


def print_accuray(results):
    relation_dict = {">": 2, "=":1, "<": 0, "≈": 1, "A>B": 2, "A=B": 1, "A<B": 0, "A≈B": 1}
    true_labels = list()
    pred_labels = list()
    for i in results:
        result = results[i]
        actual_rel = result['rel']
        pred_rel = result['result']
        if actual_rel not in relation_dict or pred_rel not in relation_dict:
            continue
        true_label = relation_dict[actual_rel]
        pred_label = relation_dict[pred_rel]
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    # 
    matrix_test = confusion_matrix(true_labels, pred_labels)
    report_on_test = classification_report(true_labels, pred_labels, digits=4)
    logger.info(matrix_test)
    logger.info(report_on_test)


def transform_classify_data2grpo_data(data):
    relation_dict = {">": 2, "=":1, "<": 0}
    relation_reverse_dict = {v: k for k, v in relation_dict.items()}

    def generate_conversation_grpo_format(examples):
        titles_A = examples["titles_A"]
        titles_C = examples["titles_C"]
        abstracts_A = examples["abstracts_A"]
        abstracts_C = examples["abstracts_C"]
        #
        level_A = examples["level_A"]
        level_C = examples["level_C"]
        labels = examples["labels"]

        prompts = list()
        anwers = list()
        for title_A, title_C, abstract_A, abstract_C, level_A, level_C, label in zip(titles_A, titles_C, abstracts_A, abstracts_C, level_A, level_C, labels):
            abstract_A = utils.trancate_abstract(abstract_A)
            abstract_C = utils.trancate_abstract(abstract_C)

            question = (
                "***TASK Instruction***\n"
                "Your task is to evaluate and compare the research quality of Paper A and Paper B across three key dimensions: Originality, Significance, and Rigour."
                # Originality definition
                "(1) Originality will be understood as the extent to which the output makes an important and innovative contribution to understanding and knowledge in the field.\n"
                "Research outputs that demonstrate originality may do one or more of the following: "
                "produce and interpret new empirical findings or new material; engage with new and/or complex problems; "
                "develop innovative research methods, methodologies and analytical techniques; "
                "show imaginative and creative scope; "
                "provide new arguments and/or new forms of expression, formal innovations, interpretations and/or insights; "
                "collect and engage with novel types of data; "
                "and/or advance theory or the analysis of doctrine, policy or practice, and new forms of expressions.\n"
                # Significance definition
                "(2) Significance will be understood as the extent to which the work has influenced, or has the capacity to influence, knowledge and scholarly thought, or the development and understanding of policy and/or practice.\n"
                # Rigour definition
                "(3) Rigour will be understood as the extent to which the work demonstrates intellectual coherence and integrity, and adopts robust and appropriate concepts, analyses, sources, theories and/or methodologies.\n\n"
                # 
                "Specifically, you assess the Originality, Significance, and Rigour of the two papers by analyzing their provided titles and abstracts. "
                "Assign numerical scores (integers from 1 to 10) to each paper, denoted as SCORE_A and SCORE_B, respectively. "
                "The score disparity between the two papers should reflect the quality difference between them. "
                
                "Based on the SCORE_A and SCORE_B, determine the comparison result between Paper A and Paper B:\n"
                "If SCORE_A is at least 2 points higher than SCORE_B (e.g., 7 and 3), Paper A's quality is significantly superior to Paper B's, output A>B.\n"
                "If SCORE_A is at least 2 points lower than SCORE_B (e.g., 2 and 5), Paper A's quality is significantly inferior to Paper B's, output A<B.\n"
                "If the difference between SCORE_A and SCORE_B is no more than 1 point (e.g., 2 and 1), their rearch qualities are nearly comparable, output A≈B.\n"
                
                "***INPUT***\n"
                f"Title of Paper A: {title_A}\nAbstract of Paper A: {abstract_A}\n\n"
                f"Title of Paper B: {title_C}\nAbstract of Paper B: {abstract_C}\n\n"

                "***OUTPUT***\n"
                "Provide only the results in the following output format without any other information:"
                "A=<SCORE_A>;B=<SCORE_B>;<SCORE_A-SCORE_B><comparison result>;"
                "Example: A=2;B=8;A-B=-6;A<B;"
            )
            answer = f"{level_A}\n{relation_reverse_dict[label]}\n{level_C}"
            prompts.append([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question}
            ])
            anwers.append(answer)
        return {"prompt": prompts, "answer": anwers}

    data_transform = data.map(generate_conversation_grpo_format, batched=True)
    exclude_keys = ['pids_A', 'pids_C', 'titles_A', 'abstracts_A', 'titles_C', 'abstracts_C', "level_A", "level_C", "labels"]
    data_grpo = data_transform.remove_columns(exclude_keys)
    return data_grpo


def get_medicine_papereval():
    data_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/data/DI_SCORE_PREDICTION_DATA/PUBMED_FUTURE"
    train_dataset_path = os.path.join(data_dir, "classification/train_dataset")
    valid_dataset_path = os.path.join(data_dir, "classification/valid_dataset")
    test_dataset_path  = os.path.join(data_dir, "classification/test_dataset")

    logger.info("*** Load Train/Valid/Test Dataset ***")
    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)

    logger.info("*** Transform Train/Valid/Test Dataset into GRPO format ***")
    train_dataset = transform_classify_data2grpo_data(train_dataset)
    valid_dataset = transform_classify_data2grpo_data(valid_dataset)
    test_dataset  = transform_classify_data2grpo_data(test_dataset)

    logger.info("*** Train set ***")
    logger.info(train_dataset)
    logger.info("*** Validation set ***")
    logger.info(valid_dataset)
    logger.info("*** Test set ***")
    logger.info(test_dataset)
    return train_dataset, valid_dataset, test_dataset


def extract_results(text):
    pattern = r"A=(\d+);B=(\d+);A-B=(-?\d+);(A<B|A>B|A≈B);"
    match = re.search(pattern, text)
    result_dict = {'A': None, 'B': None, "A-B": None, 'result': None}
    if match:
        # 提取匹配的组，如果无法匹配某个部分，则返回 None
        a_value = match.group(1) if match.group(1) else None
        b_value = match.group(2) if match.group(2) else None
        a_minus_b = match.group(3) if match.group(3) else None
        result = match.group(4) if match.group(4) else None
        # 生成字典
        result_dict = {
            'A': a_value,
            'B': b_value,
            'A-B': a_minus_b,
            'result': result
        }
    return result_dict


def main():
    # load model
    model_id = "Qwen-Instruct/Qwen2.5-14B-Instruct"
    model_path = os.path.join(models_dir, model_id)
    checkpoint = "4000"
    checkpoint_path = f"outputs/{model_id}/checkpoint-{checkpoint}"
    logger.info(checkpoint_path)
    max_seq_length = 1024 * 3 + 64
    
    model_vllm, tokenizer = load_model_vllm(model_id)
    train_dataset, valid_dataset, test_dataset = get_medicine_papereval()

    # inference
    test_name = "valid_dataset"
    if test_name == "valid_dataset":
        dataset = valid_dataset
    elif test_name == "test_dataset":
        dataset = test_dataset
    else:
        raise ValueError(f"{test_name}")

    save_dir = f"./results/{model_id}/checkpoint-{checkpoint}"
    save_path = os.path.join(save_dir, f"results({test_name}).json")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        results = utils.read_json(save_path)
        logger.info(f"已处理: {len(results)}")
    else:
        results = dict()

    count = 0
    for i in range(len(dataset)):
        if str(i) in results:
            continue
        message = dataset[i]['prompt']
        query = message[-1]['content']
        true_answer = dataset[i]['answer']
        level_A, rel_AB, level_B = true_answer.split("\n")

        output = infer_vllm(model_vllm, checkpoint_path, message)
        result = extract_results(output)
        
        result['A_true'] = level_A
        result['rel'] = rel_AB
        result['B_true'] = level_B
        results[str(i)] = result
        utils.save_json(results, save_path)

        logger.info(f"{result}")

    print_accuray(results)


if __name__ == "__main__":
    main()
    # print_accuray()
