import os
import re
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
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
import grpo_train_papereval


logger = grpo_train_papereval.logger
models_dir = grpo_train_papereval.models_dir
system_prompt = grpo_train_papereval.system_prompt

data_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/data/DI_SCORE_PREDICTION_DATA/PUBMED_FUTURE"
data_expert_dir = os.path.join(data_dir, "extra_eval_data", "expert_recommend_set")
data_expert_meta_path = os.path.join(data_expert_dir, "专家推荐论文-生物.xlsx")
#
data_prize_dir = os.path.join(data_dir, "extra_eval_data", "prize_winning_set")
data_prize_meta_path = os.path.join(data_prize_dir, "PrizeSet.xlsx")
#
data_random_dir = os.path.join(data_dir, "extra_eval_data", "random_sample_set")
data_random_meta_path = os.path.join(data_random_dir, "random_samples_pubmed_2024.json")

data_decay_dir = os.path.join(data_dir, "extra_eval_data", "decay_sample_set")
data_decay_meta_path = os.path.join(data_decay_dir, "meta.json")


def load_peft_model_trained(model_path, checkpoint_path, max_seq_length):
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"载入LORA训练模型: {checkpoint_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_path, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            load_in_4bit = True, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            gpu_memory_utilization = 0.90, # Reduce if out of memory
        )
    else:
        logger.info(f"载入未训练训练模型: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            load_in_4bit = True, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            gpu_memory_utilization = 0.9, # Reduce if out of memory
    )
    print(model)
    return model, tokenizer


def infer(model, tokenizer, query, max_new_tokens):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    output = infer_message(model, tokenizer, messages, max_new_tokens)
    return output


def infer_message(model, tokenizer, message, max_new_tokens):
    text = tokenizer.apply_chat_template(
        message,
        add_generation_prompt = True, # Must add for generation
        tokenize = False,
    )
    sampling_params = SamplingParams(
        temperature = 0.1,
        top_k = 50,
        max_tokens = max_new_tokens,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
    )[0].outputs[0].text
    return output


def infer_batch(
    model, 
    tokenizer, 
    message, 
    do_sample=False, 
    max_new_tokens=128,
    temperature=0.1
    ):
    """ Mistral推断 """
    text = tokenizer.apply_chat_template(
        message,
        add_generation_prompt = True, # Must add for generation
        tokenize = False,
    )
    inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        max_length=4096, 
        truncation=True, 
        padding=True
    ).to('cuda')

    outputs = model.generate(
        **inputs, 
        do_sample=do_sample, 
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=max_new_tokens, 
    )
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = list()
    for completion in completions:
         answers.append(completion)
    return answers[0]


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
    return model_vllm


def infer_vllm(model_vllm, adapter_path, messages, temperature=0.1, max_tokens=64):
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = 0.95,
        max_tokens = max_tokens,
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


def test_on_vllm(data_name, model_id, checkpoint_path):
    # load data
    train_dataset, valid_dataset, test_dataset = grpo_train_papereval.get_medicine_papereval()
    save_dir = f"./results/{model_id}/{os.path.basename(checkpoint_path)}"
    if data_name == 'test':
        dataset = test_dataset
        save_path = os.path.join(save_dir, "eval_on_test.json")
    elif data_name == 'valid':
        dataset = valid_dataset
        save_path = os.path.join(save_dir, "eval_on_valid.json")
    else:
        dataset = train_dataset
        save_path = os.path.join(save_dir, "eval_on_train.json")

    # load model
    model_vllm = load_model_vllm(model_id)
    # test data
    logger.info(f"Base model: {model_id}")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Dataset name (Select from Training/Validation/Test set): {data_name}")

    # inference
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        results = utils.read_json(save_path)
        logger.info(f"已处理: {len(results)}")
    else:
        results = dict()

    count = 0
    for i in tqdm(range(len(dataset))):
        if str(i) in results:
            continue
        message = dataset[i]['prompt']
        query = message[-1]['content']
        acutal_answer = dataset[i]['answer']
        pred_answer = infer_vllm(model_vllm, checkpoint_path, message)
        results[i] = {"actual": acutal_answer, "pred": pred_answer}
        utils.save_json(results, save_path)


def additional_test_on_vllm(
    model_id, 
    checkpoint_path, 
    data_name, 
    meta_name="meta.json", 
    match_name="match.json", 
    results_name="results_grpo.json",
    ):
    logger.info("***Using vllm for inference***")
    logger.info("***Hence, adapter cannot be used at lm_head***")
    logger.info("***额外的测试数据: 专家推荐论文 or 获奖论文 or JCR随机采样论文***")

    if data_name == "expert_recommend_set":
        logger.info("***专家推荐论文集***")
        data_dir = data_expert_dir
    elif data_name == "prize_winning_set":
        logger.info("***获奖论文集***")
        data_dir = data_prize_dir
    elif data_name == "random_sample_set":
        logger.info("***2024年根据JCR分区随机采样集***")
        data_dir = data_random_dir
    elif data_name == "decay_sample_set":
        logger.info("***2010年根据JCR分区衰减采样集***")
        data_dir = data_decay_dir
    else:
        logger.info("***数据集异常***")
        return 

    # 载入模型
    model_vllm = load_model_vllm(model_id)

    # 读取元数据 & 检索对照论文
    data_meta_json = utils.read_json(os.path.join(data_dir, meta_name))
    data_match_json = utils.read_json(os.path.join(data_dir, match_name))

    # 结果储存路径
    res_save_path = os.path.join(data_dir, results_name)
    if os.path.exists(res_save_path):
        pred_res_dict = utils.read_json(res_save_path)
    else:
        pred_res_dict = dict()

    for idx in data_meta_json:
        if idx in pred_res_dict:
            continue
        if idx not in data_match_json:
            continue

        result = dict()
        title_A = data_meta_json[idx]['title']
        abstract_A = data_meta_json[idx]['abstract']
        pubt = data_meta_json[idx]['pubt']
        similars = data_match_json[idx]
        for year in similars:
            result[year] = list()
            for info_sim in similars[year]:
                title_C = info_sim['title_sim']
                abstract_C = info_sim['abstract_sim']
                question = (
                    f"As an expert in medical field, "
                    "you are required to meticulously scrutinize the titles and abstracts of both Paper A and Paper B in order to ascertain whether Paper A is superior in quality to Paper B, comparable in quality to Paper B, or inferior in quality to Paper B."
                    "\n\n"

                    "Title of Paper A:\n"
                    f"{title_A}\n"
                    "Abstract of Paper A:\n"
                    f"{abstract_A}\n"

                    "Title of Paper B:\n"
                    f"{title_C}\n"
                    "Abstract of Paper B:\n"
                    f"{abstract_C}\n\n"

                    "You should generate the result in the following format: [SCORE_A][>, <, or ≈][SCORE_B].\n"
                    "If SCORE_A (i.e., the score of Paper A) is at least 2 points higher than SCORE_B (i.e., the score of Paper B), you generate SCORE_A>SCORE_B (e.g, 7>3);\n"
                    "If SCORE_A is at least 2 points lower than SCORE_B, you generate SCORE_A<SCORE_B (e.g, 2<6);\n"
                    "If the difference between SCORE_A and SCORE_B is no more than 1 point, you generate SCORE_A≈SCORE_B (e.g, 4≈5);\n"
                    "Evaluation result:"
                )
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                output = infer_vllm(model_vllm, checkpoint_path, message, temperature=0.1, max_tokens=64)
                result[year].append(output)
                
        pred_res_dict[idx] = result
        utils.save_json(pred_res_dict, res_save_path)


def print_accuray(results_save_path):
    """打印准确率"""
    def split_expression(expr):
        tokens = re.findall(r'(\d+|\D)', expr)
        tokens = [token for token in tokens if token.strip()]
        return tokens

    logger.info(f"results_save_path: {results_save_path}")
    results = utils.read_json(results_save_path)

    error_count = 0
    true_labels = list()
    pred_labels = list()
    actual_A_list = list()
    pred_A_list = list()
    actual_B_list = list()
    pred_B_list = list()
    actual_dist_list = list()
    pred_dist_list = list()
    for i in results:
        actual_response = results[i]["actual"]
        pred_response = results[i]["pred"]
        
        actual_response = split_expression(actual_response)
        pred_response = split_expression(pred_response)

        actual_A, actual_label, actual_B = actual_response
        pred_A, pred_label, pred_B = pred_response

        actual_A_list.append(actual_A)
        pred_A_list.append(pred_A)
        actual_B_list.append(actual_B)
        pred_B_list.append(pred_B)
        actual_dist_list.append(int(actual_A)-int(actual_B))
        pred_dist_list.append(int(pred_A)-int(pred_B))

        if pred_label in ["<", "≈", ">"]:
            true_labels.append(actual_label)
            pred_labels.append(pred_label)
        else:
            print(pred_response)
            error_count += 1
    
    matrix_test = confusion_matrix(true_labels, pred_labels)
    report_on_test = classification_report(true_labels, pred_labels, digits=4)
    report_on_test_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)

    r2_A = round(r2_score(actual_A_list, pred_A_list), 6)
    r2_B = round(r2_score(actual_B_list, pred_B_list), 6)
    r2_dist = round(r2_score(actual_dist_list, pred_dist_list), 6)
    logger.info(f"推理格式错误数目: {error_count}")
    logger.info(matrix_test)
    logger.info(report_on_test)
    logger.info(f"R2(A): {r2_A}   R2(B): {r2_B}   R2(A-B): {r2_dist}")
    logger.info("#"*20 + "\n\n")


    print(report_on_test_dict.keys())
    results = {
        "R2(A)": r2_A,
        "R2(B)": r2_B,
        "R2(A-B)": r2_dist,
    }
    return results


def main():
    for ms in [1.5]:
        # model_id = f"Qwen-Instruct/Qwen2.5-{ms}B-Instruct-merged"
        # model_id = f"Llama-Instruct/Llama-3.2-3B-Instruct-merged"
        model_id = "Mistral-Instruct/Mistral-7B-Instruct-v0.3-merged"
        for ck in [7251]:
            test_on_vllm("test", model_id, f"./outputs/{model_id}/checkpoint-{ck}")
            test_on_vllm("valid", model_id, f"./outputs/{model_id}/checkpoint-{ck}")


if __name__ == "__main__":
    main()
