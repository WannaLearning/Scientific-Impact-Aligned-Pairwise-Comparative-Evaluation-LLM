import os
import re
import sys
import torch
import logging
import argparse
import random
import unsloth
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

import utils

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", default='chemistry', type=str)
args = parser.parse_args()

models_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/models"
data_huggingface_dir = "/public/home/lab8/project/PaperOriginality/code/hsz/data/huggingface"

reasoning_start, reasoning_end = "<think>", "</think>"
solution_start, solution_end = "<answer>", "</answer>"

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)

# system_prompt = \
# f"""You are given a problem.
# Think about the problem and provide your working out.
# Place it between {reasoning_start} and {reasoning_end}.
# Then, provide your answer between {solution_start} and {solution_end}"""
system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def print_parameters(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    logger.info(f'Total params: {Total_params}')
    logger.info(f'Trainable params: {Trainable_params}')
    logger.info(f'Non-trainable params: {NonTrainable_params}')


def load_peft_model_from_scratch(model_path, max_seq_length, lora_rank):
    """初始化加载lora模块"""
    # Load model
    logger.info("*** load_peft_model_from_scratch: unsloth + qlora + grpo ***")
    logger.info(f"model_path: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.90, # Reduce if out of memory
    )
    target_modules = ["q_proj", 'k_proj', "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=target_modules, # Remove QKVO if out of memory
        r=lora_rank, 
        lora_alpha=lora_rank,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth", # Enable long context finetuning
        random_state=9527,
    )
    logger.info(model)
    print_parameters(model)
    return model, tokenizer


def load_peft_model_for_continue_training(checkpoint_path, max_seq_length):
    """加载已训练的lora模块"""
    logger.info("*** load_peft_model_for_continue_training (SFT as the basis): unsloth + qlora + grpo ***")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        gpu_memory_utilization = 0.80, # Reduce if out of memory
    )
    logger.info(model)
    print_parameters(model)
    return model, tokenizer


def extract_results(text):
    tokens = re.findall(r'(\d+|\D)', text)
    tokens = [token for token in tokens if token.strip()]
    try:
        result_dict = {"A": tokens[0], "B": tokens[-1], "label": tokens[1]}
    except:
        result_dict = {"A": "", "B": "", "label": ""}
    return result_dict


def transform_classify_data2grpo_data(data):
    def generate_conversation_grpo_format(examples):
        relation_dict = {">": 2, "≈":1, "<": 0}
        relation_reverse_dict = {v: k for k, v in relation_dict.items()}
        # 输入信息
        titles_A = examples["titles_A"]
        titles_C = examples["titles_C"]
        abstracts_A = examples["abstracts_A"]
        abstracts_C = examples["abstracts_C"]
        # 输出信息
        level_A = examples["level_A"]
        level_C = examples["level_C"]
        labels = examples["labels"]

        prompts = list()
        anwers = list()
        for title_A, title_C, abstract_A, abstract_C, level_A, level_C, label in zip(titles_A, titles_C, abstracts_A, abstracts_C, level_A, level_C, labels):
            abstract_A = utils.trancate_abstract(abstract_A)
            abstract_C = utils.trancate_abstract(abstract_C)

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

            # SFT OUTPUT
            level_A = int(level_A)
            level_B = int(level_C)
            if relation_reverse_dict[label] == '>':
                response = f"{level_A}>{level_B}"
            elif relation_reverse_dict[label] == '<':
                response = f"{level_A}<{level_B}"
            else:
                response = f"{level_A}≈{level_B}"
            answer = response

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


def get_papereval_data():
    data_dir = f"/public/home/lab8/project/PaperOriginality/code/hsz/data/DI_SCORE_PREDICTION_DATA/{args.domain.upper()}_FUTURE"
    logger.info(f"载入数据: {data_dir}")
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


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 1
        else:
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.2 if response.count(reasoning_start) == 1 else -0.25
            score += 0.2 if response.count(reasoning_end)   == 1 else -0.25
            score += 0.2 if response.count(solution_start)  == 1 else -0.25
            score += 0.2 if response.count(solution_end)    == 1 else -0.25
        scores.append(score)
    logger.info(f"\nReward of FORMAT: {scores}")
    return scores


def check_comment(prompts, completions, answer, **kwargs):
    """
    从三个维度对比评价
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    tb = PrettyTable()
    tb.title = "Reward of COMMENT"
    tb.field_names = ["Reward", "Reward details", "Count details"]
    scores = list()
    for text in responses:
        score = 0
        score_detail = {"OE": 0, "SE": 0, "RE": 0, "Others": 0}
        count_detail = {"OE": 0, "SE": 0, "RE": 0, "Others": 0}
        # 匹配 <think> 和 </think>
        think_content_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        # 
        if think_content_match:
            think_content = think_content_match.group(1)
            pattern = r'###(.*?)###\s*(.*?)(?=\s*###.*?###|\Z)'
            matches = re.findall(pattern, think_content, re.DOTALL)
            reward = 0.25
            # 
            for title, content in matches:
                title = title.lower().strip()
                if title == 'Originality Evaluation'.lower():
                    count_detail["OE"] += 1
                elif title == 'Significance Evaluation'.lower():
                    count_detail["SE"] += 1
                elif title == 'Rigour Evaluation'.lower():
                    count_detail["RE"] += 1
                else:
                    count_detail["Others"] += 1
            #
            for content_type in count_detail:
                if content_type in ["OE", "SE", "RE"]:
                    content_type_num = 1
                else:
                    content_type_num = 0
                if count_detail[content_type] == content_type_num:
                    score += reward
                    score_detail[content_type] += reward
                else:
                    score -= reward
                    score_detail[content_type] -= reward
        else:
            score += -1
        tb.add_row([round(score, 4), score_detail, count_detail])
        scores.append(score)
    
    logger.info(f"\n{tb}")
    return scores


def check_score_reward_dist_func(pred_dist):
    # A准确奖励 / 无需B， 避免重复奖励
    return (abs(pred_dist) + 1) ** (-0.5)


def check_score_reward_gap_func(cls_bool, acutal_label, pred_gap, actual_gap):
    """
    由pred_dist和actual_dist的距离决定的奖励函数
    cls_bool: 是否正确分类
    acutal_label: 实际标签
    pred_dist: 预测距离 (pred_A - pred_B)
    actual_dist: 实际距离 (level_A - level_B)
    """
    gap_dist = abs(pred_gap - actual_gap)
    if cls_bool:
        # 在分类预测正确时,
        if acutal_label == ">" or acutal_label == "<":
            # gap_dist belong to [0, 7], gap_dist_norm belong to 1, 5, 10
            if gap_dist <= 1:
                gap_dist_norm = 1
            elif gap_dist <= 3:
                gap_dist_norm = 5
            elif gap_dist <= 7:
                gap_dist_norm = 10
            else:
                raise ValueError(f"When {acutal_label}, gap_dist ({gap_dist}) error")
        elif acutal_label == "≈":
            # gap_dist belong to [0, 2], gap_dist_norm belong to [1, 3]
            if gap_dist == 0:
                gap_dist_norm = 1
            elif gap_dist == 1:
                gap_dist_norm = 5
            elif gap_dist == 2:
                gap_dist_norm = 10
            else:
                raise ValueError(f"When {acutal_label}, gap_dist ({gap_dist}) error")
        else:
            raise ValueError(f"acutal_label ({acutal_label}) error")
    else:
        # 在分类预测错误时,
        if acutal_label == ">" or acutal_label == "<":
            # gap_dist belong to [1, 18] -> normalize into [1, 10]
            if gap_dist <= 2:
                gap_dist_norm = 1
            elif gap_dist <= 5:
                gap_dist_norm = 2
            elif gap_dist <= 7:
                gap_dist_norm = 3
            elif gap_dist <= 11:
                gap_dist_norm = gap_dist - 4
            elif gap_dist <= 13:
                gap_dist_norm = 8
            elif gap_dist <= 16:
                gap_dist_norm = 9
            elif gap_dist <= 18:
                gap_dist_norm = 10
            else:
                raise ValueError(f"When {acutal_label}, gap_dist ({gap_dist}) error")
        elif acutal_label == "≈":
            # gap_dist belong to [1, 10]
            if gap_dist <= 10:
                gap_dist_norm = gap_dist
            else:
                raise ValueError(f"When {acutal_label}, gap_dist ({gap_dist}) error")
        else:
            raise ValueError(f"acutal_label ({acutal_label}) error")
    # 奖励反比于距离
    reward = gap_dist_norm ** (-0.5)
    return reward


def check_score(prompts, completions, answer, **kwargs):
    """
    # (1)可以成功提取A和B的得分
    # (2)两个得分的大小关系正确
    # (3)两个得分的数量关系正确
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    tb = PrettyTable()
    tb.title = "Reward Table"
    tb.field_names = ["Answer", "Extracted responses", "Reward", "Details"]
    scores = list()
    for pred_answer, true_answer in zip(responses, answer):
        score = 0
        score_detail = dict()
        #
        true_answer = extract_results(true_answer)
        level_A, rel_AB, level_B = true_answer['A'], true_answer['label'], true_answer['B']
        level_A, level_B = int(level_A), int(level_B)
        # 
        pred_answer = extract_results(pred_answer)
        pred_A, pred_rel_AB, pred_B = pred_answer["A"], pred_answer['label'], pred_answer["B"]
        #
        if pred_rel_AB in ['>', '≈', '<']:
            # 分类奖励
            if (rel_AB == ">" and pred_rel_AB == ">") or \
               (rel_AB == "<" and pred_rel_AB == "<") or \
               (rel_AB == "≈" and pred_rel_AB == "≈"):
                reward_cls = 1
                flag = True
            else:
                reward_cls = -3
                flag = False
            # 额外奖励
            try:
                reward_A = check_score_reward_dist_func(int(pred_A) - level_A)
                reward_B = check_score_reward_dist_func(int(pred_B) - level_B)
                reward_dist = check_score_reward_gap_func(flag, rel_AB, int(pred_A) - int(pred_B), level_A - level_B)
            except:
                reward_dist = 0
                reward_A = 0
                reward_B = 0
        else:
            reward_cls = -3
            reward_dist = 0
            reward_A = 0
            reward_B = 0

        reward_cls = round(reward_cls, 3)
        reward_dist = round(reward_dist, 3)
        reward_A = round(reward_A, 3)
        reward_B = round(reward_B, 3)
        score = round(reward_cls + reward_dist + reward_A + reward_B, 3)

        score_detail = {
            "reward_cls": reward_cls,
            "reward_dist": reward_dist,
            "reward_A": reward_A,
            "reward_B": reward_B,
        }
        tb.add_row([f"{level_A}{rel_AB}{level_B}", pred_answer, score, score_detail])
        scores.append(score)

    # logger.info("".join(['*'*20, f"\nQuestion:\n{question}", f"\nResponse:\n{responses[0]}"]))
    # logger.info("".join([f"\nResponse:\n{responses[0]}"]))
    logger.info(f"\n{tb}")
    return scores


def main():
    lora_rank = 32
    max_prompt_length = 1024 * 3
    max_seq_length = 1024 * 3 + 64  # Can increase for longer reasoning traces
    max_completion_length = max_seq_length - max_prompt_length

    logger.info("*** Loading model ***")
    model_id = f"Qwen-Instruct/Qwen2.5-14B-Instruct-merged-{args.domain.lower()}"
    # model_id = "Llama-Instruct/Llama-3.2-3B-Instruct-merged"
    # model_id = "Mistral-Instruct/Mistral-7B-Instruct-v0.3-merged"
    model_path = os.path.join(models_dir, model_id)
    
    # GRPO from scratch / GRPO based on the merged model obtained by SFT
    model, tokenizer = load_peft_model_from_scratch(model_path, max_seq_length, lora_rank)

    logger.info("*** Loading dataset ***")
    train_dataset, valid_dataset, test_dataset = get_papereval_data()
    logger.info(train_dataset[0]['prompt'][-1]['content'])
    logger.info(train_dataset[0]['answer'])
    
    #################
    # Training loop #
    #################
    logger.info("*** Train ***")
    training_args = GRPOConfig(
        output_dir = os.path.join("outputs", model_id),
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        # warmup_ratio = 0.01,
        temperature=1,
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.99,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_32bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 500,
        save_steps = 1000,
        max_grad_norm = 50,
        weight_decay = 0.01,
        report_to = "tensorboard", # Can use Weights & Biases
        seed=9527,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            check_score,
            # check_comment,
            # match_format_approximately,
        ],
        args = training_args,
        train_dataset = train_dataset,
    )
    trainer_stats = trainer.train()


if __name__ == "__main__":
    main()

