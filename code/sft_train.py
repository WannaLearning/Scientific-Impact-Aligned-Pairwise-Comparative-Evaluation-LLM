import os
import re
import sys
import torch
import logging
import argparse
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from trl import SFTTrainer, SFTConfig

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

logger.info(f"*** 正在使用领域数据: {args.domain.upper()} ***")

def load_peft_model_unsloth(model_id, max_seq_length=1024, lora_rank=32):
    # Load model
    logger.info("*** load_peft_model_unsloth: unsloth + qlora + sft ***")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = os.path.join(models_dir, model_id),
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        full_finetuning = False, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.90, # Reduce if out of memory
    )
    target_modules = ["q_proj", 'k_proj', "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=target_modules, # Remove QKVO if out of memory
        r=lora_rank, 
        lora_alpha=lora_rank * 1,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth", # Enable long context finetuning
        random_state=9527,
    )
    # print(model)
    return model, tokenizer


def transform_classify_data2sft_data(data, tokenizer, save_dir, save_file_name):
    """
    将classifcation data的数据格式转化成SFT data格式
    因为需要运用template, 所以要传入tokenizer
    """
    domain_adj = {
        "pubmed": "medical",
        "chemistry": "chemical",
    }

    def generate_conversation_sft_format(examples):
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
        # 
        conversations = list()
        for title_A, title_C, abstract_A, abstract_C, level_A, level_C, label in zip(titles_A, titles_C, abstracts_A, abstracts_C, level_A, level_C, labels):
            # SFT INPUT
            # question = (
            #     "***TASK Instruction***\n"
            #     "Your task is to evaluate and compare the research quality of Paper A and Paper B across three key dimensions: Originality, Significance, and Rigour."
            #     # Originality definition
            #     "(1) Originality will be understood as the extent to which the output makes an important and innovative contribution to understanding and knowledge in the field.\n"
            #     "Research outputs that demonstrate originality may do one or more of the following: "
            #     "produce and interpret new empirical findings or new material; engage with new and/or complex problems; "
            #     "develop innovative research methods, methodologies and analytical techniques; "
            #     "show imaginative and creative scope; "
            #     "provide new arguments and/or new forms of expression, formal innovations, interpretations and/or insights; "
            #     "collect and engage with novel types of data; "
            #     "and/or advance theory or the analysis of doctrine, policy or practice, and new forms of expressions.\n"
            #     # Significance definition
            #     "(2) Significance will be understood as the extent to which the work has influenced, or has the capacity to influence, knowledge and scholarly thought, or the development and understanding of policy and/or practice.\n"
            #     # Rigour definition
            #     "(3) Rigour will be understood as the extent to which the work demonstrates intellectual coherence and integrity, and adopts robust and appropriate concepts, analyses, sources, theories and/or methodologies.\n\n"
            #     # 
            #     "Specifically, you assess the Originality, Significance, and Rigour of the two papers by analyzing their provided titles and abstracts. "
            #     "Assign numerical scores (integers from 1 to 10) to each paper, denoted as SCORE_A and SCORE_B, respectively. "
            #     "The score disparity between the two papers should reflect the quality difference between them. "
                
            #     "Based on the SCORE_A and SCORE_B, determine the comparison result between Paper A and Paper B:\n"
            #     "If SCORE_A is at least 2 points higher than SCORE_B (e.g., 7 and 3), Paper A's quality is significantly superior to Paper B's, output A>B.\n"
            #     "If SCORE_A is at least 2 points lower than SCORE_B (e.g., 2 and 5), Paper A's quality is significantly inferior to Paper B's, output A<B.\n"
            #     "If the difference between SCORE_A and SCORE_B is no more than 1 point (e.g., 2 and 1), their rearch qualities are nearly comparable, output A≈B.\n"
                 
            #     "***INPUT***\n"
            #     f"Title of Paper A: {title_A}\nAbstract of Paper A: {abstract_A}\n\n"
            #     f"Title of Paper B: {title_C}\nAbstract of Paper B: {abstract_C}\n\n"

            #     "***OUTPUT***\n"
            #     "Provide only the results in the following output format without any other information:"
            #     "A=<D>;B=<SCORE_B>;<SCORE_A-SCORE_B><comparison result>;"
            #     "Example: A=2;B=8;A-B=-6;A<B;"
            # )

            question = (
                f"As an expert in {domain_adj[args.domain]} field, "
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

            conversations.append([
                {"role" : "user", "content" : question},
                {"role" : "assistant", "content" : response},
            ])
        return { "conversations": conversations, }

    data_transform = data.map(generate_conversation_sft_format, batched=True)

    jsonl_format = list()
    for i in range(len(data_transform)):
        question = data_transform[i]["conversations"][0]["content"]
        response = data_transform[i]["conversations"][1]["content"]
        oneline = {
            "pids_A": data_transform[i]["pids_A"],
            "pids_C": data_transform[i]["pids_C"],
            "conversations": [{"human": question, "assistant": response}], 
        }
        jsonl_format.append(oneline)
    os.makedirs(save_dir, exist_ok=True)
    utils.save_jsonl(jsonl_format, os.path.join(save_dir, save_file_name))
    logger.info(f"*** SFT Data case ***\n{question}\n{response}")
    
    data_SFT = tokenizer.apply_chat_template(
        data_transform["conversations"],
        tokenize = False,
    )
    data_SFT = pd.Series(data_SFT)
    data_SFT.name = "text"
    data_SFT = Dataset.from_pandas(pd.DataFrame(data_SFT))
    return data_SFT


def get_data_papereval(tokenizer):
    data_dir = f"/public/home/lab8/project/PaperOriginality/code/hsz/data/DI_SCORE_PREDICTION_DATA/{args.domain.upper()}_FUTURE/classification"
    train_data_path = os.path.join(data_dir, "train_dataset")
    valid_data_path = os.path.join(data_dir, "valid_dataset")
    test_data_path  = os.path.join(data_dir, "test_dataset")

    logger.info("*** Load Train/Valid/Test Dataset ***")
    train_dataset = load_from_disk(train_data_path)
    valid_dataset = load_from_disk(valid_data_path)
    test_dataset  = load_from_disk(test_data_path)

    logger.info("*** Transform Train/Valid/Test Dataset into SFT format ***")
    save_dir = f"/public/home/lab8/project/PaperOriginality/code/hsz/data/DI_SCORE_PREDICTION_DATA/{args.domain.upper()}_FUTURE/sft"
    train_dataset = transform_classify_data2sft_data(train_dataset, tokenizer, save_dir, "train.jsonl")
    valid_dataset = transform_classify_data2sft_data(valid_dataset, tokenizer, save_dir, "valid.jsonl")
    test_dataset  = transform_classify_data2sft_data(test_dataset, tokenizer, save_dir, "test.jsonl")

    # 统计样本数目
    logger.info("*** Train set ***")
    logger.info(train_dataset)
    logger.info("*** Validation set ***")
    logger.info(valid_dataset)
    logger.info("*** Test set ***")
    logger.info(test_dataset)
    return train_dataset, valid_dataset, test_dataset


def main():
    lora_rank = 32
    max_prompt_length = 1024 * 3
    max_seq_length = 1024 * 3 + 64  # Can increase for longer reasoning traces
    max_completion_length = max_seq_length - max_prompt_length

    logger.info("*** Loading model ***")
    model_id = "Qwen-Instruct/Qwen2.5-14B-Instruct"
    model, tokenizer = load_peft_model_unsloth(model_id, max_seq_length=max_seq_length, lora_rank=lora_rank)

    logger.info("*** Loading dataset ***")
    train_dataset, valid_dataset, test_dataset = get_data_papereval(tokenizer)

    #################
    # Training loop #
    #################
    logger.info("*** Train ***")
    training_args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 16, # Use GA to mimic batch size!
            warmup_steps = 5,
            learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            num_train_epochs = 3,
            save_steps = 1000,
            optim = "paged_adamw_32bit",
            max_grad_norm=50,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 9527,
            report_to = "tensorboard", # Use this for WandB etc
            output_dir = os.path.join("outputs", model_id),
            dataset_num_proc=64,
    )
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = training_args,
    )
    # trainer_stats = trainer.train()


if __name__ == "__main__":
    main()