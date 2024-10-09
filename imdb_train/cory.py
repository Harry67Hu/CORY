# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
import random
import sys

import tyro
import wandb
from openai import OpenAI

# Set PYTHONPATH to th/trl/trl') # e root directory of the project, set working directory to the directory of the file
# sys.path.append('/home/trl/trl')

import os
import subprocess
def select_least_used_gpu():

    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8')
    gpu_memory = [int(x) for x in smi_output.strip().split('\n')]
    least_used_gpu = gpu_memory.index(min(gpu_memory))
    return least_used_gpu

least_used_gpu = select_least_used_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(least_used_gpu)

print(">> Chosen GPU: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import HfArgumentParser, pipeline, AutoTokenizer, BitsAndBytesConfig

from trl import PPOConfig, set_seed, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler
from utils.exp_tracker import ExperimentTracker
from utils.misc import train_test_val_split
from utils.model_generation import Model_Generator
from utils.text_generation import AgentManager, Ranker

tqdm.pandas()

random_name = str(random.random()).split('.')[1]
exp_class = 'extensive-game-' + random_name

# NOTE Set your base name here
base_name = '/home/trl/trl/hf_hub/models/gpt2-'

temp_model_name_1 = base_name + exp_class + '-1'
temp_model_name_2 = base_name + exp_class + '-2'
if not os.path.exists(temp_model_name_1):
    os.makedirs(temp_model_name_1)
if not os.path.exists(temp_model_name_2):
    os.makedirs(temp_model_name_2)


@dataclass
class ScriptArguments:
    # NOTE Set your model name and exp name here 
    model_name: str = "/home/trl/trl/hf_hub/models/gpt2"
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            tracker_kwargs={"wandb": {"entity": "Population-LLM",
                                      "name": f"ext-rl-llm2-{time.strftime('%m%d%H%M', time.localtime())}"}},
            tracker_project_name="imdb",
            model_name="/home/trl/trl/hf_hub/models/gpt2",
            query_dataset="/home/trl/trl/hf_hub/datasets/imdb",
            reward_model="sentiment-analysis:/home/trl/trl/hf_hub/models/lvwerra-distilbert-imdb",
            learning_rate=1.41e-5,
            log_with="wandb",
            mini_batch_size=256,
            batch_size=256,
            gradient_accumulation_steps=1,
            early_stopping=False,
            # target_kl=6.0,
            kl_penalty="full",
            seed=0,
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=None,
            init_kl_coef=0.4,
            # adap_kl_ctrl=False
        )
    )
    use_seq2seq: bool = False
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # Training loop control parameters
    eval_freq: Optional[int] = field(default=7, metadata={"help": "n steps to evaluate the model"})
    swap_freq: Optional[int] = field(default=5, metadata={"help": "n steps to swap the model"})

    # Optimization parameters
    adafactor: Optional[bool] = field(default=False, metadata={"help": "Use Adafactor optimizer"})

    # Duo training parameters
    temp_model_name_1: Optional[str] = field(default=temp_model_name_1,
                                             metadata={"help": "the temp model name of LLM-1(but may be changed)"})
    temp_model_name_2: Optional[str] = field(default=temp_model_name_2,
                                             metadata={"help": "the temp model name of LLM-2(but may be changed)"})
    reward_type: Optional[str] = field(default="independent", metadata={"help": "cooperative or competitive or cooperative-regular"})

    # Wandb grouping
    group: Optional[str] = field(default="imdb-extensive-game", metadata={"help": "Wandb grouping"})


args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 100, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


current_device = Accelerator().local_process_index

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

# model = trl_model_class.from_pretrained(
#     args.ppo_config.model_name,
#     trust_remote_code=args.trust_remote_code,
#     device_map=device_map,
#     peft_config=peft_config,
# )

tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

model_generator = Model_Generator(tokenizer, dataset, collator)

model_1, optimizer_1, ppo_trainer_1, device_1 = model_generator.generate_pretrained_model(
    args.ppo_config,
    peft_config,
    None,
    args,
    ref_model=ref_model,
    device_map=device_map)

model_2, optimizer_2, ppo_trainer_2, device_2 = model_generator.generate_pretrained_model(
    args.ppo_config,
    peft_config,
    None,
    args,
    ref_model=ref_model,
    device_map=device_map)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "repetition_penalty": 1.1,
}

task, model_name = args.ppo_config.reward_model.split(":")

sentiment_pipe = pipeline(task, model=model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

swap = False
data_loader = ppo_trainer_1.dataloader

# backup all code on wandb if you want
# arti_code = wandb.Artifact("code", type="code")
# arti_code.add_file("/home/trl/trl/imdb_train/cory.py")
# arti_code.add_dir("/home/trl/trl/eval", name="eval")
# arti_code.add_dir("/home/trl/trl/utils", name="utils")
# wandb.log_artifact(arti_code)

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

merge_template = 'I can make this sentence "{}" more positive: {}'

for epoch, batch in tqdm(enumerate(data_loader)):
    query_tensors = batch["input_ids"]

    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len
    # Get response from the observer
    response_tensors_1, ref_response_tensors = ppo_trainer_1.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response_llm1"] = tokenizer.batch_decode(response_tensors_1)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Get response from the pioneer based on q + r
    merged_queries = [merge_template.format(q + r, q) for q, r in zip(batch["query"], batch["response_llm1"])]
    merged_query_tensors = [torch.tensor(tokenizer.encode(merged)) for merged in merged_queries]
    response_tensors_2 = ppo_trainer_2.generate(
        merged_query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
    )
    batch["response_llm2"] = tokenizer.batch_decode(response_tensors_2)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response_llm1"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards_1 = [output[1]["score"] for output in pipe_outputs]

    texts = [q + r for q, r in zip(batch["query"], batch["response_llm2"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards_2 = [output[1]["score"] for output in pipe_outputs]

    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Construct game reward
    game_reward_1 = [torch.tensor(r1 + r2) for r1, r2 in zip(rewards_1, rewards_2)]
    game_reward_2 = [torch.tensor(r1 + r2) for r1, r2 in zip(rewards_1, rewards_2)]

    # Independent PPO update
    stats_1 = ppo_trainer_1.step(query_tensors, response_tensors_1, game_reward_1)
    stats_2 = ppo_trainer_2.step(merged_query_tensors, response_tensors_2, game_reward_2)

    # synthesis average reward = average_kl_penalty_per_step + average_task_reward_per_step
    average_kl_penalty_per_step = stats_2["ppo/mean_non_score_reward"]
    average_task_reward_per_response = sum(rewards_1) / len(rewards_1)
    average_new_tokens = stats_2["temp/average_new_tokens"]
    average_task_reward_per_step = average_task_reward_per_response / average_new_tokens
    average_reward = average_kl_penalty_per_step + average_task_reward_per_step
    # delete "temp/average_new_tokens" in dict
    stats_2.pop("temp/average_new_tokens")
    stats_2.pop("ppo/average_reward")
    
    # log data
    if swap:
        LLM1_task_reward_part = rewards_2
        LLM1_kl_penalty_part = [stats_2["tokens/responses_len_mean"] * stats_2["ppo/mean_non_score_reward"] for ii in range(len(rewards_2))]
        LLM1_index = [p1 + p2 for p1, p2 in zip(LLM1_task_reward_part, LLM1_kl_penalty_part)]

        LLM2_task_reward_part = rewards_1
        LLM2_kl_penalty_part = [stats_1["tokens/responses_len_mean"] * stats_1["ppo/mean_non_score_reward"] for ii in range(len(rewards_1))]
        LLM2_index = [p1 + p2 for p1, p2 in zip(LLM2_task_reward_part, LLM2_kl_penalty_part)]

        ppo_trainer_1.log_stats(stats_2,
                                batch,
                                rewards_2,
                                columns_to_log=["query", "response_llm1", "response_llm2", "ref_response", "ref_rewards"],
                                LLM1_index=LLM1_index,
                                LLM1_task_reward_part=LLM1_task_reward_part,
                                LLM1_kl_penalty_part=LLM1_kl_penalty_part,
                                LLM2_index=LLM2_index,
                                LLM2_task_reward_part=LLM2_task_reward_part,
                                LLM2_kl_penalty_part=LLM2_kl_penalty_part,
                                reward_pioneer=rewards_2,
                                reward_observer=rewards_1,
                                average_reward=[average_reward])
    else:
        LLM1_task_reward_part = rewards_1
        LLM1_kl_penalty_part = [stats_1["tokens/responses_len_mean"] * stats_1["ppo/mean_non_score_reward"] for ii in range(len(rewards_1))]
        LLM1_index = [p1 + p2 for p1, p2 in zip(LLM1_task_reward_part, LLM1_kl_penalty_part)]

        LLM2_task_reward_part = rewards_2
        LLM2_kl_penalty_part = [stats_2["tokens/responses_len_mean"] * stats_2["ppo/mean_non_score_reward"] for ii in range(len(rewards_2))]
        LLM2_index = [p1 + p2 for p1, p2 in zip(LLM2_task_reward_part, LLM2_kl_penalty_part)]

        ppo_trainer_1.log_stats(stats_1,
                                batch,
                                rewards_1,
                                columns_to_log=["query", "response_llm1", "response_llm2", "ref_response", "ref_rewards"],
                                LLM1_index=LLM1_index,
                                LLM1_task_reward_part=LLM1_task_reward_part,
                                LLM1_kl_penalty_part=LLM1_kl_penalty_part,
                                LLM2_index=LLM2_index,
                                LLM2_task_reward_part=LLM2_task_reward_part,
                                LLM2_kl_penalty_part=LLM2_kl_penalty_part,
                                reward_pioneer=rewards_1,
                                reward_observer=rewards_2,
                                average_reward=[average_reward])
        
      

    if (epoch + 1) % args.swap_freq == 0:
        # This is a version where exchanging large model parameters is used to achieve role swapping.
        swap = not swap
        # Save LLM-1 as temp-1, LLM-2 as temp-2
        model_1.save_pretrained(args.temp_model_name_1, push_to_hub=False)
        model_2.save_pretrained(args.temp_model_name_2, push_to_hub=False)
        # Load temp-2 as LLM-1, temp-1 as LLM-2
        model_1, optimizer_1, ppo_trainer_1, device_1 = model_generator.generate_peft_model(
            args.ppo_config,
            peft_config,
            None,
            args,
            args.temp_model_name_2,
            ref_model=ref_model,
            device_map=device_map
        )
        model_2, optimizer_2, ppo_trainer_2, device_2 = model_generator.generate_peft_model(
            args.ppo_config,
            peft_config,
            None,
            args,
            args.temp_model_name_1,
            ref_model=ref_model,
            device_map=device_map
        )

    if epoch > 100:
        break
