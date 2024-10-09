import torch
from transformers import Adafactor

from trl import AutoModelForCausalLMWithValueHead, PPOTrainer


class Model_Generator:
    def __init__(self, tokenizer, dataset, collator):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collator = collator

    def generate_pretrained_model(self, ppo_config, lora_config, nf4_config, script_args, ref_model=None, device_map='auto'):
        if not hasattr(script_args, 'ppo_config'):
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                ppo_config.model_name,
                quantization_config=nf4_config,
                # torch_dtype=torch.float16,
                device_map=device_map,
                peft_config=lora_config,
                # ignore_mismatched_sizes=True
            )
        else:

            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                script_args.model_name,
                quantization_config=nf4_config,
                # torch_dtype=torch.float16,
                device_map=device_map,
                peft_config=lora_config,
                # ignore_mismatched_sizes=True
            )
        # if tokenizer is not None:
            # self.tokenizer = tokenizer

        optimizer = None  # default Adam
        if script_args.adafactor:
            optimizer = Adafactor(
                filter(lambda p: p.requires_grad, model.parameters()),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=ppo_config.learning_rate,
            )

        ppo_trainer = PPOTrainer(
            ppo_config,
            model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            data_collator=self.collator,
            optimizer=optimizer,
        )

        device = ppo_trainer.accelerator.device
        # if ppo_trainer.accelerator.num_processes == 1:
        #     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

        return model, optimizer, ppo_trainer, device

    def generate_peft_model(self, ppo_config, lora_config, nf4_config, script_args, model_name, ref_model=None, device_map='auto'):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            # torch_dtype=torch.float16,
            device_map=device_map,
            peft_config=lora_config,
            # ignore_mismatched_sizes=True
            # use_safetensors=False
        )

        optimizer = None  # default Adam
        if script_args.adafactor:
            optimizer = Adafactor(
                filter(lambda p: p.requires_grad, model.parameters()),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=ppo_config.learning_rate,
            )
        ppo_trainer = PPOTrainer(
            ppo_config,
            model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            data_collator=self.collator,
            optimizer=optimizer,
        )

        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

        return model, optimizer, ppo_trainer, device
