"""
This module defines the core training routine for fine‑uning causal language
models with LowRank Adapters (LoRA).
"""


import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from src.config import TrainingConfig
from src.data import ft_dataset_builder_map 


class BaseTrainer(Trainer):
    """
    A custom Trainer subclass that can dynamically load and train on either
    the MATH dataset, the BBQ dataset, or a combination of both.

    Parameters
    ----------
    config : TrainingConfig
        Hyperparameters and model/dataset paths controlling the training run.
    dataset_type : str, optional
        Which dataset to load for training.  Accepts ``"math"``, ``"bbq"``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        # Persist configuration and dataset selection for later reference
        self.config = config
        self.dataset_type = config.dataset_type

        # Ensure reproducibility
        set_seed(config.seed)

        # Load tokenizer and set padding token to EOS if not already defined
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        # Load base language model and inject LoRA adapters
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, 
                                                     torch_dtype=torch.bfloat16,
                                                     device_map={"": "cuda"}, 
                                                     attn_implementation="sdpa")
        model.config.use_cache = False
        # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.init_with_lora:
            lora_config = config.lora_config.get_lora_config()
            model = get_peft_model(model, lora_config)
            
        

        self.train_ds_builder = ft_dataset_builder_map[self.dataset_type.lower()](self.tokenizer)
        self.train_dataset = self.train_ds_builder.build_dataset()

        # Construct TrainingArguments.  Disable evaluation since we do not
        # include validation sets during fine‑tuning.
       
        # Call the base Trainer constructor with the prepared objects
        super().__init__(
            model=model,
            args=config.get_trainer_args(),
            train_dataset=self.train_dataset,
            tokenizer=tokenizer,
        )
        
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        loader = DataLoader(self.train_dataset, 
                            batch_size=self.config.per_device_batch_size, 
                            shuffle=True, 
                            collate_fn=self.train_ds_builder.collate_fn,
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory)
   
        return loader

    def merge_and_save_final_model(self, output_dir: str | None = None) -> None:
        """
        Persist the trained model and tokenizer to disk.

        LoRA adapters are merged into the base model before saving, ensuring the
        checkpoint contains all weights needed for inference [oai_citation:2‡huggingface.co](https://huggingface.co/docs/peft/main/en/developer_guides/lora#:~:text=Merge%20LoRA%20weights%20into%20the,base%20model).
        If output_dir is None, defaults to self.args.output_dir.

        Without merging, PEFT's save_pretrained() only saves the adapter weights,
        not the base model [oai_citation:3‡huggingface.co](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint#:~:text=When%20you%20call%20save_pretrained,saves%20three%20files%2C%20described%20below).
        """
        target_dir = output_dir or self.args.output_dir

        # If the model is a PEFT model, merge LoRA weights into the base model.
        model_to_save = self.model
        if hasattr(self.model, "merge_and_unload"):
            try:
                model_to_save = self.model.merge_and_unload()
            except Exception:
                model_to_save = self.model

        # Save the full (merged) model and tokenizer.
        model_to_save.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)
