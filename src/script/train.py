"""
Entry point for finetuning a causal language model on the MATH and BBQ
datasets using LoRA adapters.  This script parses commandline arguments,
constructs a ``TrainingConfig`` instance, and invokes the training routine
implemented in ``trainer.py``.
"""

import draccus

from src.config import TrainingConfig
from src.trainer import BaseTrainer

@draccus.wrap()
def main(cfg: TrainingConfig):
    cfg.init_wandb()
    trainer = BaseTrainer(config=cfg)

    trainer.train()
    trainer.merge_and_save_final_model(cfg.ckpt_dir)
    
    


if __name__ == "__main__":
    main()