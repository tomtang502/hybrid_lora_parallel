import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging
logging.set_verbosity_error()

import argparse
from typing import Dict, Any, Optional, Tuple
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token="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")

# ==============================
#   TASK & METRIC DEFINITIONS
# ==============================
PRIMARY_METRICS = {
    "mmlu": ["acc"],
    "mmlu_pro": ["exact_match"],
    "gsm8k": ["exact_match", "acc"],
    "humaneval": ["pass@1"],
    "bbq": ["acc", "bias"],
    "bbh": ["exact_match"],
    "truthfulqa_gen": ["bleu_max", "bleu_acc"],
    "truthfulqa_mc1": ["acc"],
    "truthfulqa_mc2": ["acc"],
    "gpqa": ["acc"],
    "mbpp": ["pass_at_1"],
}


TASK_CFG = {
    "mmlu": {"num_fewshot": 5, "limit": None},
    "mmlu_pro": {"num_fewshot": 5, "limit": None},
    "gsm8k": {"num_fewshot": 4, "limit": None},
    "humaneval": {"num_fewshot": 0, "limit": None},
    "bbq": {"num_fewshot": 0, "limit": None},
    "bbh": {"num_fewshot": 3, "limit": None},
    "truthfulqa": {"num_fewshot": 0, "limit": None},
    "gpqa": {"num_fewshot": 5, "limit": None},
    "mbpp": {"num_fewshot": 0, "limit": None},
}

# ==============================
#   CORE FUNCTIONS
# ==============================

def choose_metric(task_name: str, metrics: Dict[str, Any]) -> Tuple[str, float]:
    for key in PRIMARY_METRICS.get(task_name, []):
        if key in metrics:
            return key, float(metrics[key])
    for k, v in metrics.items():
        try:
            return k, float(v)
        except Exception:
            pass
    raise RuntimeError(f"No numeric metric found for task={task_name}: {metrics}")


def load_model(pretrained: str, dtype: str = "float16"):
    print(f"Loading model: {pretrained} with dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch.float16 if dtype == "float16" else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def run_one_task(
    task_name: str,
    model: HFLM,
    limit: Optional[int] = None,
    num_fewshot: int = 0,
    apply_chat_template: bool = False,
    batch_size: str = "auto",
    confirm_run_unsafe_code: bool = False,
    verbosity: str = "ERROR",
) -> Dict[str, Any]:
    
    print(f"\nRunning task: {task_name}")
    print(f"   > apply_chat_template={apply_chat_template}")
    print(f"   > confirm_run_unsafe_code={confirm_run_unsafe_code}")
    print(f"   > num_fewshot={num_fewshot}, limit={limit}\n")

    kwargs = dict(
        model=model,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        batch_size=int(batch_size) if batch_size != "auto" else "auto",
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        apply_chat_template=apply_chat_template,
        verbosity=verbosity,
    )
    if limit is not None:
        kwargs["limit"] = limit

    out = evaluator.simple_evaluate(**kwargs)
    return out["results"]


def evaluate_model(
    checkpoint: str,
    tasks,
    dtype: str = "float16",
    apply_chat_template: bool = False,
    batch_size: str = "auto",
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    
    model, tokenizer = load_model(checkpoint, dtype=dtype)
    model_wrapper = HFLM(pretrained=model, tokenizer=tokenizer)

    for task in tasks:
        print(f"\n======== Evaluating Task: {task} ========")
        cfg = TASK_CFG.get(task, {"num_fewshot": 0, "limit": None})
        
        results_dict = run_one_task(
            task_name=task,
            model=model_wrapper,
            limit=cfg["limit"],
            num_fewshot=cfg["num_fewshot"],
            apply_chat_template=False,
            confirm_run_unsafe_code=True if task in ["humaneval", "mbpp"] else False,
            batch_size=batch_size,
        )

        if task in results_dict:
            metrics = results_dict[task]
            metric_name, metric_value = choose_metric(task, metrics)
            summary[task] = {
                "primary_metric": metric_name,
                "primary_score": float(metric_value),
                "all": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }

            print(f"{task} | {metric_name}: {metric_value:.4f}")
            for k, v in summary[task]["all"].items():
                if k != metric_name:
                    print(f"    - {k}: {v:.4f}")

        else:
            for subtask_name, metrics in results_dict.items():
                metric_name, metric_value = choose_metric(subtask_name, metrics)
                summary[subtask_name] = {
                    "primary_metric": metric_name,
                    "primary_score": float(metric_value),
                    "all": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                }

                print(f"{subtask_name} | {metric_name}: {metric_value:.4f}")
                for k, v in summary[subtask_name]["all"].items():
                    if k != metric_name:
                        print(f"    - {k}: {v:.4f}")

    return summary

# ==============================
#   MAIN ENTRY
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an LLM on benchmark tasks using lm_eval.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path, e.g. 'Qwen/Qwen2.5-1.5B'")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. 'mmlu', 'gsm8k'")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type: float16 or bfloat16")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size for evaluation")
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template (for chat models)")

    args = parser.parse_args()

    print("=======================================")
    print(f"Model: {args.model}")
    print(f"Task : {args.task}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch: {args.batch_size}")
    print(f"GPUs : {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print("=======================================")

    results = evaluate_model(
        checkpoint=args.model,
        tasks=[args.task],
        dtype=args.dtype,
        apply_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
    )

    print("\n======== Final Summary ========")
    for task, result in results.items():
        print(f"{task}")
        print(f"    Primary Metric: {result['primary_metric']} = {result['primary_score']:.4f}")
        print("    All Metrics:")
        for k, v in result["all"].items():
            print(f"        {k}: {v:.4f}")
    print("===============================")
