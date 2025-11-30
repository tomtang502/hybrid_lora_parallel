import argparse
import os
import random
import re
import sys
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from datasets import load_dataset
from src.data.prompts import MATH_COT_PROMPT, MATH_N_SHOT_DEMOS_PREFIX, MATH_N_SHOT_DEMOS_SUFFIX

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

try:
    import sympy
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False

# -----------------------------------------------------------------------------
# 2. Robust Answer Extraction (The Logic Change)
# -----------------------------------------------------------------------------

def extract_last_boxed(text: str) -> Optional[str]:
    """
    Finds the *last* \boxed{...} in the string using brace counting.
    Handles nested braces like \boxed{\frac{a}{b}} which regex fails on.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    
    # Move index to the opening brace '{' of \boxed{
    i = idx + len("\\boxed")
    
    # Skip any whitespace between \boxed and {
    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text) or text[i] != "{":
        return None
    
    # Stack-based parsing to find the matching closing brace
    balance = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            balance += 1
        elif text[j] == "}":
            balance -= 1
        
        if balance == 0:
            # Found the closing brace
            return text[start+1 : j] # Content inside braces
            
    return None

def extract_answer(text: str) -> Optional[str]:
    """
    Two-step extraction:
    1. If <answer> tags exist, restrict search to that block.
    2. Otherwise, search the whole text (Fallback for base model).
    3. Use brace-counting to find the last \boxed{...}.
    """
    # 1. Try to isolate the <answer> block
    # re.DOTALL allows matching across newlines
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    if tag_match:
        content_to_search = tag_match.group(1)
    else:
        # Fallback: Model didn't use tags (common in zero-shot base models)
        content_to_search = text

    # 2. Extract the boxed content
    return extract_last_boxed(content_to_search)

# -----------------------------------------------------------------------------
# 3. Equivalence Checking (Unchanged)
# -----------------------------------------------------------------------------

def are_answers_equivalent(pred: str, truth: str) -> bool:
    if pred is None or truth is None:
        return False
    
    def normalise(s: str) -> str:
        # Aggressive normalization
        s = str(s).replace("\n", "").replace(" ", "").replace("\\,", "")
        return s.strip(" .")
    
    pred_norm = normalise(pred)
    truth_norm = normalise(truth)
    
    if pred_norm == truth_norm:
        return True
    
    if HAVE_SYMPY:
        try:
            # simple string equality failed, try symbolic
            pred_expr = sympy.simplify(pred_norm)
            truth_expr = sympy.simplify(truth_norm)
            # Check if difference is zero
            return sympy.simplify(pred_expr - truth_expr) == 0
        except Exception:
            pass
    return False

# -----------------------------------------------------------------------------
# 4. Prompt Construction
# -----------------------------------------------------------------------------

def build_prompt(demos: List[Tuple[str, str]], problem: str) -> str:
    parts: List[str] = []
    
    # 1. Few-Shot Demos (Optional)
    if len(demos) > 0:
        parts.append(f"{MATH_N_SHOT_DEMOS_PREFIX}\n")
    for demo_problem, demo_solution in demos:
        parts.append(f"Problem:\n{demo_problem.strip()}\n")
        # For demos, we just dump the solution string. 
        # Ideally, demos should match the training format, but raw text works okay for base models.
        parts.append(f"Solution:\n{demo_solution.strip()}\n\n")
    if len(demos) > 0:
        parts.append(f"{MATH_N_SHOT_DEMOS_SUFFIX}\n")
    
    # 2. The Actual Problem
    parts.append(f"Problem:\n{problem.strip()}\n")
    
    # 3. The Trigger
    # We prime the model by writing "Solution:" followed by the training trigger phrase.
    parts.append(f"Solution:\n{MATH_COT_PROMPT}\n")
    
    return "".join(parts)

# -----------------------------------------------------------------------------
# 5. Evaluation Loop
# -----------------------------------------------------------------------------

def evaluate_model(model_name: str,
                   dataset_name: str,
                   n_shots: int,
                   max_problems: Optional[int],
                   seed: int,
                   device: Optional[str],
                   batch_size: int,
                   temperature: float,
                   max_new_tokens: int) -> None:
    set_seed(seed=seed)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading dataset '{dataset_name}'...", file=sys.stderr)
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Prepare Demos
    demos = []
    if n_shots > 0:
        demo_indices = random.sample(range(len(train_data)), n_shots)
        demos = [(train_data[i]["problem"], train_data[i]["solution"]) for i in demo_indices]

    # Select Test Indices
    eval_indices = list(range(len(test_data)))
    random.shuffle(eval_indices)
    if max_problems is not None:
        eval_indices = eval_indices[:max_problems]

    print(f"Loading model '{model_name}' on {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure padding side is left for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    # --- Stop Token Setup ---
    # We want to stop at <|endoftext|> OR </answer>
    stop_token_ids = [tokenizer.eos_token_id]
    
    # Check if </answer> is in vocab (it might be split into tokens if not added specially)
    # Ideally, we just check if it generates the closing tag. 
    # For simplicity in this script, we rely on EOS, but you can add explicit IDs if known.
    # Qwen typically handles text-based stopping via generation config or custom criteria, 
    # but strictly adding token IDs is safer.
    closing_tag_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    if len(closing_tag_ids) == 1:
        stop_token_ids.append(closing_tag_ids[0])

    correct = 0
    total = 0

    print(f"Starting evaluation of {len(eval_indices)} problems...", file=sys.stderr)

    for start in tqdm(range(0, len(eval_indices), batch_size)):
        batch_indices = eval_indices[start:start + batch_size]
        batch_prompts = []
        batch_truths = []
        
        for idx in batch_indices:
            prob = test_data[idx]["problem"]
            sol = test_data[idx]["solution"]
            
            # Use the aligned prompt builder
            prompt = build_prompt(demos, prob)
            batch_prompts.append(prompt)
            
            # Extract ground truth using the new robust function too (sanity check)
            batch_truths.append(extract_last_boxed(sol))

        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False if temperature == 0 else True,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode
        # We only care about the *new* tokens
        input_len = inputs["input_ids"].shape[1]
        
        for i, output_ids in enumerate(gen_ids):
            generated_ids = output_ids[input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 1. Extract Prediction
            pred_ans = extract_answer(generated_text)
            
            # 2. Compare
            truth_ans = batch_truths[i]
            is_correct = are_answers_equivalent(pred_ans, truth_ans)
            
            correct += int(is_correct)
            total += 1
            
            # Optional: Debug Print for first few
            if total <= 2:
                print(f"\n--- Debug Sample {total} ---")
                print(f"Prompt:\n{batch_prompts[i]}")
                # print(f"Prompt Tail: ...{batch_prompts[i][-100:]}")
                print(f"Generated (First 100): {generated_text[:100]}...")
                print(f"Extracted: {pred_ans}")
                print(f"Truth: {truth_ans}")
                print(f"Correct: {is_correct}")

    accuracy = correct / max(1, total)
    print(f"\nEvaluated {total} problems.")
    print(f"Accuracy: {accuracy:.2%}")

# -----------------------------------------------------------------------------
# 6. Main Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Base/Fine-tuned LLM on MATH")
    parser.add_argument("--model", type=str, required=True, help="Path to model or HF hub name")
    parser.add_argument("--dataset", type=str, default="nlile/hendrycks-MATH-benchmark")
    parser.add_argument("--n-shots", type=int, default=0, help="Zero-shot is recommended for fine-tuned models")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="Greedy decoding usually best for math")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    
    evaluate_model(
        args.model,
        args.dataset,
        args.n_shots,
        args.max_problems,
        args.seed,
        args.device,
        args.batch_size,
        args.temperature,
        args.max_new_tokens,
    )

if __name__ == "__main__":
    main()