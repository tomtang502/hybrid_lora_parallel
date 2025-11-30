from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from src.data.prompts import MATH_COT_PROMPT

class HendrycksMathDatasetBuilder:
    def __init__(self, tokenizer, dataset_name="nlile/hendrycks-MATH-benchmark"):
        self.dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        self.tokenizer = tokenizer
        
        # Qwen/Llama often have no pad token, so we use EOS.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _find_last_boxed_string(self, text):
        start_idx = text.rfind("\\boxed")
        if start_idx == -1:
            return None
        i = start_idx + len("\\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            return None
        balance = 0
        end_idx = -1
        for j in range(i, len(text)):
            if text[j] == "{":
                balance += 1
            elif text[j] == "}":
                balance -= 1
            if balance == 0:
                end_idx = j + 1
                break
        if end_idx != -1:
            return text[start_idx:end_idx]
        return None

    def format_solution(self, solution):
        boxed_str = self._find_last_boxed_string(solution)
        if boxed_str is None:
            boxed_str = "Cannot Determine" 
        
        return (
            f"{MATH_COT_PROMPT}\n"
            f"<think>\n{solution}\n</think>\n"
            f"<answer>\n{boxed_str}\n</answer>"
        )

    def preprocess(self, ex):
        formatted_target = self.format_solution(ex['solution'])
        formatted_target += self.tokenizer.eos_token
        
        prompt_text = f"Problem:\n{ex['problem'].strip()}\nSolution:\n"
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(formatted_target, add_special_tokens=False)
        
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            # We explicitly compute length here for the collator
            "length": len(input_ids) 
        }

    def build_dataset(self):
        return self.dataset.map(
            self.preprocess,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing MATH",
            num_proc=4
        )
    
    def collate_fn(self, batch):
        # 1. Pad Inputs and Labels
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        
        # Note: pad_sequence pads to the RIGHT by default
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # 2. Create Attention Mask based on LENGTHS
        # (This fixes the issue where valid EOS tokens were getting masked out)
        seq_lens = torch.tensor([b["length"] for b in batch])
        max_len = input_ids_padded.shape[1]
        
        # Create a [Batch, MaxLen] mask
        # range_tensor: [0, 1, 2, ... max_len-1]
        range_tensor = torch.arange(max_len).unsqueeze(0).expand(len(batch), max_len)
        # Mask is 1 where index < length, 0 otherwise
        attention_mask = (range_tensor < seq_lens.unsqueeze(1)).long()
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

# --- Fixed Visualization Block ---
if __name__ == "__main__":
    class Colors:
        PROMPT = '\033[93m'
        TARGET = '\033[96m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 4}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    builder = HendrycksMathDatasetBuilder(tokenizer=tokenizer)
    dataset = builder.build_dataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=builder.collate_fn, shuffle=True)
    
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"\n{Colors.BOLD}--- Inspecting Data Samples ---{Colors.ENDC}")
    
    for i in range(len(input_ids)):
        print(f"\n{Colors.BOLD}{'='*40} Sample {i+1} {'='*40}{Colors.ENDC}")
        
        # Logic to separate Prompt, Target, and Padding
        # 1. Identify valid targets (where labels are NOT -100)
        is_target = labels[i] != -100
        
        # 2. Identify padding (where input is PAD token AND label is -100)
        # Note: We must be careful not to catch the prompt tokens which also have label -100
        # Simplest way: The attention mask tells us what is REAL vs PADDING
        is_real_token = batch["attention_mask"][i] == 1
        
        # Prompt = Real Token AND Not Target
        is_prompt = is_real_token & (~is_target)
        
        prompt_tokens = input_ids[i][is_prompt]
        target_tokens = input_ids[i][is_target]
        
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
        
        print(f"{Colors.PROMPT}[PROMPT / CONTEXT]:\n{prompt_text}{Colors.ENDC}")
        print(f"{Colors.TARGET}[TARGET / LEARNED]:\n{target_text}{Colors.ENDC}")