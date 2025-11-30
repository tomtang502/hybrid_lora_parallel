from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from src.data.prompts import CODEFEEDBACK_PROMPT

class CodeFeedbackDatasetBuilder:
    """
    Build a dataset for m-a-p/CodeFeedback-Filtered-Instruction.
    Only the 'Answer' part is used for loss computation (masked labeling).
    Implements Dynamic Padding per batch AND Smart Filtering for CruxEval.
    """

    def __init__(self, tokenizer, dataset_name="m-a-p/CodeFeedback-Filtered-Instruction", max_len_limit=None, ignore_idx=-100, data_step=1, smart_filter=True):
        self.dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        self.tokenizer = tokenizer
        self.ignore_idx = ignore_idx
        
        # --- LOGIC CHANGE START ---
        # Prioritize Smart Filtering over random slicing (data_step)
        if smart_filter:
            print(f"Original size: {len(self.dataset)}")
            print("Applying Smart Filter (Python + Execution logic)...")
            self.dataset = self.dataset.filter(self._filter_logic)
            print(f"Filtered size: {len(self.dataset)}")
        if data_step > 1:
            # Fallback to stride slicing if filter is OFF
            print(f"Applying stride slicing (step={data_step})...")
            self.dataset = self.dataset.select(range(0, len(self.dataset), data_step))
        # --- LOGIC CHANGE END ---

        # Use EOS token as padding token if none is provided
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # We still compute max_length to use it as a TRUNCATION limit,
        # ensuring no single sample exceeds the context window.
        self.truncation_max_len = self._compute_maxlen(limit=max_len_limit)

    def _filter_logic(self, ex):
        """
        [NEW] Filter logic for CruxEval Optimization.
        Keep only samples that are:
        1. Python related
        2. Contain elements of execution logic (print, return, output, trace)
        """
        # 1. Check metadata language column if it exists
        if 'lang' in ex and ex['lang'] is not None:
            if ex['lang'].lower() not in ['python', 'py']:
                return False

        # 2. Text Search for relevance
        query_lower = ex['query'].lower()
        answer_lower = ex['answer'].lower()

        # Keywords that indicate Python or Execution reasoning
        is_python = 'python' in query_lower or 'def ' in ex['answer']
        
        # Keywords that indicate trace/execution (CruxEval style)
        has_logic = any(kw in answer_lower for kw in ['output', 'return', 'print', 'execution', 'trace', 'result'])

        return is_python and has_logic

    def _compute_maxlen(self, limit=None):
        """Compute the maximum token length across samples to determine truncation threshold."""
        lengths = []
        
        # If limit is set, only check the first N samples
        # Ensure we don't select out of bounds if dataset shrank due to filtering
        limit = min(limit, len(self.dataset)) if limit else len(self.dataset)
        data_to_scan = self.dataset.select(range(limit))

        for ex in tqdm(data_to_scan, desc="Computing global max token length for truncation"):
            text = (CODEFEEDBACK_PROMPT + 
                   "\n### Instruction:\n" + ex["query"] + 
                   "\n\n### Response:\n" + ex["answer"])
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
        
        # Add a small buffer
        return int(np.max(lengths)) + 16

    def preprocess(self, ex):
        """
        Tokenize and mask labels. 
        NOTE: This does NOT pad. It only truncates to global max len.
        """
        
        prompt = CODEFEEDBACK_PROMPT + "\n### Instruction:\n" + ex["query"] + "\n\n### Response:"
        answer = " " + ex["answer"]

        # Tokenize separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        # Combine
        input_ids = prompt_ids + answer_ids
        
        # Mask out prompt tokens
        labels = [self.ignore_idx] * len(prompt_ids) + answer_ids

        # Truncate if exceeding the global calculated max_length (to prevent OOM)
        if len(input_ids) > self.truncation_max_len:
            input_ids = input_ids[:self.truncation_max_len]
            labels = labels[:self.truncation_max_len]
        
        # Note: We return raw lists, not tensors, and NO padding yet.
        return {
            "input_ids": input_ids,
            "labels": labels,
            # We don't strictly need to create attention_mask here as it's just 1s,
            # but we pass it for consistency.
            "attention_mask": [1] * len(input_ids)
        }

    def build_dataset(self, num_proc=16):
        """Tokenize the entire training dataset."""
        dataset = self.dataset.map(
            self.preprocess,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=num_proc
        )
        return dataset
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle Dynamic Padding.
        Pads batch to the length of the longest sequence IN THIS BATCH.
        """
        # Convert lists to tensors
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

        # Dynamic Padding using pad_sequence
        # batch_first=True results in [Batch, Seq_Len]
        
        # 1. Pad Input IDs with pad_token_id
        input_ids_padded = rnn_utils.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # 2. Pad Labels with ignore_idx (-100) so loss isn't computed on pads
        labels_padded = rnn_utils.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_idx
        )

        # 3. Pad Attention Mask with 0
        attention_masks_padded = rnn_utils.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_padded,
        }

if __name__ == "__main__":
    # Example usage
    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 4}
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Initialize builder with SMART FILTER enabled
    print("Initializing builder...")
    builder = CodeFeedbackDatasetBuilder(tokenizer=tokenizer, smart_filter=True, data_step=2)
    
    print(f"Global Truncation Limit: {builder.truncation_max_len}")
    
    dataset = builder.build_dataset(num_proc=16)

    # DataLoader
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=builder.collate_fn)
    print(f"Total Batches: {len(loader)}")
    
    # Validation loop
    length_L = []
    print("\n--- Inspecting Batch Widths (Dynamic Padding Check) ---")
    for i, batch in enumerate(loader):
        curr_width = batch["input_ids"].shape[1]
        length_L.append(curr_width)
        
        # Print info for first few batches to show variance
        if i < 5:
            print(f"Batch {i}: Shape {batch['input_ids'].shape} | Max Len in this batch: {curr_width}")
            
        if i == 0:
            print("\n[Batch 0 Check] Last 5 label tokens (should be -100 if padded):")
            print(batch["labels"][0][-5:])

        if i >= 20: break 
        
    print(f"\nStats (First 20 batches):")
    print(f"Widest Batch: {max(length_L)}")
    print(f"Narrowest Batch: {min(length_L)}")
    print(f"Average Batch Width: {sum(length_L)/len(length_L):.2f}")
    print(f"(Note: Average should be significantly lower than Global Limit {builder.truncation_max_len})")