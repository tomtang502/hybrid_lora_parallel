from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

class BBQDatasetBuilder:
    """
    Build a dataset for the heegyu/bbq benchmark for fine-tuning.
    Aggregates all subsets using the specific parquet revision.
    Only the 'Answer' part is used for loss computation (masked labeling).
    """

    ALL_SUBSETS = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation",
        "Race_x_SES",
        "Race_x_gender",
    ]

    def __init__(self, tokenizer, dataset_path="heegyu/bbq"):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and concatenate all subsets using the specific revision and data_dir
        # as seen in your original script
        print(f"Loading all subsets from {dataset_path} (parquet revision)...")
        all_ds = []
        for subset in tqdm(self.ALL_SUBSETS, desc="Loading Subsets"):
            try:
                # Using the specific revision and data_dir structure that works for you
                ds = load_dataset(
                    dataset_path, 
                    data_dir=subset, 
                    revision="refs/convert/parquet",
                    trust_remote_code=True
                )["test"]
                all_ds.append(ds)
            except Exception as e:
                print(f"Warning: Could not load subset {subset}: {e}")
        
        if not all_ds:
            raise ValueError(f"No datasets were loaded from {dataset_path}. Check connection.")

        self.dataset = concatenate_datasets(all_ds)
        print(f"Dataset loaded. Columns: {self.dataset.column_names}")
        
        self.max_length = self._compute_maxlen()

    def _compute_maxlen(self):
        """Compute the maximum token length across all samples."""
        lengths = []
        for ex in tqdm(self.dataset, desc="Computing max token length"):
            # Using ans0, ans1, ans2 directly as per the heegyu/bbq schema
            correct_answer_text = [ex['ans0'], ex['ans1'], ex['ans2']][ex['label']]
            
            text = (
                f"Context: {ex['context']}\n"
                f"Question: {ex['question']}\n"
                f"A. {ex['ans0']}\nB. {ex['ans1']}\nC. {ex['ans2']}\n"
                f"Answer: {correct_answer_text}"
            )
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
        
        return int(np.max(lengths))

    def preprocess(self, ex):
        """Tokenize, with loss only on the solution part."""
        correct_answer_text = [ex['ans0'], ex['ans1'], ex['ans2']][ex['label']]

        # 1. Format the Prompt (Context + Question + Options)
        prompt = (
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"A. {ex['ans0']}\nB. {ex['ans1']}\nC. {ex['ans2']}\n"
            f"Answer:"
        )
        
        # 2. Format the Answer
        answer = f" {correct_answer_text}"

        # 3. Tokenize separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        # 4. Combine
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)

        # 5. Mask out prompt tokens (-100 ignored by CrossEntropyLoss)
        labels = [-100] * len(prompt_ids) + answer_ids

        # 6. Pad or truncate to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def build_dataset(self):
        """Tokenize the entire dataset."""
        dataset = self.dataset.map(
            self.preprocess,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )
        return dataset

    @staticmethod
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }

if __name__ == "__main__":
    # Example usage
    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 2}
    
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Initialize Builder
    # Using defaults which now point to heegyu/bbq with correct revision
    builder = BBQDatasetBuilder(tokenizer=tokenizer)
    
    # Build
    dataset = builder.build_dataset()

    # DataLoader
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=builder.collate_fn)
    print(f"Total batches: {len(loader)}")
    
    length_L = []
    for batch in loader:
        length_L.append(batch["input_ids"].shape[1])
        if len(length_L) == 1:
            print("input_ids shape:", batch["input_ids"].shape)
            print("labels shape:", batch["labels"].shape)
            print("attention_mask shape:", batch["attention_mask"].shape)
            print("First sample labels (head):", batch["labels"][0][:10])
            break
            
    print("max_length for this run:", max(length_L))
    
