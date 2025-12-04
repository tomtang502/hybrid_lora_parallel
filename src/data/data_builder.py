import os
import glob
import math
import torch
import resource # specific for linux/mac to handle file limits
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer
from src.config.constants import DATA_PATH, MODEL_PATH, DATASET_BASE_CHUNK_SIZE

def increase_rlimit():
    """
    Attempts to increase the maximum number of open file descriptors 
    to the hard limit allowed by the system.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard  # Try to set soft limit to the hard limit
        
        # If hard limit is unlimited or unreasonably high, cap it for safety
        if target == resource.RLIM_INFINITY:
            target = 65535 # A reasonable upper bound
            
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"System file limit increased from {soft} to {target}")
    except Exception as e:
        print(f"Warning: Could not increase system file limit: {e}")

class NemotronIterableDataset(IterableDataset):
    def __init__(self, data_source, tokenizer, chunk_size=2048, infinite=True, rank=0, world_size=1, seed=42):
        """
        Args:
            data_source (str): Path to a directory of .parquet files OR a single .parquet file.
            tokenizer: Hugging Face tokenizer object.
            chunk_size (int): Context window size. Must be a multiple of DATASET_BASE_CHUNK_SIZE.
            infinite (bool): If True, restarts the dataset iteration automatically when exhausted.
            rank (int): Current GPU rank (for DDP/FSDP).
            world_size (int): Total number of GPUs (for DDP/FSDP).
            seed (int): Random seed for shuffling.
        """
        # 0. PRE-FLIGHT CHECK
        # Increase file limits immediately to prevent "Too many open files"
        increase_rlimit()

        if chunk_size % DATASET_BASE_CHUNK_SIZE != 0:
            raise ValueError(f"chunk_size must be a multiple of {DATASET_BASE_CHUNK_SIZE}, got {chunk_size}")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.infinite = infinite
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.pad_token_id = tokenizer.pad_token_id

        if os.path.isdir(data_source):
            self.file_paths = sorted(glob.glob(os.path.join(data_source, "**/*.parquet"), recursive=True))
        elif os.path.isfile(data_source):
            self.file_paths = [data_source]
        else:
            raise ValueError(f"Invalid data_source: {data_source}")

        if not self.file_paths:
            raise ValueError(f"No parquet files found in {data_source}")

        # We index by Row Groups to allow DDP even on a single large file.
        self.work_units = []
        print(f"[Rank {self.rank}] Indexing parquet metadata for {len(self.file_paths)} files...")
        
        for fp in self.file_paths:
            try:
                # FIX: Use read_metadata instead of ParquetFile class instantiation.
                # read_metadata is stateless and closes the file handle immediately.
                metadata = pq.read_metadata(fp)
                num_groups = metadata.num_row_groups
                
                for rg_index in range(num_groups):
                    self.work_units.append((fp, rg_index))
            except Exception as e:
                print(f"Skipping corrupt file {fp}: {e}")

        print(f"[Rank {self.rank}] Found {len(self.work_units)} total row groups.")

    def __iter__(self):
        worker_info = get_worker_info()
        
        # Determine total number of parallel workers across the entire cluster
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        total_shards = self.world_size * num_workers_per_rank
        global_worker_id = (self.rank * num_workers_per_rank) + worker_id

        # --- DEBUG SAFEGUARD ---
        # If we are in infinite mode (training) but have fewer work units than workers,
        # standard sharding would leave some workers with 0 items, causing immediate StopIteration
        # and recursion errors in training loops that expect infinite data.
        # In this specific case, we relax sharding to ensure everyone gets data.
        if self.infinite and len(self.work_units) < total_shards:
            if global_worker_id == 0: # Only print once
                print(f"WARNING: Dataset units ({len(self.work_units)}) < Total Workers ({total_shards}). Disabling strict sharding to prevent empty workers.")
            my_work_units = self.work_units # Give everyone everything
        else:
            # SHARDING: Assign specific row groups to this specific worker
            # Slice format: list[start:end:step]
            my_work_units = self.work_units[global_worker_id::total_shards]
        
        if not my_work_units:
            # This should only happen if infinite=False and we genuinely ran out of shards for this worker
            return 

        while True:
            # ITERATION
            for file_path, row_group_idx in my_work_units:
                # Read specific row group
                try:
                    # We open the file here for reading the actual data.
                    # This happens during training, so only 1 file is open per worker at a time.
                    table = pq.ParquetFile(file_path).read_row_group(row_group_idx)
                    # Assuming column is named 'text' or 'content'. Adjust if necessary.
                    column_names = table.column_names
                    text_col = 'text' if 'text' in column_names else column_names[0]
                    
                    texts = table[text_col].to_pylist()

                    for text in texts:
                        if not text.strip():
                            continue
                        encodings = self.tokenizer(
                            text,
                            truncation=True,
                            max_length=self.chunk_size,
                            padding="max_length",
                            return_tensors="pt"
                        )

                        yield {
                            "input_ids": encodings["input_ids"].squeeze(0),
                            "attention_mask": encodings["attention_mask"].squeeze(0),
                            "labels": encodings["input_ids"].squeeze(0) # For Causal LM
                        }

                except Exception as e:
                    print(f"Error reading {file_path} RG {row_group_idx}: {e}")
                    continue
            
            # If not infinite, stop after one pass
            if not self.infinite:
                break

def get_dataloader(data_path, tokenizer, batch_size, chunk_size=2048, infinite=True, rank=0, world_size=1, num_workers=4):
    """
    Helper function to instantiate the dataset and loader.
    """
    dataset = NemotronIterableDataset(
        data_source=data_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        infinite=infinite,
        rank=rank,
        world_size=world_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True is often helpful for IterableDatasets to avoid respawning overhead
        # but requires careful handling if infinite=False. Since we default infinite=True, it's safe.
        persistent_workers=(num_workers > 0)
    )
    dataloader.pad_token_id = dataset.pad_token_id
    return dataloader

# --- DEBUG / USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. SETUP DUMMY ARGS
    RANK = 0
    WORLD_SIZE = 1
    BATCH_SIZE = 8
    CHUNK_SIZE = 16384 # Must be multiple of DATASET_BASE_CHUNK_SIZE
  
    print("--- initializing tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    print("--- creating dataloader ---")
    try:
        loader = get_dataloader(
            DATA_PATH, 
            tokenizer, 
            BATCH_SIZE, 
            chunk_size=CHUNK_SIZE,
            infinite=True,
            rank=RANK, 
            world_size=WORLD_SIZE
        )
        
        print(f"--- iterating (Debugging 2 batches with chunk_size={CHUNK_SIZE}) ---")
        for i, batch in enumerate(loader):
            print(f"Batch {i} Input Shape: {batch['input_ids'].shape}")
            
            if i >= 1: break
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nNote: Ensure you have run 'download_nemotron.py' first and have data in './nemotron_data'")