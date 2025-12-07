import os
import glob
import resource
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer

# Assuming these are defined in your src.config.constants
from src.config.constants import DATA_DIR_PATH, MODEL_PATH, DATASET_BASE_CHUNK_SIZE
# defining defaults for this snippet to be runnable standalone:

def increase_rlimit():
    """
    Attempts to increase the maximum number of open file descriptors.
    Crucial for high concurrency data loading.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard
        if target == resource.RLIM_INFINITY:
            target = 65535
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as e:
        # Don't crash if we can't change this, just warn.
        print(f"Warning: Could not increase system file limit: {e}")

class NemotronIterableDataset(IterableDataset):
    def __init__(self, data_source, tokenizer, chunk_size=2048, infinite=True, rank=0, world_size=1, seed=42):
        """
        Args:
            data_source (str): Path to directory of .parquet files.
            tokenizer: HF tokenizer.
            chunk_size (int): Context window.
            infinite (bool): Loop forever (for training).
            rank (int): Global rank.
            world_size (int): Global world size.
        """
        increase_rlimit()

        if chunk_size % DATASET_BASE_CHUNK_SIZE != 0:
            raise ValueError(f"chunk_size must be a multiple of {DATASET_BASE_CHUNK_SIZE}, got {chunk_size}")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.infinite = infinite
        self.rank = rank
        self.world_size = world_size
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # 1. DISCOVERY: Find all files
        if os.path.isdir(data_source):
            all_files = sorted(glob.glob(os.path.join(data_source, "**/*.parquet"), recursive=True))
        elif os.path.isfile(data_source):
            all_files = [data_source]
        else:
            raise ValueError(f"Invalid data_source: {data_source}")

        if not all_files:
            raise ValueError(f"No parquet files found in {data_source}")

        # 2. RANK PARTITIONING: Assign files to THIS rank
        # Use simple striding: File 0 -> Rank 0, File 1 -> Rank 1... File N -> Rank 0
        self.my_files = all_files[self.rank::self.world_size]

        if not self.my_files:
            print(f"[Rank {self.rank}] WARNING: World size ({self.world_size}) > File count ({len(all_files)}). This rank has NO data and will idle.")
        else:
            print(f"[Rank {self.rank}] Assigned {len(self.my_files)} files (Indices: {list(range(self.rank, len(all_files), self.world_size))})")

        # 3. INDEXING: Only read metadata for MY files
        # We break files down into "Row Groups" (chunks inside the parquet file).
        # This is the atomic unit of work for a Worker.
        self.work_units = []
        
        for fp in self.my_files:
            try:
                # Lightweight metadata read (does not load data)
                metadata = pq.read_metadata(fp)
                num_groups = metadata.num_row_groups
                for rg_index in range(num_groups):
                    self.work_units.append((fp, rg_index))
            except Exception as e:
                print(f"[Rank {self.rank}] Skipping corrupt file {fp}: {e}")

        # Shuffle work units deterministically if needed, though sequential reads usually faster for disk
        # random.Random(seed).shuffle(self.work_units)

    def __iter__(self):
        # 4. WORKER PARTITIONING
        # Now we are inside a specific Process (Worker).
        # We need to split self.work_units among the workers of this rank.
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            my_work_units = self.work_units
            worker_id = 0
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Slice the list: start:stop:step
            # Worker 0 gets [0, 4, 8...], Worker 1 gets [1, 5, 9...]
            my_work_units = self.work_units[worker_id::num_workers]

        if not my_work_units:
            return

        while True:
            # Main Training Loop
            for file_path, row_group_idx in my_work_units:
                try:
                    # 5. SAFE CONCURRENT READ
                    # We instantiate ParquetFile HERE.
                    # Because PyTorch DataLoader workers are separate processes (Process Forking),
                    # opening the file here ensures a unique file descriptor for this process.
                    # This is thread-safe and process-safe.
                    pf = pq.ParquetFile(file_path)
                    table = pf.read_row_group(row_group_idx)
                    
                    # Detect text column automatically
                    column_names = table.column_names
                    text_col = 'text' if 'text' in column_names else column_names[0]
                    
                    # Convert to python list (expensive but easiest for tokenization)
                    texts = table[text_col].to_pylist()

                    for text in texts:
                        if not text or not text.strip():
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
                            # For Causal LM, labels are usually the same as inputs
                            "labels": encodings["input_ids"].squeeze(0) 
                        }

                except Exception as e:
                    print(f"[Rank {self.rank} Worker {worker_id}] Error reading {file_path} RG {row_group_idx}: {e}")
                    continue
            
            if not self.infinite:
                break

def get_dataloader(data_path, tokenizer, batch_size, chunk_size=2048, infinite=True, rank=0, world_size=1, num_workers=4):
    dataset = NemotronIterableDataset(
        data_source=data_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        infinite=infinite,
        rank=rank,
        world_size=world_size
    )
    loader =  DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True keeps the workers alive between epochs.
        # This prevents the overhead of respawning processes and re-opening parquet files.
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    loader.pad_token_id = dataset.pad_token_id
    return loader

if __name__ == "__main__":
    BATCH_SIZE = 4
    RANK = 0         # <--- Single Rank
    WORLD_SIZE = 1   # <--- Single World (No DDP)
    NUM_WORKERS = 2  # Test multiprocessing (CPU workers)
    CHUNK_SIZE = 2048

    print(f"--- [Main] Initializing for Rank {RANK}/{WORLD_SIZE} ---")

    try:
        # 1. Load Tokenizer
        print(f"-> Loading tokenizer from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Robustness fix: Ensure pad_token exists (Llama/Mistral often lack this)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   (Note: Pad token was None, set to EOS token)")

        # 2. Create Loader
        # This will scan all files in DATA_DIR_PATH, assuming Rank 0 owns them all.
        print(f"-> Creating DataLoader reading from: {DATA_DIR_PATH}")
        loader = get_dataloader(
            DATA_DIR_PATH, 
            tokenizer, 
            batch_size=BATCH_SIZE, 
            chunk_size=CHUNK_SIZE,
            rank=RANK, 
            world_size=WORLD_SIZE, 
            num_workers=NUM_WORKERS
        )

        # 3. Verify Iteration
        print("-> Starting iteration (Simulating training loop)...")
        print("-" * 60)
        
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids']
            labels = batch.get('labels', 'N/A')
            
            print(f"Batch {i}:")
            print(f"  Input Shape: {input_ids.shape}") # Should be [4, 2048]
            
            # Sanity Check: Decode the first sequence in the batch
            # This proves we aren't just loading zeros or garbage.
            decoded_snippet = tokenizer.decode(input_ids[0][:20], skip_special_tokens=False)
            print(f"  Sample Content (First 20 tokens): {decoded_snippet!r}...")
            
            # Stop after 3 batches so we don't loop forever
            if i >= 2:
                print("-" * 60)
                print("Test Complete: Successfully loaded and tokenized 3 batches.")
                break

    except KeyboardInterrupt:
        print("\n[Stopped by user]")
    except FileNotFoundError:
        print(f"\n[Error] Could not find data directory: {DATA_DIR_PATH}")
        print("Please ensure the path exists and contains .parquet files.")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
