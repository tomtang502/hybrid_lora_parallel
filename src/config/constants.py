import numpy as np, torch
DATA_DIR_PATH = "/ocean/projects/cis250196p/ltang2/nemotron_data_sample"
DATA_PATH = "/home/haozhan-tang/ml/db/nemotron_cc_v2/Diverse-QA/part_000000.parquet"
MODEL_PATH = "jet-aiJet-Nemotron-2B"

DATASET_BASE_CHUNK_SIZE = 1024

DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}

ORANGE = "\033[38;5;208m" 
BLUE = "\033[38;5;27m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;51m"
DEBUG_RED = "\033[38;5;88m"
RESET = "\033[0m"

NUM_DECODER_LAYERS = 28
MHA_IDX = (14, 19, 20, 21)

MHA_MODULE_NAMES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MHA_LORA_MODULE_NAMES = ('q_proj.lora_A', 'k_proj.lora_A', 'v_proj.lora_A', 'o_proj.lora_A',
                         'q_proj.lora_B', 'k_proj.lora_B', 'v_proj.lora_B', 'o_proj.lora_B')

GDN_MODULE_NAMES = ('q_proj', 'k_proj', 'v_proj', 'b_proj', 'a_proj', 'kernel_generator.w1', 'kernel_generator.w2', 'g_proj', 'o_proj')
GDN_LORA_MODULE_NAMES = ('q_proj.lora_A', 'k_proj.lora_A', 'v_proj.lora_A', 'b_proj.lora_A', 'a_proj.lora_A', 'kernel_generator.w1.lora_A', 'kernel_generator.w2.lora_A', 'g_proj.lora_A', 'o_proj.lora_A',
                         'q_proj.lora_B', 'k_proj.lora_B', 'v_proj.lora_B', 'b_proj.lora_B', 'a_proj.lora_B', 'kernel_generator.w1.lora_B', 'kernel_generator.w2.lora_B', 'g_proj.lora_B', 'o_proj.lora_B')

FFN_MODULE_NAMES = ('gate_proj', 'up_proj', 'down_proj')
FFN_LORA_MODULE_NAMES = ('gate_proj.lora_A', 'up_proj.lora_A', 'down_proj.lora_A',
                         'gate_proj.lora_B', 'up_proj.lora_B', 'down_proj.lora_B')

MODUEL_REGEX_FORMAT = r'.*\b{layer_idx}\b.*{module_name_suffix}\Z'