from .math import *
from .bbq import *
from .codefeedback import *

ft_dataset_builder_map = {
    'math': HendrycksMathDatasetBuilder,
    'bbq': BBQDatasetBuilder,
    'code': CodeFeedbackDatasetBuilder,
}

# ft_dataset_max_seq_len = {
#     'math': 1024,
#     'bbq': -1, # TODO measure this
# }