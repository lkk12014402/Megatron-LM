import habana_frameworks.torch.core as htcore
import torch
device = torch.device("hpu")

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.utils import (
    check_habana_frameworks_version,
    check_optimum_habana_min_version,
    get_habana_frameworks_version,
    set_seed,
)

import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers

# Set up argument parsing
parser = argparse.ArgumentParser(description="Script for text generation with a specific model and prompt.")
parser.add_argument('--prompt', type=str, required=True, help="Prompt text to use for text generation")
parser.add_argument('--model-path', type=str, required=True, help="Path to the Huggingface model checkpoint")

# Parse command-line arguments
args = parser.parse_args()

model_path = args.model_path
prompt = args.prompt


pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="hpu",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "What is the deep learning?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=32,
)
print(outputs[0]["generated_text"][-1])

