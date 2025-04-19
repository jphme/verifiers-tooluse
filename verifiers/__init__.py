from .envs.environment import Environment
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.config_utils import get_default_grpo_config
from .utils.data_utils import (
    extract_boxed_answer,
    extract_hash_answer,
    preprocess_dataset,
)
from .utils.logging_utils import print_prompt_completions_sample, setup_logging
from .utils.model_utils import get_model, get_model_and_tokenizer, get_tokenizer

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Environment",
    "GRPOEnvTrainer",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]
