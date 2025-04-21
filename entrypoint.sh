#!/bin/bash

# Login to Hugging Face
huggingface-cli login --token $HF_TOKEN

# Run the training script with accelerate and deepspeed config
# num processes = 1- GPUs
accelerate launch --config-file configs/zero3.yaml --num-processes 1 verifiers/examples/bfcl_inthinking_agent.py

# Simple entrypoint that executes passed commands
exec "$@"