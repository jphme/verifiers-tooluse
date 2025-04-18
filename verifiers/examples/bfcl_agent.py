import verifiers as vf
import os 

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

TEST_RUN = False
PRINT_SAMPLE_COMPLETIONS = True
NUM_GPUS = 2
PER_DEVICE_BATCH_SIZE = 1
# Rollouts per prompt
NUM_GENERATIONS = 1
# (NUM_GPUS - 1) * PER_DEVICE_BATCH_SIZE / NUM_GENERATIONS = NUM PROMPT PER DEVICE
EVAL_DATASET_SIZE = 100
MAX_STEPS_PER_TURN = 10
CURRICULUM_LEARNING = True
MAX_TURNS_ONLY = True
MAX_NUM_TURNS = 1 # 4 turns is almost max
DEBUG_GENERATE = False
DEBUG_REWARDS = False
EVAL_STEPS = 50
SAVE_STEPS = 100
NUM_EPOCHS = 100
USE_DR_GRPO = False
USE_LATEST_TRL = False
EVAL_ON_START = False
TEST_HYPOTHESIS = False
if TEST_HYPOTHESIS:
    BASELINE_RUN = False
BETA = 0.001
if BETA == 0:
    UPDATE_REF_MODEL = False
else:
    UPDATE_REF_MODEL = True
MAX_GRAD_NORM = 0.2
# steps per global batch (1 on-policy, N-1 off-policy), mu in DeepSeekMath paper
NUM_ITERATIONS = 2
GRADIENT_ACCUMULATION_STEPS = 32
APPLY_OVERLONG_FILTERING = True
MAX_COMPLETION_LENGTH = 2048
PUSH_TO_VIEWER = False
if PUSH_TO_VIEWER:
    os.environ["CURATOR_VIEWER"] = "1"
else:
    os.environ["CURATOR_VIEWER"] = "0"

# Initialize tool environment for GSM8K
vf_env = vf.BfclEnv(
    dataset="bfcl",
    tools=[],
    max_num_turns=MAX_NUM_TURNS,
    max_steps_per_turn=MAX_STEPS_PER_TURN,
    curriculum_learning=CURRICULUM_LEARNING,
    use_latest_trl=USE_LATEST_TRL,
)

train_dataset = vf_env.get_dataset(max_num_turns=MAX_NUM_TURNS)
eval_dataset = vf_env.get_eval_dataset(max_num_turns=MAX_NUM_TURNS, max_turn_only=MAX_TURNS_ONLY)
print(train_dataset)
print(eval_dataset)

rubric = vf_env.get_rubric()

run_name = "bfcl-" + model_name.split("/")[-1].lower() + f"-{MAX_NUM_TURNS}-turns"
if USE_LATEST_TRL:
    run_name += "-latest-trl"
if USE_DR_GRPO:
    run_name += "-dr-grpo"
if UPDATE_REF_MODEL:
    run_name += "-update-ref-model"

run_name += "-no-format-score"
run_name += "-new-prompt"
if TEST_HYPOTHESIS:
    run_name += "-test-hypothesis"
    if BASELINE_RUN:
        run_name += "-baseline"
    else:
        # run_name += "-clip-advantage-negative-only"
        run_name += "-clip-advantage-positive-only"
if APPLY_OVERLONG_FILTERING:
    run_name += "-apply-overlong-filtering"
if TEST_RUN:
    run_name += "-test-run"
run_name += f"-beta-{BETA}"

training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=NUM_GPUS
)
training_args.num_train_epochs = NUM_EPOCHS
training_args.num_generations = NUM_GENERATIONS
training_args.max_completion_length = MAX_COMPLETION_LENGTH
if DEBUG_GENERATE or DEBUG_REWARDS:
    training_args.report_to = "none"
training_args.per_device_train_batch_size = PER_DEVICE_BATCH_SIZE
training_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
training_args.num_iterations = NUM_ITERATIONS
training_args.beta = BETA
training_args.max_grad_norm = MAX_GRAD_NORM
if TEST_HYPOTHESIS:
    training_args.eval_strategy = "steps"
    training_args.eval_on_start = EVAL_ON_START
    training_args.eval_steps = EVAL_STEPS
else:
    training_args.eval_strategy = "steps"
    training_args.eval_on_start = EVAL_ON_START
    training_args.eval_steps = EVAL_STEPS
training_args.save_strategy = "steps"
training_args.save_steps = SAVE_STEPS
training_args.per_device_eval_batch_size = PER_DEVICE_BATCH_SIZE
training_args.eval_accumulation_steps = 1
training_args.data_seed = 42
if UPDATE_REF_MODEL:
    training_args.sync_ref_model = True
    training_args.ref_model_mixup_alpha = 1.0
    training_args.ref_model_sync_steps = SAVE_STEPS
if TEST_RUN:
    training_args.report_to = "none"
else:
    training_args.report_to = "wandb"

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    debug_generate=DEBUG_GENERATE,
    debug_rewards=DEBUG_REWARDS,
    model_name=model_name,
    run_name=run_name,
    use_dr_grpo=USE_DR_GRPO,
    test_hypothesis_clip_advantage=(TEST_HYPOTHESIS and not BASELINE_RUN),
    apply_overlong_filtering=APPLY_OVERLONG_FILTERING,
    print_sample_completions=PRINT_SAMPLE_COMPLETIONS,
)

trainer.train() 