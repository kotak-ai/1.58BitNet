# 1.58 BitNet

This repository provides an implementation of the 1.58 BitNet LLaMA model. The project aims to reduce memory usage through ternary quantisation while keeping the training workflow close to standard LLaMA models.

### Packed Hadamard Linear Layers

The code includes an `HBitLinear` module implementing a Hadamard transform with
packed 1.58‑bit weights. `LlamaModel` uses this layer by default.  To fall back
to the older quantised `BitLinear` simply pass `linear_cls=BitLinear` when
instantiating or loading a model.

`gemm_lowbit` now supports multiplying int8 activations with these packed
weights directly via CUDA and MPS kernels. Pass the packed tensor and its
original shape to perform the multiplication without unpacking.

## Requirements and Setup

The code depends on a handful of well‑known libraries.  A
`requirements.txt` file is provided with the minimum tested versions:

```text
torch>=2.0
transformers>=4.31
datasets>=2.14
nltk>=3.8
numpy>=1.23
psutil>=5.9
safetensors>=0.3.1
tqdm>=4.66
```

Install them with:

```bash
pip install -r requirements.txt
```

The training scripts are tested on Apple MPS hardware but will run on any
PyTorch device (CPU or CUDA).  Multi‑billion parameter models may require tens of
gigabytes of RAM.

## Creating a Model

Use `new-model-architecture-creation.py` to generate a blank ternary model.
Specify the desired parameter count on the command line. The model is written
to `--output_dir` if provided, otherwise `llama_<params>_ternary_quantized_optimized`.
Passing `--e` enables an experimental quantisation mode.

```bash
python new-model-architecture-creation.py --params 750M --output_dir llama_750M
```

## Quantized Model Format

`LlamaModel.save_pretrained` writes quantised weights to `model.safetensors`.
Each tensor from the model `state_dict` is saved with a `model.` prefix and a
matching `<name>.shape` entry holding the original dimensions.  The accompanying
`model.safetensors.index.json` lists the tensor names and includes a
`metadata.total_size` field giving the total byte size.  The byte count is
calculated using `numel * 1.58 / 8` for each tensor, with a small overhead for
the stored shape. Config and tokenizer files are stored alongside these.

## Cross‑Entropy Training

`trainingv2.py` performs standard CE training on tokenised text datasets.  The
file may be plain text or a JSON/JSONL file containing a `text` field.  Each
line or record is treated as one training example. Pass `--save_interval` to
periodically checkpoint training and `--resume` to continue from a saved
checkpoint. Use `--grad_checkpoint` to reduce memory usage at the cost of extra
compute.

```bash
python trainingv2.py --dataset path/to/data.jsonl --model_path path/to/model --output_dir ce_out --iters 1000
```

Pass `--run_eval` along with `--eval_dataset` to automatically evaluate the
new model against the original checkpoint using `evaluation.py`. Metrics are
printed to the console and can be appended to a CSV file via `--eval_csv`.

## GRPO Training

`grpo_train.py` implements Grouped Response Policy Optimisation (GRPO).  The
dataset should be JSON or JSONL with one object per record:

```json
{"query": "...", "answer": "..."}
```

During training candidate answers are generated for each query and scored.  The
robust F1 reward from `qa_reward` can be mixed with one or more
`--reward_model` checkpoints.  Model scores are combined (optionally weighted
with `--reward_weights`) then interpolated with the rule-based reward using
`--rule_weight`.  When no reward model is supplied, only the F1 reward is used.

Optional features:

- `--config FILE` &ndash; JSON file with argument defaults (e.g. `guiding_prompt`).
- `--two_layer` &ndash; enable the two stage trainer with self-correction.
- `--augmentation_size` &ndash; number of corrected answers sampled for each
  initial response when self-correction is enabled (the H parameter from the
  [GRPO paper](https://arxiv.org/pdf/2506.04746)).
- `--csv_log LOG.csv` &ndash; append metrics to a CSV file. When `--two_layer` is
  active the log also includes up to three `corrected_n` columns with corrected
  answers for inspection.
- `--resume CKPT` &ndash; resume training from a checkpoint created with
  `save_checkpoint`.
- `--grad_checkpoint` &ndash; wrap policy forward passes with `custom_checkpoint`
  to save memory.
- `--rule_weight` &ndash; weight assigned to the rule-based F1 reward when
  combining with external reward models.
- `--run_eval` &ndash; evaluate the final model using `evaluation.py` after
  training. Requires `--eval_dataset` and optionally `--eval_csv`.
- Reward models now support **dense** scoring. When available the training
  loop consumes a sequence of per-token rewards instead of a single scalar.
- `--guiding_probabilities` &ndash; probabilities corresponding to each entry in
  `--guiding_prompt` for weighted random selection.
- `--guiding_schedule` &ndash; sequence of prompt indices determining which
  guiding prompt to use for successive batches. Overrides
  `--guiding_probabilities` when provided.

Example `config.json` providing defaults:

```json
{
  "lr": 0.0005,
  "group_size": 4,
  "guiding_prompt": "Review and correct the answer:"
}
```
`guiding_prompt` can be either the prompt text itself or a path to a file
containing one or more prompts. When multiple prompts are provided one is chosen
at random for each correction step.

### GRPO Hyperparameters

The training scripts implement the equations from the GRPO paper. Important
parameters which mirror the publication are:

- `clip_eps` – the PPO style clipping range controlling the update magnitude.
- `beta` – weight applied to the KL penalty against the reference policy.
- `verifier` – optional function used in the second layer to decide if a
  correction is considered an improvement.
- `improvement_threshold` – reward margin required for a correction to count as
  an improvement when the final answer is not exactly correct.
- `verifier_mode` – choose between the default reward based verification or the
  paper's dynamic verification which only accepts corrections with the correct
  final answer.
- The second GRPO layer trains on corrected answers whose reward improves over
  the first pass, using the reward difference as the advantage.

Advantages are normalised by their standard deviation. The KL penalty uses the
forward KL estimator $\sum_t p_\theta(t) (\log p_\theta(t) - \log p_{\text{ref}}(t))$
averaged over tokens.

```bash
python grpo_train.py --config scripts/paper_config.json --dataset qa.jsonl \
    --model_path path/to/model --output_dir grpo_out --csv_log metrics.csv
```

Include `--run_eval` with an evaluation dataset to score the resulting model
immediately after training. Use `--eval_csv` to append these metrics to a log
file.

## Hardware and Example Commands

Training large models is memory intensive. The scripts are tested on Apple MPS with CPU fallbacks but work on any hardware that PyTorch supports. Expect to require tens of gigabytes of RAM for multi‑billion parameter models.

The decoder layers use a small `custom_checkpoint` wrapper to recompute the
attention and MLP blocks during backpropagation. This reduces peak memory usage
so the code can train larger models on commodity GPUs at the cost of some extra
compute.
Passing `--grad_checkpoint` to the training scripts additionally wraps the full
forward pass with this checkpointing mechanism for further memory savings.

Example command for a small CE run:

```bash
python trainingv2.py --dataset data/train.jsonl --model_path llama_750m \
    --output_dir ce_model --iters 10000 --batch_size 8 --grad_checkpoint \
    --resume ce_model/ce.ckpt --save_interval 1000
```

Example command for GRPO with two reward models:

```bash
python grpo_train.py --dataset qa.jsonl --model_path llama_750m \
    --reward_model rm1.ckpt rm2.ckpt --reward_weights 0.7 0.3 --rule_weight 0.5 \
    --output_dir grpo_model --grad_checkpoint
```

## Two-Layer Self-Correction

Passing `--two_layer` enables a second GRPO pass that attempts to refine the first answer.
The correction prompt follows the template:

```
<user>{query}<think>{first_answer}</think>{guiding_prompt}</user><assistant>
```

`--guiding_prompt` may be a path to a text or JSON file containing multiple
prompts. By default a prompt is chosen at random for each correction. Pass
`--guiding_probabilities` to weight this random selection or provide
`--guiding_schedule` with a list of indices to follow a fixed order.

```bash
python grpo_train.py --dataset qa.jsonl --model_path llama_750m \
    --reward_model rm1.ckpt rm2.ckpt --reward_weights 0.7 0.3 --rule_weight 0.5 \
    --output_dir grpo_model --two_layer --guiding_prompt prompts.txt
```
Here `prompts.txt` contains one prompt per line (or a JSON list) used for the
second pass.

## MGRPO Benchmark Scripts

The `scripts` directory contains helpers for running multi‑layer GRPO on the
official reasoning benchmarks. Each wrapper downloads the dataset and launches
`grpo_train.py` with the paper hyperparameters from `scripts/paper_config.json`.

```bash
# Train on the MATH benchmark
scripts/mgrpo_math.sh

# Train on GSM8K
scripts/mgrpo_gsm8k.sh

# Train on Minerva Math
scripts/mgrpo_minerva.sh

# Train on OlympiadBench
scripts/mgrpo_olympiadbench.sh
```
Edit the reward model path inside each script to point at your checkpoint.

## Reproducing the Paper Experiments

The full pipeline used in the paper starts from an empty quantised model and
applies CE training followed by GRPO or MGRPO.  The steps below mirror this
process.

1. **Create a blank model**

   ```bash
   python new-model-architecture-creation.py --params 750M --output_dir llama_750m
   ```

2. **Cross‑entropy fine tuning**

   Train the baseline model on your text corpus.  The `scripts/ce_training.sh`
   helper contains the command used in the paper.  Edit `DATA_PATH` and
   `MODEL_PATH` inside the script then run:

   ```bash
   scripts/ce_training.sh
   ```

3. **Single‑layer GRPO**

   Use `scripts/grpo_training.sh` to reproduce the GRPO baseline.  This invokes
   `grpo_train.py` with the hyperparameters from
   `scripts/paper_config.json`.

4. **Multi‑Layer GRPO**

   Launch one of the benchmark wrappers, for example:

   ```bash
   scripts/mgrpo_math.sh
   ```

   These wrappers download the dataset, pass `--two_layer` to `grpo_train.py`
   and write the final model to the output directory specified in the script.

5. **Evaluation**

   After training both CE and GRPO models run:

   ```bash
   python evaluation.py --dataset math --ce_model ce_model --grpo_model mgrpo_math --two_layer
   ```

   The script prints `accuracy_t1`, `accuracy_t1_prime`, `accuracy_t2`,
   `delta_t1_t2`, `delta_t1p_t2`, `delta_i2c` and `delta_c2i`.  On QA datasets it
   instead reports F1 scores.

Expected results should match Table 2 from the paper with final accuracies of
approximately **90.4%** on MATH, **95.6%** on GSM8K, **39.3%** on Minerva Math
and **50.4%** on OlympiadBench.

## Evaluation

`evaluation.py` compares a CE fine‑tuned model to a GRPO model using the same QA
F1 reward as training.  The dataset format matches the GRPO training set
(`{ "query": ..., "answer": ... }`). When a record includes a `reasoning` field the
script also reports token level F1 for the text inside `<think>` tags and a
`step_correctness` metric measuring how many reasoning steps match the reference.
When using the two layer mode additional statistics are printed, including
`accuracy_t1_prime` (accuracy after the second pass prior to RL updates) and
`delta_t1p_t2` showing the change from this baseline to the final accuracy. A
`delta_t1_t2` metric is also reported measuring the accuracy improvement from
the first to second pass.

```bash
python evaluation.py --dataset qa.jsonl --ce_model ce_model --grpo_model grpo_model
```

When both models are evaluated on a reasoning dataset the script prints a table
mirroring Table 2 in the paper summarising `accuracy_t1`, `accuracy_t1_prime`,
`accuracy_t2`, `delta_t1_t2`, `delta_t1p_t2`, `delta_i2c` and `delta_c2i` for
each model.

Passing `--two_layer` runs a second correction pass before scoring each answer.
The second pass uses the same template as training with `<think>` tags.  The
guiding prompt text defaults to "Review and correct the answer:" but may be
overridden or loaded from a file.  Customize the prompt with `--guiding_prompt`
and use
`--second_max_length` controls the number of tokens generated for each
correction. Use `--augmentation_size` to sample multiple corrections per
response (the H parameter in the [GRPO paper](https://arxiv.org/pdf/2506.04746)).

Helper loader functions such as `load_math_dataset` rely on the
`datasets` library. They now accept an optional `path` argument for
loading a local copy instead of downloading from the Hugging Face hub.
If a dataset cannot be loaded a readable `RuntimeError` is raised.
### Intrinsic Self-Correction Baseline

Use `intrinsic_baseline.py` to measure a model's ability to correct its own answers without any RL updates. The script performs a second pass with a guiding prompt and reports the same metrics as `evaluation.py`.

```bash
python intrinsic_baseline.py --dataset qa.jsonl --model ce_model
```

For reasoning datasets pass `--task reasoning`:

```bash
python intrinsic_baseline.py --dataset math --model ce_model --task reasoning
```


## Inference

`inference.py` generates text from a saved model. Provide one or more prompts on
the command line:

```bash
python inference.py --model_path path/to/model --prompt "Hello" --prompt "World"
```

The script can also be imported and called programmatically:

```python
from inference import run
run("path/to/model", ["Hello"], max_length=20)
```

Models saved with `LlamaModel.save_pretrained` contain quantised weights.
The inference script automatically loads these packed tensors without
additional steps.

## Transformers Compatibility

`data_loading_compatibility.py` demonstrates loading a checkpoint with
`transformers` and generating text:

```bash
python data_loading_compatibility.py --model_path path/to/model --text "Example"
```

The `run` function can be used directly from Python for integration tests.


## Reward Model Examples

Two reference implementations are provided for scoring generated answers:

- `simple_reward_model.py` – a minimal linear classifier trained on a small
  labelled dataset.
- `reward_model.py` – a more expressive Transformer based scorer supporting
  contrastive training and saving/loading checkpoints.

Run the demo for the simple model with:

```bash
python simple_reward_model.py --epochs 100 --lr 0.1
```

The demo prints higher scores for correct answers and can be extended to create
custom reward models for GRPO training.

`reward_train.py` trains the larger Transformer-based `RewardModel`. Provide a
labelled dataset with `--dataset` or a file of positive/negative pairs with
`--pairs`:

```bash
python reward_train.py --pairs pairs.jsonl --tokenizer path/to/tokenizer
```

## Energy RL Demo

The repository also includes a small reinforcement learning environment to
experiment with energy‑aware scheduling. The `energy_rl_train.py` script trains a
tabular Q‑learning agent and reports the average reward before and after
training. Environment parameters can be customised on the command line:

```bash
python energy_rl_train.py --episodes 100 --max_steps 20 --harvest_rate 3
```

This prints the initial and final rewards so you can verify the agent learns a
better policy.

## Running the Tests

Unit tests cover the data utilities, GRPO trainer, reward models and
evaluation code. Install the requirements first and then run:

```bash
pip install -r requirements.txt
pytest
```

All tests should pass; one test is skipped when WordNet data is unavailable.
Recent tests also verify that `MultiLayerGRPOTrainer` handles `--augmentation_size`
values greater than one by generating multiple corrections and updating the model.
Successful second-pass corrections are kept in a buffer so that future calls to
`MultiLayerGRPOTrainer.train_batch` can reinforce them alongside newly generated
samples.
