# 1.58 BitNet

This repository provides an implementation of the 1.58 BitNet LLaMA model. The project aims to reduce memory usage through ternary quantisation while keeping the training workflow close to standard LLaMA models.

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
checkpoint.

```bash
python trainingv2.py --dataset path/to/data.jsonl --model_path path/to/model --output_dir ce_out --iters 1000
```

## GRPO Training

`grpo_train.py` implements Grouped Response Policy Optimisation (GRPO).  The
dataset should be JSON or JSONL with one object per record:

```json
{"query": "...", "answer": "..."}
```

During training candidate answers are generated for each query and scored. If
one or more `--reward_model` checkpoints are provided the scores from each model
are combined (optionally weighted with `--reward_weights`) to approximate dense
feedback. Without a reward model the robust F1 reward from `qa_reward` is used.

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
- The second GRPO layer trains on corrected answers whose reward improves over
  the first pass, using the reward difference as the advantage.

Advantages are normalised by their standard deviation. The KL penalty uses the
forward KL estimator $\sum_t p_\theta(t) (\log p_\theta(t) - \log p_{\text{ref}}(t))$
averaged over tokens.

```bash
python grpo_train.py --dataset qa.jsonl --model_path path/to/model \
    --output_dir grpo_out --steps 1000 --csv_log metrics.csv
```

## Hardware and Example Commands

Training large models is memory intensive. The scripts are tested on Apple MPS with CPU fallbacks but work on any hardware that PyTorch supports. Expect to require tens of gigabytes of RAM for multi‑billion parameter models.

Example command for a small CE run:

```bash
python trainingv2.py --dataset data/train.jsonl --model_path llama_750m \
    --output_dir ce_model --iters 10000 --batch_size 8 \
    --resume ce_model/ce.ckpt --save_interval 1000
```

Example command for GRPO with two reward models:

```bash
python grpo_train.py --dataset qa.jsonl --model_path llama_750m \
    --reward_model rm1.ckpt rm2.ckpt --reward_weights 0.7 0.3 \
    --output_dir grpo_model
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
    --reward_model rm1.ckpt rm2.ckpt --reward_weights 0.7 0.3 \
    --output_dir grpo_model --two_layer --guiding_prompt prompts.txt
```
Here `prompts.txt` contains one prompt per line (or a JSON list) used for the
second pass.

## Evaluation

`evaluation.py` compares a CE fine‑tuned model to a GRPO model using the same QA
F1 reward as training.  The dataset format matches the GRPO training set
(`{ "query": ..., "answer": ... }`). When a record includes a `reasoning` field the
script also reports token level F1 for the text inside `<think>` tags and a
`step_correctness` metric measuring how many reasoning steps match the reference.

```bash
python evaluation.py --dataset qa.jsonl --ce_model ce_model --grpo_model grpo_model
```

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
