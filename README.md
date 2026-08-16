# FMCTTA

> Continual test-time adaptation for fault diagnosis with pretrained
> [UniFreq](https://github.com/Bojian-Chen/UniFreq) encoders.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Hugging%20Face-UniFreq-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bojian1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

FMCTTA turns a pretrained UniFreq encoder into a stateful online fault
diagnosis system. It first learns a classifier from one labeled source domain,
then adapts to an ordered stream of unlabeled target domains without resetting
the model between domains.

## Highlights

- Loads `UniFreq-tiny`, `UniFreq-small`, or `UniFreq-base` directly from
  Hugging Face; no pretraining pipeline is included.
- Implements Source-only, TENT, EATA, SAR, CoTTA, and ViDA for Transformer-based
  vibration encoders.
- Keeps target labels outside every adapter. Labels are read only by the
  evaluation meters after prediction.
- Uses a strict single-pass stream and preserves adaptation state across
  domain boundaries.
- Saves source weights as `safetensors` and records online, post-adaptation,
  per-domain, per-sample, and source-retention results.
- Reuses the processed datasets beside this project by default, so large MAT
  files are not duplicated.

## Models

| CLI value | Hugging Face model | Recommended use |
|---|---|---|
| `bojian1/UniFreq-tiny` | [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | Fast experiments and smoke tests |
| `bojian1/UniFreq-small` | [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | Accuracy/compute balance |
| `bojian1/UniFreq-base` | [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | Highest-capacity encoder |

The exact UniFreq inference architecture is included under
`fmctta/unifreq/`, synchronized from UniFreq commit
`eeb2930a35f5668b80000c7f19875f39714b1c44`. Model weights remain on Hugging
Face and are downloaded on demand.

## Supported adaptation methods

| Method | Online objective | Default trainable parameters |
|---|---|---|
| `source` | No target update | None |
| `tent` | Entropy minimization | LayerNorm affine parameters |
| `eata` | Reliable/diverse entropy + Fisher regularization | LayerNorm affine parameters |
| `sar` | Reliable entropy with sharpness-aware updates | LayerNorm affine parameters |
| `cotta` | EMA teacher consistency + stochastic restoration | All parameters |
| `vida` | EMA consistency + high/low-rank ViDA with HKA | ViDA branches + all base parameters |

The [ViDA](https://github.com/Yangsenqiao/vida) integration injects parallel
low- and high-rank linear branches into the UniFreq backbone. Homeostatic
Knowledge Allotment (HKA) uses augmentation-based teacher uncertainty to favor
the high-rank branch under a large shift and the low-rank branch otherwise.
The branches are zero-initialized after source checkpoint loading, so existing
FMCTTA source checkpoints remain compatible.

UniFreq uses Transformer/LayerNorm blocks, so BatchNorm-only AdaBN is not
applicable. Classic TTT is intentionally omitted because it requires an
auxiliary self-supervised head trained together with the source model; the
published UniFreq checkpoints do not contain that auxiliary task.

## Project layout

```text
PHMT/
├── OCTTA/
│   └── data/                 # processed MAT datasets
└── FMCTTA/
    ├── fmctta/
    │   ├── adapters.py       # online CTTA algorithms
    │   ├── data.py           # source split and ordered target streams
    │   ├── model.py          # UniFreq + downstream classifier
    │   ├── runner.py         # complete CTTA protocol
    │   ├── source.py         # source training and safe checkpoints
    │   ├── vida.py           # high/low-rank ViDA injection
    │   └── unifreq/          # inference-only UniFreq implementation
    ├── tests/
    └── main.py
```

The default data directory is `../OCTTA/data`. The preprocessing and dataset
description are available in
[PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing).

## Installation

```bash
cd /data1/PHMT/FMCTTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Hugging Face token is required for public UniFreq models. For offline runs,
download a model in advance and pass either its local directory to
`--model-id` or use `--local-files-only` with an existing cache.

## Quick start

### 1. Train the source classifier and run a source-only baseline

```bash
python main.py \
  --dataset PU_Real \
  --method source \
  --model-id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --run-name pu-source
```

This downloads UniFreq, trains the downstream classifier on domain `3`, and
evaluates domains `2 → 0 → 1` in order. The source checkpoint is saved at:

```text
outputs/PU_Real/source/pu-source-seed2025/source_checkpoint/
```

Set `--source-backbone-lr-scale 0` to train only the classifier. The default
`0.01` also fine-tunes UniFreq using one hundredth of the classifier learning
rate.

### Branch selection

Use `--feature-branch fused` (the default) for the original time/frequency
fusion. For branch ablations, `--feature-branch time` runs only the time-domain
encoder and `--feature-branch frequency` runs only the frequency-domain
encoder. A single branch produces `embed_dim` features instead of the
concatenated feature vector, so train a new source classifier/checkpoint for
each selection. Checkpoints record the selection and reject a mismatched run.

For example, train and use a frequency-only source model:

```bash
python main.py \
  --dataset PU_Real --domains 3,2,0,1 --method source \
  --feature-branch frequency --run-name pu-frequency-source
```

### Per-class sampling and repeated target domains

`--source-samples-per-class` accepts one count for every source class, or a
comma-separated count for each class. `--target-samples-per-class` works the
same way for targets; one entry applies to every target-stream occurrence, and
semicolon-separated entries assign counts by stream position. Repeated target
domains are allowed. Their samples are allocated without replacement, so the
sum requested for each repeated domain/class cannot exceed the data available.
When loading a source checkpoint, pass the same source sampling selection used
to train it; that selection is recorded in the checkpoint metadata.

```bash
python main.py \
  --dataset PU_Real --method tent \
  --domains 0,1,2,3,1,3,2,3 \
  --source-samples-per-class 60 \
  --target-samples-per-class 20 \
  --run-name repeated-targets
```

The target stream above has seven occurrences. Each occurrence receives 20
samples from every class; domain 1 therefore uses 40 distinct samples per
class, domain 2 uses 40, and domain 3 uses 60. For different allocations by
occurrence, use for example `--target-samples-per-class '20;10;15;20;10;15;20'`.

The same setup is available as an executable YAML configuration at
[`configs/experiments/tent_frequency_repeated.yaml`](configs/experiments/tent_frequency_repeated.yaml).
It inherits the shared values in [`configs/base.yaml`](configs/base.yaml), so
the experiment file contains only parameters specific to this run:

```bash
python main.py --config configs/experiments/tent_frequency_repeated.yaml
```

Command-line options override YAML values.

### 2. Run continual test-time adaptation

```bash
python main.py \
  --dataset PU_Real \
  --method tent \
  --model-id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --source-checkpoint outputs/PU_Real/source/pu-source-seed2025/source_checkpoint \
  --run-name pu-tent
```

Replace `tent` with `eata`, `sar`, `cotta`, or `vida`. For example:

```bash
python main.py \
  --dataset PU_Real --domains 3,2,0,1 --method vida \
  --source-checkpoint outputs/PU_Real/source/pu-source-seed2025/source_checkpoint \
  --vida-low-rank 1 --vida-high-rank 128 \
  --run-name pu-vida
```

Use
`--adaptation-parameter-scope classifier`, `backbone`, `layernorm`, or `all`
for controlled ablations. ViDA branches remain trainable under every ViDA
scope; the scope controls the additional base-model parameters.

For a quick end-to-end check:

```bash
python main.py \
  --dataset PU_Real --domains 3,2 --method tent \
  --source-epochs 1 --source-backbone-lr-scale 0 \
  --max-source-per-class 4 --max-target-per-domain 8 \
  --stream-batch-size 4 --device cpu --run-name smoke
```

## Evaluation protocol

1. Split the first domain class-stratifiably into source-train and
   source-validation subsets.
2. Fit optional normalization statistics on source-train only.
3. Train or load the source classifier.
4. Process every target sample exactly once in the specified order.
5. Produce the online prediction, perform a label-free update, and optionally
   re-predict the same batch for a post-adaptation diagnostic.
6. Carry the adapted state into the next batch and next domain.
7. Re-evaluate source validation data after each target domain to quantify
   forgetting. This diagnostic never updates the adapter.

`online` is the primary metric because it measures the prediction available
before learning from the current batch. `post_adapt` is reported separately
and must not be compared as if it were the same prediction stage.

## Outputs

Each run writes:

| File | Contents |
|---|---|
| `config.json` | Fully resolved command-line configuration |
| `source_history.json` | Source training curve |
| `source_checkpoint/` | Safe source model weights and metadata |
| `domains.csv` | Online/post-adaptation metrics and source retention by domain |
| `predictions.csv` | Sample IDs, targets, predictions, and confidence |
| `summary.json` | Protocol declaration and aggregate results |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover target-label-free adapter calls, parameter scopes, all six
methods, deterministic stream construction, source-only normalization, and
safe checkpoint validation.

## Acknowledgements

FMCTTA builds on [UniFreq](https://github.com/Bojian-Chen/UniFreq) and follows
the online protocol organization of the neighboring OCTTA implementation.

## License

This project is released under the [MIT License](LICENSE).
# FMCTTA
