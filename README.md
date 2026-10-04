# FMCTTA

> Continual test-time adaptation for fault diagnosis with pretrained UniFreq,
> UniFault, and TF-ProFM encoders.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Hugging%20Face-UniFreq-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bojian1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

FMCTTA turns a pretrained fault foundation model into a stateful online fault
diagnosis system. It first learns a classifier from one labeled source domain,
then adapts to an ordered stream of unlabeled target domains without resetting
the model between domains.

## Highlights

- Loads UniFreq from Hugging Face and UniFault/TF-ProFM from PHMFD pretraining
  checkpoints; no new pretraining run is required.
- Implements Source-only, source-reference kNN, TENT, NOTE, EATA, SAR, CoTTA, ViDA, Continual-MAE, RoTTA, TCA, GOLD, and classic TTT for Transformer-based
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

| `foundation_model` | Weight source | Default model-side preprocessing |
|---|---|---|
| `unifreq` | Hugging Face/local `--model_id` | Built-in UniFreq processor |
| `unifault` | PHMFD `--pretrained_checkpoint` | Resample to 1024 and sample min-max |
| `tf_profm` | PHMFD `--pretrained_checkpoint` | Resize/truncate to 2048 and sample z-score |

For the local PHMFD checkout, `--pretrained_checkpoint auto` selects the
seed-42 checkpoint under `/data1/PHM-Vibench/PHMFD/log`. Model profiles live in
`configs/foundation_models/`. Checkpoint loading requires complete backbone
coverage and intentionally discards pretraining-only heads; FMCTTA trains a
new source classifier for the selected dataset.

### UniFreq variants

| CLI value | Hugging Face model | Recommended use |
|---|---|---|
| `bojian1/UniFreq-tiny` | [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | Fast experiments and smoke tests |
| `bojian1/UniFreq-small` | [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | Accuracy/compute balance |
| `bojian1/UniFreq-base` | [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | Highest-capacity encoder |

The exact UniFreq inference architecture is included under
`fmctta/unifreq/`, synchronized from UniFreq commit
`eeb2930a35f5668b80000c7f19875f39714b1c44`. Model weights remain on Hugging
Face and are downloaded on demand. Migrated UniFault/TF-ProFM backbones are in
`fmctta/phmfd_models.py`; their large weight files remain in PHMFD `log`.

## Supported adaptation methods

| Method | Online objective | Default trainable parameters |
|---|---|---|
| `source` | No target update | None |
| `knn` | Cosine kNN over a fixed labeled source-train reference bank | None |
| `tent` | Entropy minimization | LayerNorm affine parameters |
| `note` | Entropy minimization over pseudo-balanced replay (PBRS) | LayerNorm affine parameters |
| `eata` | Reliable/diverse entropy + Fisher regularization | LayerNorm affine parameters |
| `sar` | Reliable entropy with sharpness-aware updates | LayerNorm affine parameters |
| `cotta` | EMA teacher consistency + stochastic restoration | All parameters |
| `vida` | EMA consistency + high/low-rank ViDA with HKA | ViDA branches + all base parameters |
| `continual_mae` | Frequency-bin distribution-aware masking + reconstruction + teacher consistency | All base parameters + MAE auxiliary parameters |
| `rotta` | CSTU memory and timeliness-weighted EMA consistency | LayerNorm affine parameters |
| `tca` | Persistent class topology, consistency, and view alignment | All parameters |
| `gold` | AGOP subspace scaling, teacher self-training, and source-prototype contrast | All parameters + scaling vector and projector |
| `ttt` | Four-way circular-shift prediction before task inference | Foundation backbone + auxiliary head |

All methods support all three backbones except `continual_mae`, whose current
frequency-bin reconstruction interface is specific to UniFreq global-FFT
tokens.

The [NOTE](https://github.com/TaesikGong/NOTE) integration keeps its
pseudo-label balanced reservoir and periodic memory-based entropy updates. The
original image implementation's optional instance-aware BatchNorm is not
ported because UniFreq uses Transformer/LayerNorm blocks; the standard
adaptation scope controls the LayerNorm affine parameters instead.

The [ViDA](https://github.com/Yangsenqiao/vida) integration injects parallel
low- and high-rank linear branches into the foundation backbone. Homeostatic
Knowledge Allotment (HKA) uses augmentation-based teacher uncertainty to favor
the high-rank branch under a large shift and the low-rank branch otherwise.
The branches are zero-initialized after source checkpoint loading, so existing
FMCTTA source checkpoints remain compatible.

UniFreq uses Transformer/LayerNorm blocks, so BatchNorm-only AdaBN is not
applicable. The RoTTA port keeps CSTU and its EMA objective while explicitly
using LayerNorm affine adaptation instead of pretending Robust BN exists.
Classic TTT jointly trains its shift-prediction head on the labeled source
stage and stores that head in a separate safe checkpoint file.

## Project layout

```text
FMCTTA/
├── data/ctta_custom/         # bundled PU/WT MAT files and manifests
├── fmctta/
│   ├── adapters.py           # online CTTA algorithms
│   ├── knn.py                # source-reference kNN baseline
│   ├── note.py               # NOTE pseudo-balanced replay adapter
│   ├── rotta.py              # LayerNorm RoTTA port and CSTU memory
│   ├── tca.py                # topological consistency adaptation
│   ├── ttt.py                # source-trained test-time training task
│   ├── data.py               # source split and ordered target streams
│   ├── model.py              # unified model factory and classifier
│   ├── phmfd_models.py       # UniFault and TF-ProFM backbones
│   ├── runner.py             # complete CTTA protocol
│   ├── source.py             # source training and safe checkpoints
│   ├── vida.py               # high/low-rank ViDA injection
│   └── unifreq/              # inference-only UniFreq implementation
├── tests/
└── main.py
```

The default data directory is `data/ctta_custom`, which contains the bundled
PU and WT datasets. HUST and MCC5 experiments are loaded through their
generated manifests or an explicit `--data_root`. The raw-data preprocessing
instructions are available in
[PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing).

The local [dataset builder](ctta_data_builder/README_zh.md) supports SK, PU, WT,
SDUST-Bearing/Gear, MCC5-Gearbox/Motor, and HUST-Gearbox, with configurable native
window lengths and strides. Current OCTTA templates are included under
`ctta_data_builder/configs/octta/`; generated manifests contain `fmctta_spec`
and load directly through `--dataset_manifest`.

## Installation

```bash
cd /data1/PHMT/FMCTTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Hugging Face token is required for public UniFreq models. For offline runs,
download a model in advance and pass either its local directory to
`--model_id` or use `--local_files_only` with an existing cache.

## Quick start

### 1. Train the source classifier and run a source-only baseline

```bash
python main.py \
  --dataset PU \
  --method source \
  --model_id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --run_name pu-source
```

This downloads UniFreq, trains the downstream classifier on domain `3`, and
evaluates domains `2 → 0 → 1` in order. The source checkpoint is saved at:

```text
outputs/PU/source/pu-source-seed2025/source_checkpoint/
```

Set `--source_backbone_lr_scale 0` to train only the classifier. The default
`0.01` also fine-tunes UniFreq using one hundredth of the classifier learning
rate.

### Branch selection

Use `--feature_branch fused` (the default) for the original time/frequency
fusion. For branch ablations, `--feature_branch time` runs only the time-domain
encoder and `--feature_branch frequency` runs only the frequency-domain
encoder. A single branch produces `embed_dim` features instead of the
concatenated feature vector, so train a new source classifier/checkpoint for
each selection. Checkpoints record the selection and reject a mismatched run.

For example, train and use a frequency-only source model:

```bash
python main.py \
  --dataset PU --domains 3,2,0,1 --method source \
  --feature_branch frequency --run_name pu-frequency-source
```

### Per-class sampling and repeated target domains

`--source_samples_per_class` accepts one count for every source class, or a
comma-separated count for each class. `--target_samples_per_class` works the
same way for targets; one entry applies to every target-stream occurrence, and
semicolon-separated entries assign counts by stream position. Repeated target
domains are allowed. Their samples are allocated without replacement, so the
sum requested for each repeated domain/class cannot exceed the data available.
Set `--allow_repeated_target_samples` to independently draw each occurrence
from the full domain pool instead; samples may then overlap between occurrences.
When loading a source checkpoint, pass the same source sampling selection used
to train it; that selection is recorded in the checkpoint metadata.

```bash
python main.py \
  --dataset PU --method tent \
  --domains 0,1,2,3,1,3,2,3 \
  --source_samples_per_class 60 \
  --target_samples_per_class 20 \
  --run_name repeated-targets
```

The target stream above has seven occurrences. Each occurrence receives 20
samples from every class; domain 1 therefore uses 40 distinct samples per
class, domain 2 uses 40, and domain 3 uses 60. For different allocations by
occurrence, use for example `--target_samples_per_class '20;10;15;20;10;15;20'`.

Complete YAML experiments are organized by dataset. Every dataset directory has
`unifreq/`, `unifault/`, and `tf_profm/` subdirectories for the UniFreq,
UniFault-small, and TF-ProFM matrices. Shared method overlays are
kept in `configs/methods/`; ordered YAML `base` lists compose a dataset/model base
with an algorithm overlay. The UniFreq matrix includes Continual-MAE; the two
PHMFD matrices include Source-only, kNN, TENT, NOTE, EATA, SAR, CoTTA, ViDA,
RoTTA, TCA, GOLD, and TTT. See [`configs/README_zh.md`](configs/README_zh.md)
for the layout, run order, and parameter reference. For example, train the PU
source model before running TENT:

```bash
python main.py --config configs/PU/unifreq/source.yaml
python main.py --config configs/PU/unifreq/tent.yaml

python main.py --config configs/PU/unifault/source.yaml
python main.py --config configs/PU/unifault/tent.yaml

python main.py --config configs/PU/tf_profm/source.yaml
python main.py --config configs/PU/tf_profm/tent.yaml
```

Command-line options override YAML values.

### 2. Run continual test-time adaptation

```bash
python main.py --config configs/PU/unifreq/tent.yaml
```

Replace `tent` with `knn`, `note`, `eata`, `sar`, `cotta`, `vida`, or `continual_mae`.
The kNN baseline extracts source-train UniFreq features once and never consumes
target labels or updates model parameters. For example:

```bash
python main.py \
  --dataset PU --domains 3,2,0,1 --method vida \
  --source_checkpoint outputs/PU/source/pu-source-seed2025/source_checkpoint \
  --vida_low_rank 1 --vida_high_rank 128 \
  --run_name pu-vida
```

Use
`--adaptation_parameter_scope classifier`, `backbone`, `layernorm`, or `all`
for controlled ablations. ViDA branches remain trainable under every ViDA
scope; the scope controls the additional base-model parameters.

For a quick end-to-end check:

```bash
python main.py \
  --dataset PU --domains 3,2 --method tent \
  --source_epochs 1 --source_backbone_lr_scale 0 \
  --max_source_per_class 4 --max_target_per_domain 8 \
  --stream_batch_size 4 --device cpu --run_name smoke
```

## Evaluation protocol

1. Split the first domain class-stratifiably into source-train and
   source-validation subsets.
2. Apply either normalization fitted on source-train or independent per-sample
   normalization, according to `--normalization`.
3. Train or load the source classifier.
4. Process every target sample exactly once in the specified order.
5. Produce the online prediction, perform a label-free update, and optionally
   re-predict the same batch for a post-adaptation diagnostic.
6. Carry the adapted state into the next batch and next domain.
7. Re-evaluate source validation data after each target domain to quantify
   forgetting. This diagnostic never updates the adapter. For kNN, the source
   validation split is excluded from its reference bank.

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

The tests cover target-label-free adapter calls, parameter scopes, all methods,
deterministic stream construction, source-only normalization, and safe
checkpoint validation.

## Acknowledgements

FMCTTA builds on UniFreq and the UniFault/TF-ProFM implementations and weights
in PHMFD, and follows the online protocol organization of the neighboring
OCTTA implementation.

## License

This project is released under the [MIT License](LICENSE).
# FMCTTA

GOLD signal adaptation and protocol differences are documented in [docs/GOLD_ZH.md](docs/GOLD_ZH.md).
