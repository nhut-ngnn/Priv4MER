# Federated Dynamic Gating Framework for Semi-Supervised Multimodal Emotion Recognition

> Official code repository for the capstone project 
  <b>"Federated Dynamic Gating Framework for Semi-Supervised Multimodal Emotion Recognition"</b>, Group code: <b>GSP26AI05</b>, Project code: <b>SP26AI27</b>.


<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-21.04.2026-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Repository Structure**](#repository-structure) •
[**Setup**](#setup) •
[**Quick Start**](#quick-start) •
[**LOSO Runners**](#loso-runners) •
[**References**](#references) •
[**Contact**](#contact)

</div>

## Repository Structure

```text
centralized/        # centralized training, MELD pipeline, LOSO runner
federated/          # federated preprocess/train/eval pipelines
feature_extract/    # feature extraction CLI
src/                # model architectures + shared utilities
demo_FL/            # FastAPI + Streamlit demo for global/local FL simulation
metadata/           # generated metadata PKLs
features/           # extracted feature PKLs
checkpoints/        # saved model checkpoints
logs/               # training/evaluation logs
NoiseX-92/          # example noise files used for noisy-test generation
```

## Setup

Use Python 3.10+ (the Dockerfiles use `python:3.10-slim`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended extras used by scripts/UI:

```bash
pip install librosa streamlit
```

## Quick Start

Typical data processing workflow in this repository:

### 1) Preprocess metadata

```bash
# centralized metadata
python3 centralized/preprocess.py \
  --dataset MELD \
  --data_root /path/to/data_root \
  --out_dir metadata/MELD_preprocessed

# federated client splits
python3 federated/preprocess.py \
  --dataset MELD \
  --data_root /path/to/data_root \
  --out_dir metadata/MELD_federated/clients \
  --split_by speaker \
  --num_clients 5 \
  --labeled_ratio 1.0 \
  --seed 42
```

### 2) Extract features

```bash
# centralized features
python3 feature_extract/extract_feature.py \
  --dataset MELD \
  --pkl_dir metadata/MELD_preprocessed \
  --output_dir features/MELD/centralized \
  --wav_base /path/to/data_root

# federated features (per client)
for c in metadata/MELD_federated/clients/client_*; do
  python3 feature_extract/extract_feature.py \
    --dataset MELD \
    --client_dir "$c" \
    --out_dir "features/MELD/clients/$(basename "$c")" \
    --wav_base /path/to/data_root
done
```

### 3) Train model

```bash
# centralized training
python3 centralized/train.py \
  --data_dir features/MELD/centralized \
  --dataset MELD \
  --num_classes 7 \
  --model_name fedalmer \
  --epochs 50

# federated training
python3 -m federated.run_federated \
  --dataset MELD \
  --num_classes 7 \
  --model_name fedalmer \
  --clients_root metadata/MELD_federated/clients \
  --features_root features/MELD/clients \
  --rounds_pretrain 10 \
  --rounds_ssl 0
```

### 4) Run full pipeline (shortcut)

```bash
# centralized all-in-one
python3 centralized/run_meld_pipeline.py --config centralized/configs/meld.yaml

# federated all-in-one
python3 federated/run_meld_pipeline.py --config federated/configs/meld.yaml
```

## LOSO Runners

### Centralized LOSO

```bash
python3 centralized/loso_runner.py --config centralized/configs/iemocap.yaml
```

### Federated LOSO

```bash
python3 federated/loso_runner.py --config federated/configs/iemocap.yaml
```

Both runners use session-based LOSO folds and output per-fold/per-seed summaries.

## Outputs

- `metadata/...`: preprocessed train/val/test PKLs, client partitions, `client_map.json`
- `features/...`: extracted embeddings (`*_features.pkl`, optional manifests)
- `checkpoints/...`: centralized/federated checkpoints
- `logs/...`: run logs, per-round/client metrics, eval summaries
- `results/...`: aggregated centralized metrics CSV


## License

This project is licensed under the MIT License. See `LICENSE`.

## References
[1] Nhut Minh Nguyen, Enhancing multimodal emotion recognition with dynamic fuzzy membership and attention fusion, (Engineering Applications of Artificial Intelligence), 2026. Available https://github.com/aita-lab/FleSER.

[2] Nhut Minh Nguyen, CemoBAM: Advancing Multimodal Emotion Recognition through Heterogeneous Graph Networks and Cross-Modal Attention Mechanisms (APNOMS), 2025. Available https://github.com/nhut-ngnn/CemoBAM.

[3] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[4] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[5] Nhat Truong Pham, SER-Fuse: An Emotion Recognition Application Utilizing Multi-Modal, Multi-Lingual, and Multi-Feature Fusion (SOICT), 2023. Available https://github.com/nhattruongpham/SER-Fuse.


## Contact

- Email: `minhnhut.ngnn@gmail.com`
- GitHub: https://github.com/nhut-ngnn
- ORCID: https://orcid.org/0009-0003-1281-5346
# Priv4MER
# Priv4MER
# Priv4MER
