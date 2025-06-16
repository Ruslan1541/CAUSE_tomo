# CAUSE\_tomo — Causal Unsupervised Segmentation for tomographic images

This code is the further development of [**Causal‑Unsupervised‑Segmentation**](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation) with an extended preprocessing pipeline, Hydra‑based configuration, and one‑command training.

## ✨ What's new in this fork?

- **Adaptive standardization** – per‑image mean/std computed automatically.
- **Patchifying & stitching** – work with gigapixel images without memory errors.
- **Single YAML configuration** – set everything in `config/config.yaml`; no shell edits.
- **Hydra + TensorBoard logging** – experiment tracking out of the box.
- **Simplified CLI** – `python train.py` for the full pipeline.
- **Optional supervision** – evaluate your own labels alongside unsupervised outputs.

## Installation

### 1. Clone

```bash
git clone https://github.com/Ruslan1541/CAUSE_tomo.git
cd CAUSE_tomo
```

### 2. Create a Conda environment

```bash
conda create -n cause_tomo python=3.10
conda activate cause_tomo
```

### 3. Install the package

```bash
pip install -r requirements.txt
```
## Directory overview

```
CAUSE_tomo
├── config/                  # <‑‑ all hyper‑parameters live here
│   ├── dataset/*.yaml      # config with parameters for specific dataset
│   ├── model/*.yaml        # config with parameters for specific ViT backbone
│   └── config.yaml         # main config file
├── loader/                 # data loading & preprocessing
├── models/                 # ViT backbones (DINO, DINOv2, MAE, iBOT, MSN)
├── modules/                # segmentation heads & utilities
├── outputs/                # created automatically – logs, checkpoints, hydra configs (will be created after run)
├── train.py                # full pipeline
├── train_mediator.py       # generate clustrbook
├── train_front_door.py     # frontdoor adjustment through constrastive learning
├── fine_tuning.py          # repeat clustering on refined concepts
└── inference.py            # inference on images
```

## Quick start

1. **Download pre‑trained backbones from** [**Causal‑Unsupervised‑Segmentation**](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation)\
   Place the checkpoint files under `ckpt/` (path configurable in `config/model/*.yaml`).

2. **Create *dataset*.yaml** in 'config/dataset'

   ```yaml
   data_dir: "/path/to/data_root"
   dst:  "my_dataset"
    n_classes: "number_of_classes"
    patchify: "true/false (preprocess your dataset)"
    standardization: "true/false (preprocess your dataset)"
   ```

3. **Train and inference**

    ```bash
    python train.py
    ```
4. **Monitor**

    ```bash
    tensorboard --logdir logs
    ```

## Bringing your own data

Your dataset folder must initially contain only high‑resolution TIFF images:

```
data_root/my_dataset
├── img_0001.tif
├── img_0002.tif
└── ...
```

After preprocessing, the script adds:

```
data_root/my_dataset
├── img_0001.tif
├── img_0002.tif
├── ...
├── patches/        # extracted patches
├── std_mean/       # mean.pt & std.pt
└── results/        # segmentation maps
```
## Acknowledgements

- Original code: [Causal‑Unsupervised‑Segmentation](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation)
