# SSG

## Results

## Download 

## Install

### Clone this repository:
```shell script
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
│   └── config.yaml
├── loader/                 # data loading & preprocessing
├── models/                 # ViT backbones (DINO, DINOv2, MAE, iBOT, MSN)
├── modules/                # segmentation heads & utilities
├── outputs/                # created automatically – logs, checkpoints, hydra configs
├── train.py                # full pipeline
├── train_mediator.py       # generate clustrbook
├── train_front_door.py     # frontdoor adjustment through constrastive learning
├── fine_tuning.py          # repeat clustering on refined concepts
├── inference.py            # inference on images
└── ...
```

## Quick start

1. **Download pre‑trained backbones from** [**Causal‑Unsupervised‑Segmentation**](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation)\
   Place the checkpoint files under `ckpt/` (path configurable in `config.yaml`).

2. **Edit config.yaml**

   ```yaml
   data_dir: "/path/to/data_root"
   dataset:  "my_dataset"
   ...
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

To train CAUSE_tomo on your own dataset please specify in [`CAUSE_tomo/config/config.yaml`](/config/config.yaml) set the following parameters:

```yaml
data_dir: "directory"
dataset: "dataset_name"
```