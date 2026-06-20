<div align="center">

# DA-F2F: Domain-Adaptive Object Detection with Feature-to-Feature Modulation and Alignment

</div>

<p align="center">
  <img src="assets/overview.png" width="95%">
</p>

## Introduction

**DA-F2F** is a domain-adaptive object detection framework that directly performs domain transformation in the feature space.

Existing image-to-image translation based DAOD methods often suffer from unstable pixel-level artifacts and additional computational overhead. Instead of translating images, DA-F2F performs **feature-to-feature modulation**, transforming source-domain representations into target-aligned feature representations.

DA-F2F consists of two main components:

* **Style-Aware Feature Modulation (SFM)**
  Transfers target-domain style statistics to source features and generates target-aligned source representations.

* **Soft-Weighted Proposal Alignment (SPA)**
  Performs region-separated adversarial alignment with a soft-weighted strategy for robust foreground/background alignment in the target domain.


## Main Contributions

In this paper, we:

1. Develop **Style-Aware Feature Modulation (SFM)**, which implements feature-to-feature modulation by transforming source-domain representations into the target domain within the feature space.

2. Introduce **Soft-Weighted Proposal Alignment (SPA)**, which improves adversarial learning by stably separating foreground and background regions in the target domain.

3. Demonstrate strong performance across multiple domain shift scenarios, including weather adaptation, small-to-large scale dataset adaptation, and synthetic-to-real adaptation.

## Requirements

Install the required Python packages using `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

We recommend using `python -m pip` instead of `pip` to ensure that packages are installed into the currently activated conda environment.

Note that `torch`, `torchvision`, and `detectron2` are CUDA-dependent. If they are not already installed, please install versions compatible with your CUDA environment before running DA-F2F.

## Dataset Preparation

Please organize the datasets as follows:

```text
datasets/
├── cityscapes/
├── foggy_cityscapes/
├── bdd100k/
└── sim10k/
```

## Training

```bash
python tools/train_net.py \
    --config-file ${CONFIG_FILE}
```


For multi-GPU training, specify the visible GPUs and the number of GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
    --config-file ${CONFIG_FILE} \
    --num-gpus 4
```

To resume training from a checkpoint, run the following command.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
    --config-file configs/cityscapes/daf2f-cityscapes.yaml \
    --num-gpus 4 \
    --resume
```

## Evaluation

```bash
python tools/train_net.py \
    --config-file ${CONFIG_FILE} \
    --eval-only \
    MODEL.WEIGHTS path/to/model.pth
```

## Results

| Adaptation Scenario           |  mAP |
| ----------------------------- | ---: |
| Cityscapes → Foggy Cityscapes | 60.5 |
| Cityscapes → BDD100K-daytime  | 46.9 |
| Sim10K → Cityscapes           | 69.9 |

## Model Zoo

| Scenario                      | Checkpoint  |
| ----------------------------- | ----------- |
| Cityscapes → Foggy Cityscapes | Coming soon |
| Cityscapes → BDD100K-daytime  | Coming soon |
| Sim10K → Cityscapes           | Coming soon |

