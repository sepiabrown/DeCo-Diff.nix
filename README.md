# DeCo-Diff
This Repository contain the PyTorch implementation of the multi-class unsupervised anomaly detection method: "Correcting Deviations from Normality: A Reformulated Diffusion Model for Unsupervised Anomaly Detection."


## Setup

### Environment

We utilize the `Python 3.11` interpreter in our experiments. Install the required packages using the following command:
```bash
pip3 install -r requirements.txt
```

### Datasets

Download [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) or [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) datasets, and organize them in the following file structure (the default structure of MVTec-AD):
```
├── class1
│   ├── ground_truth
│   │   ├── defect1
│   │   └── defect2
│   ├── test
│   │   ├── defect1
│   │   ├── defect2
│   │   └── good
│   └── train
│       └── good
├── class2
...
```

## Train

Train our RLR with the following command:

```bash
torchrun evaluation_DeCo_Diff.py \
            --dataset mvtec \
            --model UNet_L \
            --mask-random-ratio True \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True \
            --ckpt-every 20 
```

## Test

Test the model with the following command:

```bash
python evaluation_DeCo_Diff.py \
            --dataset mvtec \
            --model UNet_L \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True \
```
