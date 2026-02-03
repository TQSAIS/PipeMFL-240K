from rfdetr import RFDETRBase
import torch

model = RFDETRBase(pretrain_weights="/root/autodl-tmp/rf-detr-base-coco.pth")

model.train(
    dataset_dir="/root/autodl-tmp/subsamples/ratio_16",
    epochs=100,
    batch_size=8,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="/root/autodl-tmp/rfdetr_ratio16",
    device='cuda:0',
    patience=10
)
