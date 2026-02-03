#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, random, argparse
from typing import List, Dict

import numpy as np
import torch, torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------- tools ---------------- #
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


# dataset #
class CocoDetDataset(Dataset):
    def __init__(self, root: str, annFile: str, train: bool = True, min_size: int = 640, pos_ratio: float = 0.9):
        self.root = root; self.coco = COCO(annFile); self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train; self.min_size = min_size; self.pos_ratio = pos_ratio
        # 1->0 mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat2cont = {c: i for i, c in enumerate(self.cat_ids)}
        if self.train and self.pos_ratio is not None:
            self.ids = self._sample_ids()
        self.transform = T.Compose([T.ToTensor()])

    def _sample_ids(self) -> List[int]:
        pos, neg = [], []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            (pos if len(ann_ids) else neg).append(img_id)
        random.shuffle(neg)
        need_neg = int(len(pos) * (1 - self.pos_ratio) / max(self.pos_ratio, 1e-6))
        return pos + neg[:max(1, need_neg)]

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]["file_name"])
        img = torchvision.io.read_image(img_path)        # RGB uint8
        if img.shape[0] == 1: img = img.repeat(3, 1, 1)
        _, h0, w0 = img.shape
        scale = self.min_size / min(h0, w0)
        if scale != 1.0: img = T.functional.resize(img, [int(h0 * scale), int(w0 * scale)])
        img = img.float() / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        if not anns:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)
        else:
            xywh = torch.tensor([o["bbox"] for o in anns], dtype=torch.float32)
            xyxy = xywh.clone(); xyxy[:, 2:] += xyxy[:, :2]
            keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            if keep.sum().item() == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
                area = torch.empty((0,), dtype=torch.float32)
                iscrowd = torch.empty((0,), dtype=torch.int64)
            else:
                boxes = xyxy[keep]
                wh = boxes[:, 2:] - boxes[:, :2]; area = wh[:, 0] * wh[:, 1]
                labels = torch.tensor([self.cat2cont[o["category_id"]] for i, o in enumerate(anns) if keep[i]],
                                      dtype=torch.int64)
                iscrowd = torch.tensor([o.get("iscrowd", 0) for i, o in enumerate(anns) if keep[i]],
                                       dtype=torch.int64)

        target = dict(boxes=boxes, labels=labels, image_id=torch.tensor(img_id),
                      area=area, iscrowd=iscrowd)
        return img, target

    def __len__(self): return len(self.ids)


# ---------------- model generate ---------------- #
def build_retinanet(num_classes: int, pretrained: bool = True):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights="DEFAULT" if pretrained else None)


    old_cls_head = model.head.classification_head
    old_reg_head = model.head.regression_head
    in_channels = old_cls_head.conv[0][0].out_channels
    num_anchors = old_cls_head.num_anchors

    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        prior_probability=0.01)
    model.head.regression_head = torchvision.models.detection.retinanet.RetinaNetRegressionHead(
        in_channels=in_channels,
        num_anchors=num_anchors)
    return model


# ---------------- train one epoch ---------------- #
def train_one_epoch(model, optimizer, loader, device, epoch, amp=True, max_norm=0.0, log_interval=50):
    model.train()
    meter = AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for step, (imgs, tgts) in enumerate(loader, 1):
        imgs = list(img.to(device) for img in imgs)
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            print(f"[NaN/Inf] epoch={epoch} step={step}")
            for k, v in loss_dict.items(): print(f"  {k}={v.item()}")
            raise RuntimeError("Loss NaN/Inf")

        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        meter.update(loss.item(), len(imgs))
        if step % log_interval == 0 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch[{epoch:3d}] {step:5d}/{len(loader)}  loss={meter.avg:.4f}  lr={lr:.2e}")
    return meter.avg


# ---------------- COCO eval ---------------- #
@torch.no_grad()
def evaluate_coco(model, loader, coco_gt, id_map, device):
    model.eval()
    results = []
    for imgs, tgts in loader:
        imgs = list(img.to(device) for img in imgs)
        outs = model(imgs)
        for out, tgt in zip(outs, tgts):
            img_id = int(tgt["image_id"].item())
            if out["boxes"].numel() == 0: continue
            boxes = out["boxes"].cpu(); scores = out["scores"].cpu(); labels = out["labels"].cpu()
            wh = boxes[:, 2:] - boxes[:, :2]; xywh = torch.cat([boxes[:, :2], wh], 1)
            for b, s, l in zip(xywh, scores, labels):
                results.append({
                    "image_id": img_id,
                    "category_id": int(id_map[l.item()]),
                    "bbox": [float(x) for x in b],
                    "score": float(s),
                })
    if not results: return {"AP": 0.0, "AP50": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return {"AP": coco_eval.stats[0], "AP50": coco_eval.stats[1]}


# ---------------- main ---------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        choices=["retinanet"], default="retinanet")
    p.add_argument("--train-root",   required=True)
    p.add_argument("--val-root",     required=True)
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch",        type=int, default=32)
    p.add_argument("--lr",           type=float, default=0.0005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers",      type=int, default=16)
    p.add_argument("--min-size",     type=int, default=640)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--output",       type=str, default="runs/retinanet")
    p.add_argument("--resume",       type=str, default="")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)

    # 1. dataset
    train_ds = CocoDetDataset(root=os.path.join(args.train_root, "images"),
                              annFile=os.path.join(args.train_root, "_annotations.coco.json"),
                              train=True, min_size=args.min_size, pos_ratio=1.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    # 2. RetinaNet
    num_classes = len(train_ds.cat_ids) + 1          # + background
    model = build_retinanet(num_classes, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # 3. resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)

    # 4. training epoch
    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== Epoch {epoch+1:3d}/{args.epochs} ==========")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, amp=args.amp)
        print(f"Epoch[{epoch+1:3d}]  train_loss={train_loss:.4f}")

        # save per 3 epoch
        if (epoch + 1) % 3 == 0:
            ckpt_name = os.path.join(args.output, f"epoch_{epoch+1:03d}.pth")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1}, ckpt_name)
            print(f"  saved  ->  {ckpt_name}")

    print("Training finished.")


if __name__ == "__main__":
    main()