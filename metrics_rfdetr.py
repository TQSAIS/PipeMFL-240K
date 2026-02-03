#!/usr/bin/env python3

import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tabulate import tabulate
from podm.metrics import BoundingBox, get_pascal_voc_metrics, MetricPerClass
import argparse
from tqdm import tqdm
from rfdetr import RFDETRBase


CUSTOM_CLASSES = [
    'MTL', 'TEE', 'BND', 'CRC', 'BRN', 'GWA',
    'SWA', 'ESP', 'VAL', 'FLA', 'CAS', 'SLE'
]


def get_color(idx):
    idx = int(idx) + 5
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def read_boxes(txt_file, img_w, img_h):
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            cls = int(items[0])
            x_c, y_c, w, h = map(float, items[1:5])
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h
            boxes.append([cls, x1, y1, x2, y2])
    return boxes


def load_detection_model():
    weights =r'/root/autodl-tmp/rfdetr_ratio16/checkpoint_best_ema.pth'
    class_mapping = {i: class_name for i, class_name in enumerate(args.class_list)}
    print(class_mapping)
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model=RFDETRBase(pretrain_weights=weights),  # pass an instance of a trained model
        confidence_threshold=0.25,
        category_mapping=class_mapping,
        device="cuda:1",
)


parser = argparse.ArgumentParser(description="Object Detection Evaluation Script")
parser.add_argument('--filepath', type=str, default=r'/root/autodl-tmp/test/images',
                    help='Path to the images folder')
parser.add_argument('--annotation_folder', type=str, default=r'/root/autodl-tmp/test/labels',
                    help='Path to the annotation folder')
parser.add_argument('--model_type', type=str, default='roboflow', help='Type of the detection model')
parser.add_argument('--model_path', type=str,
                    default=r'/root/autodl-tmp/rfdetr_ratio16/checkpoint_best_ema.pth',
                    help='Path to the RF-DETR weights')
parser.add_argument('--confidence_threshold', type=float, default=0.35, help='Confidence threshold for the model')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run the model on')
parser.add_argument('--slice_height', type=int, default=2400, help='Height of the image slices')
parser.add_argument('--slice_width', type=int, default=2400, help='Width of the image slices')
parser.add_argument('--overlap_height_ratio', type=float, default=0.0, help='Overlap height ratio for slicing')
parser.add_argument('--overlap_width_ratio', type=float, default=0.4583, help='Overlap width ratio for slicing')
parser.add_argument('--visualize_predictions', action='store_true', default=False, help='Visualize prediction results')
parser.add_argument('--visualize_annotations', action='store_true', default=False, help='Visualize annotation results')
parser.add_argument('--class_list', type=str, nargs='+', default=CUSTOM_CLASSES, help='List of class names')
parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for evaluation')
parser.add_argument('--images_format', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                    help='List of acceptable image formats')
parser.add_argument('--max_images', type=int, default=50000, help='max images for testing')
args = parser.parse_args()

EVAL_EVERY = 5000

def process_image(image_name, model, labels, detections):
    img_path = os.path.join(args.filepath, image_name)
    img_vis = cv2.imread(img_path)
    img_h, img_w, _ = img_vis.shape

    result = get_sliced_prediction(
        img_path,
        model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        verbose=0
    )

    anno_file = os.path.join(args.annotation_folder, image_name[:-4] + '.txt')
    annotations = read_boxes(anno_file, img_w, img_h)
    for anno in annotations:
        label, xmin_gt, ymin_gt, xmax_gt, ymax_gt = anno
        labels.append(BoundingBox.of_bbox(image_name, label, xmin_gt, ymin_gt, xmax_gt, ymax_gt))

    for pred in result.object_prediction_list:
        cls = pred.category.id
        score = pred.score.value
        xmin_pd, ymin_pd, xmax_pd, ymax_pd = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        detections.append(BoundingBox.of_bbox(image_name, cls, xmin_pd, ymin_pd, xmax_pd, ymax_pd, score))

    if args.visualize_predictions or args.visualize_annotations:
        save_dir = "visualization_results"
        os.makedirs(save_dir, exist_ok=True)
        if args.visualize_annotations:
            for anno in annotations:
                cls, x1, y1, x2, y2 = anno
                cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), get_color(cls), 2)
        if args.visualize_predictions:
            for pred in result.object_prediction_list:
                cls = pred.category.id
                x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
                cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), get_color(cls + 100), 2)
        flag = "_pred" if args.visualize_predictions else ""
        flag += "_anno" if args.visualize_annotations else ""
        cv2.imwrite(os.path.join(save_dir, f"{image_name[:-4]}{flag}.jpg"), img_vis)
def evaluate_model(labels, detections):
    results = get_pascal_voc_metrics(labels, detections, args.iou_thresh)
    map50 = MetricPerClass.mAP(results)
    
    # mAP@0.5:0.95
    aps = []
    for iou in [x / 100 for x in range(50, 100, 5)]:
        res = get_pascal_voc_metrics(labels, detections, iou)
        aps.append(MetricPerClass.mAP(res))
    map5095 = sum(aps) / len(aps) if aps else 0.0

    table = []
    for cls, m in results.items():
        if m.num_groundtruth > 0:

            recall_val = m.recall[-1] if len(m.recall) > 0 else 0.0
            precision_val = m.precision[-1] if len(m.precision) > 0 else 0.0
            
            table.append([
                args.class_list[int(cls)], 
                recall_val, 
                precision_val, 
                m.ap
            ])
    
    if table:
        print(tabulate(table, headers=["Class", "Recall", "Precision", "AP@0.5"], floatfmt=".4f"))
    else:
        print("no valid result）")
    
    print(f"\nmAP@0.5:   {map50:.4f}")
    print(f"mAP@0.5:0.95: {map5095:.4f}")

def main():
    model = load_detection_model()
    image_names = [n for n in os.listdir(args.filepath)
                   if os.path.splitext(n)[1].lower() in args.images_format][:args.max_images]
    labels, detections = [], []
    print(f" {len(image_names)} images， mAP validation per {EVAL_EVERY}  ...")
    for i, name in enumerate(tqdm(image_names, desc="Processing")):
        process_image(name, model, labels, detections)
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == len(image_names):
            print(f"\n-----  {i+1} images processed -----")
            if len(labels) and len(detections):
                evaluate_model(labels, detections)
            else:
                print("no evaluation data")
            print("-" * 40)
    print("\n===== final metrics =====")
    evaluate_model(labels, detections)

if __name__ == "__main__":
    main()