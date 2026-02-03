


import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tabulate import tabulate
from podm.metrics import BoundingBox, get_pascal_voc_metrics, MetricPerClass
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Object Detection Evaluation Script")
parser.add_argument('--filepath', type=str, default=r'/root/autodl-tmp/test/images',
                    help='Path to the images folder')
parser.add_argument('--annotation_folder', type=str, default=r'/root/autodl-tmp/test/labels',
                    help='Path to the annotation folder')

parser.add_argument('--model_type', type=str, default='torchvision', help='Type of the detection model')
parser.add_argument('--model_path', type=str,
                    default=r'/root/autodl-tmp/runs/det/epoch_063.pth',
                    help='Path to the model weights')
parser.add_argument('--confidence_threshold', type=float, default=0.35, help='Confidence threshold for the model')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run the model on')

parser.add_argument('--slice_height', type=int, default=2400, help='Height of the image slices')
parser.add_argument('--slice_width', type=int, default=2400, help='Width of the image slices')
parser.add_argument('--overlap_height_ratio', type=float, default=0.0, help='Overlap height ratio for slicing')
parser.add_argument('--overlap_width_ratio', type=float, default=0.4583, help='Overlap width ratio for slicing')

parser.add_argument('--visualize_predictions', action='store_true', default=False, help='Visualize prediction results')
parser.add_argument('--visualize_annotations', action='store_true', default=False, help='Visualize annotation results')

parser.add_argument('--class_list', type=str, nargs='+',
                    default=['MTL', 'TEE', 'BND', 'CRC', 'BRN', 'GWA', 'SWA', 'ESP', 'VAL', 'FLA', 'CAS', 'SLE'],
                    help='List of class names')
parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for evaluation')
parser.add_argument('--images_format', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                    help='List of acceptable image formats')
parser.add_argument('--max_images', type=int, default=50000,
                    help='max images for testing')
args = parser.parse_args()
import torchvision
import torch
def get_color(idx):
    idx = int(idx) + 5
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def read_boxes(txt_file, img_w, img_h):
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            box = [
                int(items[0]),
                (float(items[1]) - float(items[3]) / 2) * img_w,
                (float(items[2]) - float(items[4]) / 2) * img_h,
                (float(items[1]) + float(items[3]) / 2) * img_w,
                (float(items[2]) + float(items[4]) / 2) * img_h
            ]
            boxes.append(box)
    return boxes


def load_detection_model():
    detection_model = AutoDetectionModel.from_pretrained(
    model_type='torchvision',
    confidence_threshold=0.25,
    model_path='/root/autodl-tmp/runs/retinanet/epoch_07.pth',
    config_path=None,
    device='cuda:0',
    category_mapping={
    '0': 'MTL',
    '1': 'TEE',
    '2': 'BND',
    '3': 'CRC',
    '4': 'BRN',
    '5': 'GWA',
    '6': 'SWA',
    '7': 'ESP',
    '8': 'VAL',
    '9': 'FLA',
    '10': 'CAS',
    '11': 'SLE'
    },
    load_at_init=False,  )


    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False, num_classes=13
    )
    ckpt = torch.load('/root/autodl-tmp/runs/retinanet/epoch_048.pth',map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(detection_model.device)
    detection_model.model = model
    
    return detection_model



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
        if args.visualize_annotations:
            cv2.rectangle(img_vis, (int(xmin_gt), int(ymin_gt)), (int(xmax_gt), int(ymax_gt)), get_color(label), 2,
                          cv2.LINE_AA)
            cv2.putText(img_vis, f"{args.class_list[label]}", (int(xmin_gt), int(ymin_gt - 5)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, get_color(label), thickness=2)

    for pred in result.object_prediction_list:
        bbox = pred.bbox
        cls = pred.category.id
        score = pred.score.value
        xmin_pd, ymin_pd, xmax_pd, ymax_pd = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        detections.append(BoundingBox.of_bbox(image_name, cls, xmin_pd, ymin_pd, xmax_pd, ymax_pd, score))

        if args.visualize_predictions:
            cv2.rectangle(img_vis, (int(xmin_pd), int(ymin_pd)), (int(xmax_pd), int(ymax_pd)),
                          get_color(cls + len(args.class_list)), 2, cv2.LINE_AA)
            cv2.putText(img_vis, f"{args.class_list[cls]} {score:.2f}", (int(xmin_pd), int(ymin_pd - 5)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, get_color(cls + len(args.class_list)), thickness=2)


    if args.visualize_predictions or args.visualize_annotations:

        save_dir = "visualization_results"
        os.makedirs(save_dir, exist_ok=True)


        if args.visualize_predictions and args.visualize_annotations:
            filename = f"{image_name}_both.jpg"
        elif args.visualize_predictions:
            filename = f"{image_name}_predictions.jpg"
        else:
            filename = f"{image_name}_annotations.jpg"

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_vis)


def evaluate_model(labels, detections):
    # ---------- mAP@0.5  ----------
    results_50 = get_pascal_voc_metrics(labels, detections, 0.5)
    map_50 = MetricPerClass.mAP(results_50)

    # ---------- mAP@0.5:0.95 ----------
    aps = []
    for iou in [x / 100 for x in range(50, 100, 5)]:  # 0.50, 0.55, ..., 0.95
        res = get_pascal_voc_metrics(labels, detections, iou)
        aps.append(MetricPerClass.mAP(res))
    map_50_95 = sum(aps) / len(aps)


    table = [
        [args.class_list[int(class_id)],
         results_50[class_id].recall[-1],
         results_50[class_id].precision[-1],
         results_50[class_id].ap]
        for class_id in results_50 if results_50[class_id].num_groundtruth > 0
    ]
    print(tabulate(table, headers=["ClassID", "Recall", "Precision", "AP@0.5"], floatfmt=".2f"))
    print(f"\nmAP@0.5:   {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")
def evaluate_model(labels, detections):
    # ---------- mAP@0.5  ----------
    results_50 = get_pascal_voc_metrics(labels, detections, 0.5)
    map_50 = MetricPerClass.mAP(results_50)

    # ---------- mAP@0.5:0.95 ----------
    aps = []
    for iou in [x / 100 for x in range(50, 100, 5)]:
        res = get_pascal_voc_metrics(labels, detections, iou)
        aps.append(MetricPerClass.mAP(res))
    map_50_95 = sum(aps) / len(aps)

    # ---------- table ----------
    table = []
    for class_id, m in results_50.items():
        if m.num_groundtruth == 0:
            continue
        if len(m.recall) == 0 or len(m.precision) == 0:
            continue
        table.append([
            args.class_list[int(class_id)],
            m.recall[-1],
            m.precision[-1],
            m.ap
        ])

    if table:
        print(tabulate(table, headers=["ClassID", "Recall", "Precision", "AP@0.5"], floatfmt=".4f"))
    else:
        print("no ground-truth")

    print(f"\nmAP@0.5:   {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")

EVAL_EVERY = 5000      # evaluate per 5000 images

def main():
    detection_model = load_detection_model()
    image_names = [name for name in os.listdir(args.filepath) if
                   os.path.splitext(name)[1].lower() in args.images_format][:args.max_images]
    labels, detections = [], []

    print(f" {len(image_names)} images， mAP validation per {EVAL_EVERY}  ...")
    for i, image_name in enumerate(tqdm(image_names, desc="Processing")):
        process_image(image_name, detection_model, labels, detections)

        # ===== 阶段性评估 =====
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == len(image_names):
            print(f"\n-----  {i+1} images processed -----")
            if len(labels) and len(detections):
                evaluate_model(labels, detections)
            else:
                print("no evaluation data")
            print("-" * 40)

    # 最后再打一次最终结果
    print("\n===== final metrics =====")
    evaluate_model(labels, detections)


if __name__ == "__main__":
    main() 