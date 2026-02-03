import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import os
os.environ["YOLOV5_MODE"] = "native"
os.environ["ULTRALYTICS_DOWNLOAD_DIR"] = "/"
def get_color(idx):

    idx = int(idx) + 5
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

def convert_to_yolo_format(bbox, img_w, img_h):

    xmin, ymin, xmax, ymax = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    

    xmin = max(0, min(img_w, xmin))
    ymin = max(0, min(img_h, ymin))
    xmax = max(0, min(img_w, xmax))
    ymax = max(0, min(img_h, ymax))
    

    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def sahi_sliced_inference(model_path, image_path, output_dir, 
                         slice_height=2400, slice_width=2400,
                         overlap_height_ratio=0.0, overlap_width_ratio=0.4583,
                         confidence_threshold=0.21, 
                         model_type='yolov8', device='cuda:0',
                         class_list=None):

    

    output_dir = Path(output_dir)
    txt_dir = output_dir / "labels"
    viz_dir = output_dir / "visualized"
    txt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    

    img_path = str(image_path)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"can't load image: {image_path}")
    
    img_h, img_w = img.shape[:2]
    print(f"Processing: {image_path} ({img_w}x{img_h})")
    

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )
    

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        verbose=0
    )

    img_name = Path(image_path).stem
    txt_path = txt_dir / f"{img_name}.txt"
    viz_path = viz_dir / f"{img_name}_sahi.jpg"
    

    detections = result.object_prediction_list
    if not detections:
        print(f"No detections found for {image_path}")

        txt_path.touch()
    else:
        with open(txt_path, 'w') as f:
            for pred in detections:
                bbox = pred.bbox
                cls = pred.category.id
                score = pred.score.value
                

                x_center, y_center, width, height = convert_to_yolo_format(bbox, img_w, img_h)
                

                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
    

    viz_img = img.copy()
    for i, pred in enumerate(detections):
        bbox = pred.bbox
        cls = pred.category.id
        score = pred.score.value
        

        color = get_color(cls)
        cv2.rectangle(viz_img, 
                     (int(bbox.minx), int(bbox.miny)), 
                     (int(bbox.maxx), int(bbox.maxy)), 
                     color, 2, cv2.LINE_AA)

        if class_list and cls < len(class_list):
            label_text = f"{class_list[cls]} {score:.2f}"
        else:
            label_text = f"Class{cls} {score:.2f}"
        
        cv2.putText(viz_img, label_text, 
                   (int(bbox.minx), int(bbox.miny - 5)),
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, color, thickness=2)
    
    cv2.imwrite(str(viz_path), viz_img)
    
    print(f"Results saved:")
    print(f"  TXT: {txt_path} ({len(detections)} detections)")
    print(f"  Visualization: {viz_path}")
    

    for i, pred in enumerate(detections):
        bbox = pred.bbox
        cls = pred.category.id
        score = pred.score.value
        print(f"  Detection {i+1}: class={cls}, conf={score:.3f}, "
              f"bbox=({bbox.minx:.1f},{bbox.miny:.1f},{bbox.maxx:.1f},{bbox.maxy:.1f})")

def batch_sahi_inference(model_path, image_dir, output_dir, **kwargs):

    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in img_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            sahi_sliced_inference(model_path, img_file, output_dir, **kwargs)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="SAHI sliced inference with YOLO format output")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt file)")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Directory containing images (batch processing)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    

    parser.add_argument("--slice-height", type=int, default=2400, help="Height of image slices")
    parser.add_argument("--slice-width", type=int, default=2400, help="Width of image slices")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.0, help="Height overlap ratio")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.4583, help="Width overlap ratio")
    

    parser.add_argument("--conf", type=float, default=0.21, help="Confidence threshold")
    parser.add_argument("--model-type", type=str, default='ultralytics', 
                       choices=['yolov5', 'yolov8', 'yolo11', 'yolonas', 'detectron2', 'mmdet', 'huggingface'],
                       help="Model type for SAHI")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    

    parser.add_argument("--class-names", type=str, nargs='+', 
                       default=['MTL', 'TEE', 'BND', 'CRC', 'BRN', 'GWA', 'SWA', 'ESP', 'VAL', 'FLA', 'CAS', 'SLE'],
                       help="List of class names")
    
    args = parser.parse_args()

    kwargs = {
        'slice_height': args.slice_height,
        'slice_width': args.slice_width,
        'overlap_height_ratio': args.overlap_height_ratio,
        'overlap_width_ratio': args.overlap_width_ratio,
        'confidence_threshold': args.conf,
        'model_type': args.model_type,
        'device': args.device,
        'class_list': args.class_names
    }
    
    if args.image:

        sahi_sliced_inference(args.model, args.image, args.output, **kwargs)
    elif args.image_dir:

        batch_sahi_inference(args.model, args.image_dir, args.output, **kwargs)
    else:
        print("--image / --image_dir")

if __name__ == "__main__":
    main()