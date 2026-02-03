import sys
import argparse
import os
import yaml
import random
import shutil
from pathlib import Path



def create_balanced_datasets(opt):



    with open(opt.data, 'r') as f:
        data_config = yaml.safe_load(f)


    train_path = data_config.get('train')
    val_path   = data_config.get('val')

    if not train_path or not val_path:

        sys.exit(1)


    def load_image_paths(path):
        if path.endswith('.txt'):
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            images = []
            path = Path(path)
            images_dir = path / 'images' if (path / 'images').exists() else path
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                images.extend([str(p) for p in images_dir.rglob(f'*{ext}')])
            return images

    train_images = load_image_paths(train_path)
    val_images   = load_image_paths(val_path)




    def get_label_file(img_path):
        img_file = Path(img_path)
        if 'images' in img_path:
            label_path = img_path.replace('images', 'labels').replace(img_file.suffix, '.txt')
            return Path(label_path)
        else:
            label_dir = img_file.parent / 'labels'
            return label_dir / img_file.with_suffix('.txt').name

    def is_positive_image(img_path):
        label_file = get_label_file(img_path)
        return label_file.exists() and label_file.stat().st_size > 0


    train_positives = [img for img in train_images if is_positive_image(img)]
    train_negatives = [img for img in train_images if not is_positive_image(img)]


    if opt.train_positive_ratio >= 1.0:
        balanced_train      = train_positives
        remaining_negatives = train_negatives
    else:
        desired_train_negatives = int(len(train_positives) *
                                      (1 - opt.train_positive_ratio) /
                                      opt.train_positive_ratio)
        if len(train_negatives) > desired_train_negatives:
            selected_negatives = random.sample(train_negatives, desired_train_negatives)
            remaining_negatives = [img for img in train_negatives if img not in selected_negatives]
        else:
            selected_negatives  = train_negatives
            remaining_negatives = []

        balanced_train = train_positives + selected_negatives
        random.shuffle(balanced_train)


    val_positives = [img for img in val_images if is_positive_image(img)]
    val_negatives = [img for img in val_images if not is_positive_image(img)]


    neg_samples_needed = 0
    if len(remaining_negatives) >= neg_samples_needed:
        selected_val_negatives = random.sample(remaining_negatives, neg_samples_needed)
    else:
        selected_val_negatives = remaining_negatives


    expanded_val = val_images + selected_val_negatives
    random.shuffle(expanded_val)

    new_val_total      = len(expanded_val)
    new_val_positives  = len(val_positives)
    new_val_negatives  = len(val_negatives) + len(selected_val_negatives)
    new_val_positive_ratio = new_val_positives / new_val_total * 100


    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='yolo_data_')

    train_list_path = os.path.join(temp_dir, 'train_balanced.txt')
    with open(train_list_path, 'w') as f:
        for img_path in balanced_train:
            f.write(f"{img_path}\n")

    val_list_path = os.path.join(temp_dir, 'val_expanded.txt')
    with open(val_list_path, 'w') as f:
        for img_path in expanded_val:
            f.write(f"{img_path}\n")

    balanced_config = data_config.copy()
    balanced_config['train'] = train_list_path
    balanced_config['val']   = val_list_path

    balanced_config_path = os.path.join(temp_dir, 'data_balanced.yaml')
    with open(balanced_config_path, 'w') as f:
        yaml.dump(balanced_config, f, default_flow_style=False)


    return balanced_config_path, temp_dir


def main(opt):


    yolo_config = 'yolov8l-worldv2.yaml'


    original_data_path = opt.data
    temp_dir = None

    if opt.train_positive_ratio > 0:
        opt.data, temp_dir = create_balanced_datasets(opt)
    else:
        print("no balance")


    from ultralytics import YOLO

    model = YOLO('yolov8l-worldv2.pt')

    model.info()

    try:
        results = model.train(
            data=opt.data,
            epochs=opt.epochs,
            imgsz=opt.imgsz,
            workers=opt.workers,
            batch=opt.batch,
            device=opt.device,
            amp=True,
            pretrained=True,
            rect=True,
            save=True,
            save_period=10,
            val=False
        )
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()


    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"error: {e}")

    opt.data = original_data_path


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz',   type=int, default=640)
    parser.add_argument('--batch',   type=int, default=256)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--device',  type=str, default='0,1,2,3')

    parser.add_argument('--train-positive-ratio', type=float, default=0.9,
                        help='positive sample ratio (0.0-1.0)，default: 0.9')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)