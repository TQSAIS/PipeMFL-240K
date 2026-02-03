
import os
import json
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

def stratified_sample_coco(image_files, img_to_cats, target_total, min_per_class=2, seed=42):

    random.seed(seed)
    cat_to_images = defaultdict(list)
    
    for img_file, cats in img_to_cats.items():
        if img_file in image_files:
            for cat in cats:
                cat_to_images[cat].append(img_file)
    
    sampled = set()
    remaining = target_total
    
    for cat_id in list(cat_to_images.keys()):
        imgs = cat_to_images[cat_id]
        if not imgs:
            continue
        quota = min(min_per_class, len(imgs), remaining)
        if quota > 0:
            sampled.update(random.sample(imgs, quota))
            remaining -= quota
    
    if remaining > 0:
        available = [f for f in image_files if f not in sampled]
        if len(available) < remaining:
            to_add = available
        else:
            to_add = random.sample(available, remaining)
        sampled.update(to_add)
    
    return list(sampled)

def create_rfdetr_coco_subsample(base_dir, ratio, output_root, seed=42):

    random.seed(seed)
    base_dir = Path(base_dir).resolve()
    output_root = Path(output_root).resolve()
    
    train_dir = base_dir / 'train'
    json_candidates = list(train_dir.glob('*.json')) + list(train_dir.glob('*.coco.json'))
    
    if not json_candidates:
        raise FileNotFoundError(f" COCO JSON not found!")
    
    train_json = json_candidates[0]

    

    with open(train_json) as f:
        coco_data = json.load(f)
    

    img_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    img_to_cats = defaultdict(set)
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_file:
            file_name = img_id_to_file[img_id]
            img_to_cats[file_name].add(ann['category_id'])
    

    possible_img_dirs = [train_dir, train_dir / 'images']
    existing_files = set()
    img_dir_used = None
    
    for img_dir in possible_img_dirs:
        if img_dir.exists():
            files = [f.name for f in img_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
            if files:
                existing_files.update(files)
                img_dir_used = img_dir
    
    if not img_dir_used:
        raise FileNotFoundError("image not found!")
    

    labeled_images = [f for f in existing_files if f in img_to_cats]
    target_num = max(len(labeled_images) // ratio, 1)
    
    sampled_images = stratified_sample_coco(
        labeled_images, img_to_cats, target_num, min_per_class=max(2, 12//ratio), seed=seed
    )
    print(f"[INFO] sampling 1/{ratio} = {len(sampled_images)} images")
    

    out_dir = output_root / f'ratio_{ratio}'
    new_train_dir = out_dir / 'train'
    new_train_dir.mkdir(parents=True, exist_ok=True)
    

    link_count = 0
    for img_name in sampled_images:
        src = img_dir_used / img_name
        dst = new_train_dir / img_name
        
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        
        try:
            os.symlink(src.resolve(), dst)
            link_count += 1
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    

    

    sampled_set = set(sampled_images)
    new_images = [img for img in coco_data['images'] if img['file_name'] in sampled_set]
    sampled_ids = {img['id'] for img in new_images}
    new_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in sampled_ids]
    
    new_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data.get('categories', [])
    }
    

    new_json_path = new_train_dir / train_json.name
    with open(new_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)
    

    for split in ['valid', 'val', 'test']:
        src_split = base_dir / split
        if src_split.exists():
            dst_split = out_dir / split
            if dst_split.exists() or dst_split.is_symlink():
                if dst_split.is_symlink():
                    dst_split.unlink()
                else:
                    shutil.rmtree(dst_split)
            

            try:
                os.symlink(src_split.resolve(), dst_split, target_is_directory=True)

            except OSError:
                shutil.copytree(src_split, dst_split)

            break
    

    cat_count = defaultdict(int)
    for ann in new_annotations:
        cat_count[ann['category_id']] += 1

    
    return str(out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--ratio', type=int, required=True, choices=[2, 4, 8, 16])
    parser.add_argument('--output-root', type=str, default='./rfdetr_subsets')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    

    out_dir = Path(args.output_root) / f"ratio_{args.ratio}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    
    create_rfdetr_coco_subsample(args.base_dir, args.ratio, args.output_root, args.seed)