
import argparse, os, yaml, random, shutil
from pathlib import Path
from collections import defaultdict

def stratified_sample(images, k):

    cls2imgs = defaultdict(list)
    for img in images:
        img_path = Path(img)  
        lbl = img_path.with_suffix('.txt') 
        lbl = Path(str(lbl).replace('/images/', '/labels/'))
        if lbl.exists() and lbl.stat().st_size:
            with open(lbl) as f:
                cls = set(int(line.split()[0]) for line in f)
            for c in cls:
                cls2imgs[c].append(img)
        else:
            cls2imgs[-1].append(img)

    total = sum(len(v) for v in cls2imgs.values())
    sampled = []
    for c, imgs in cls2imgs.items():
        need = max(1, int(round(len(imgs) * k / total)))
        sampled.extend(random.sample(imgs, min(need, len(imgs))))

    if len(sampled) > k:
        sampled = random.sample(sampled, k)
    while len(sampled) < k:
        sampled.append(random.choice(images))
    random.shuffle(sampled)
    return sampled

def make_subsample_dataset(opt):

    balanced_train_list = Path(opt.data).with_name('train_balanced.txt')
    if not balanced_train_list.exists():
        raise FileNotFoundError('9:1 not found!')
    with open(balanced_train_list) as f:
        balanced_train = [l.strip() for l in f if l.strip()]

    ratio_map = {2: 2, 4: 4, 8: 8, 16: 16, 64: 64}
    downsample_ratio = ratio_map[opt.ratio]
    target_num = len(balanced_train) // downsample_ratio


    subsample_train = stratified_sample(balanced_train, target_num)
    print(f'[Subsample]  {len(balanced_train)} -> 1/{downsample_ratio} = {len(subsample_train)}')


    out_dir = Path('runs/subsample') / f'ratio_{opt.ratio}'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list_path = out_dir / 'train.txt'
    with open(train_list_path, 'w') as f:
        for p in subsample_train:
            f.write(p + '\n')


    with open(opt.data) as f:
        cfg = yaml.safe_load(f)
    cfg['train'] = str(train_list_path.resolve())

    new_yaml = out_dir / 'data.yaml'
    with open(new_yaml, 'w') as f:
        yaml.dump(cfg, f)
    return new_yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml')
    parser.add_argument('--ratio', type=int, choices=[2, 4, 8, 16, 64], required=True, help=' 2/4/8/16/64')
    args = parser.parse_args()

    new_yaml = make_subsample_dataset(args)


if __name__ == '__main__':
    main()