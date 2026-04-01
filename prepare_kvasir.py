import argparse
import os
import random
import cv2
import numpy as np


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def collect_pairs(src_root):
    image_dir = os.path.join(src_root, 'images')
    mask_dir = os.path.join(src_root, 'masks')

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image folder not found: {image_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask folder not found: {mask_dir}')

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )

    common = sorted(list(set(image_files) & set(mask_files)))
    if not common:
        raise ValueError('No matching image/mask filenames found in source folders.')

    missing_images = sorted(set(mask_files) - set(image_files))
    missing_masks = sorted(set(image_files) - set(mask_files))
    if missing_images or missing_masks:
        print('Warning: some files are not paired:')
        if missing_masks:
            print(f'  Images without masks: {len(missing_masks)}')
        if missing_images:
            print(f'  Masks without images: {len(missing_images)}')

    return common


def write_npy_files(src_root, out_root, names):
    image_dir = os.path.join(src_root, 'images')
    mask_dir = os.path.join(src_root, 'masks')
    out_images = os.path.join(out_root, 'images')
    out_labels = os.path.join(out_root, 'labels')
    makedirs(out_images)
    makedirs(out_labels)

    for name in names:
        image_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {image_path}')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Cannot read mask: {mask_path}')

        mask = (mask > 127).astype(np.uint8)

        base = os.path.splitext(name)[0]
        np.save(os.path.join(out_images, base + '.npy'), image)
        np.save(os.path.join(out_labels, base + '.npy'), mask)

    print(f'Saved {len(names)} images to {out_images}')
    print(f'Saved {len(names)} masks to {out_labels}')


def write_fold_files(names, out_root, folds, seed):
    random.seed(seed)
    names = names.copy()
    random.shuffle(names)

    num = len(names)
    fold_sizes = [num // folds + (1 if i < num % folds else 0) for i in range(folds)]
    idx = 0

    for fold_idx, size in enumerate(fold_sizes, start=1):
        part = names[idx: idx + size]
        idx += size
        filename = os.path.join(out_root, f'fold_label_{fold_idx}.txt')
        with open(filename, 'w') as f:
            for name in part:
                f.write(name + '\n')
        print(f'Wrote fold {fold_idx}: {len(part)} samples -> {filename}')


def write_unlabeled(names, out_root):
    filename = os.path.join(out_root, 'unlabeled.txt')
    with open(filename, 'w') as f:
        for name in names:
            f.write(name + '\n')
    print(f'Wrote unlabeled list: {len(names)} samples -> {filename}')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Kvasir-SEG dataset for training')
    parser.add_argument('--src', default='kvasir-seg/Kvasir-SEG', help='Source raw Kvasir root folder')
    parser.add_argument('--out', default='data_processed/kvasir', help='Output processed dataset root')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds to create')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for fold generation')
    parser.add_argument('--unlabeled-all', action='store_true', help='Put all samples into unlabeled.txt')
    return parser.parse_args()


def main():
    args = parse_args()
    print('Source:', args.src)
    print('Output:', args.out)
    makedirs(args.out)

    names = collect_pairs(args.src)
    write_npy_files(args.src, args.out, names)

    npy_names = [os.path.splitext(n)[0] + '.npy' for n in names]
    write_fold_files(npy_names, args.out, args.folds, args.seed)

    if args.unlabeled_all:
        write_unlabeled(npy_names, args.out)
    else:
        # by default use all samples as unlabeled too, because current scripts expect unlabeled.txt
        write_unlabeled(npy_names, args.out)

    print('Done. You can now set train_folder and unlabeled_folder to', args.out)


if __name__ == '__main__':
    main()
