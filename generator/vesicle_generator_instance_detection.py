import cv2
import numpy as np
import random
import os
import argparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def generate_single_image_yolo(args, cluster_scale):
    """Generates a single synthetic image and YOLO labels based on provided arguments."""
    
    # Background Generation
    background_base_range = (0.1, random.random() * 0.2 + 0.1)
    halo_sigma = random.random() * 20 + 50

    gradient = np.linspace(0, np.random.uniform(*background_base_range), args.gen_size)
    background = np.tile(gradient, (args.gen_size, 1))

    bg_center_x = int(np.random.normal(args.gen_size // 2, cluster_scale))
    bg_center_y = int(np.random.normal(args.gen_size // 2, cluster_scale))
    bg_center_x = np.clip(bg_center_x, 0, args.gen_size - 1)
    bg_center_y = np.clip(bg_center_y, 0, args.gen_size - 1)
    
    y, x = np.ogrid[:args.gen_size, :args.gen_size]
    distance = np.sqrt((x - bg_center_x)**2 + (y - bg_center_y)**2)
    halo = np.exp(-distance / (args.gen_size // 2)) * np.random.uniform(0.3, 0.7)
    background += gaussian_filter(halo, sigma=halo_sigma)

    noise = np.random.normal(0, args.noise_intensity, (args.gen_size, args.gen_size))
    background += noise
    background = np.clip(background, 0, 1)

    image = background.copy()

    # Spot Generation
    num_spots = random.randint(args.num_spots_min, args.num_spots_max)
    yolo_label_lines = []
    existing_spots = []
    attempts = 0
    max_attempts = num_spots * 20

    while len(existing_spots) < num_spots and attempts < max_attempts:
        attempts += 1
        center_x = int(np.random.normal(loc=args.gen_size // 2, scale=cluster_scale))
        center_y = int(np.random.normal(loc=args.gen_size // 2, scale=cluster_scale))
        center_x = np.clip(center_x, 0, args.gen_size - 1)
        center_y = np.clip(center_y, 0, args.gen_size - 1)
        radius = np.random.randint(args.min_radius, args.max_radius)

        too_close = False
        for (ex, ey, er) in existing_spots:
            dist = np.sqrt((center_x - ex)**2 + (center_y - ey)**2)
            if dist < (radius + er) * args.overlap_factor:
                too_close = True
                break
        if too_close:
            continue

        existing_spots.append((center_x, center_y, radius))

        brightness = np.random.uniform(args.min_brightness, args.max_brightness)
        y_grid, x_grid = np.ogrid[:args.gen_size, :args.gen_size]
        mask_circle = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2

        spot = np.zeros_like(image)
        spot[mask_circle] = brightness
        blur_sigma = np.random.uniform(args.spot_blur_min, args.spot_blur_max)
        spot = gaussian_filter(spot, sigma=blur_sigma)
        image += spot

        x_center = center_x / args.gen_size
        y_center = center_y / args.gen_size
        box_w = 2 * radius / args.gen_size
        box_h = 2 * radius / args.gen_size
        yolo_label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    
    return image, yolo_label_lines

def generate_dataset_yolo(args):
    """Main function to generate and save a YOLO dataset."""
    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = os.path.join(args.output_dir, 'images')
    label_dir = os.path.join(args.output_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"--- Generating YOLO dataset in '{args.output_dir}' ---")
    print(f"Parameters: num_images={args.num_images}, gen_size={args.gen_size}, out_size={args.out_size}")

    cluster_scale = args.gen_size // args.cluster_scale_divisor

    for i in tqdm(range(args.num_images), desc="Generating Images"):
        image, yolo_labels = generate_single_image_yolo(args, cluster_scale)
        
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8) # Scale to 8-bit for JPG
        image_resized = cv2.resize(image, (args.out_size, args.out_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(img_dir, f'{i}.jpg'), image_resized)
        with open(os.path.join(label_dir, f'{i}.txt'), 'w') as f:
            for line in yolo_labels:
                f.write(line + "\n")
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic vesicle images and YOLO labels.")

    # --- Core Arguments ---
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save the dataset.')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to generate.')
    parser.add_argument('--gen_size', type=int, default=512, help='Initial generation resolution.')
    parser.add_argument('--out_size', type=int, default=256, help='Final output image resolution.')

    # --- Vesicle Parameters ---
    parser.add_argument('--num_spots_min', type=int, default=300, help='Minimum number of vesicles per image.')
    parser.add_argument('--num_spots_max', type=int, default=500, help='Maximum number of vesicles per image.')
    parser.add_argument('--min_radius', type=int, default=4, help='Minimum radius of vesicles in pixels.')
    parser.add_argument('--max_radius', type=int, default=25, help='Maximum radius of vesicles in pixels.')
    parser.add_argument('--min_brightness', type=float, default=0.3, help='Minimum brightness of vesicles.')
    parser.add_argument('--max_brightness', type=float, default=0.9, help='Maximum brightness of vesicles.')
    parser.add_argument('--spot_blur_min', type=float, default=2.0, help='Minimum sigma for vesicle Gaussian blur.')
    parser.add_argument('--spot_blur_max', type=float, default=4.0, help='Maximum sigma for vesicle Gaussian blur.')
    
    # --- Background & Distribution Parameters ---
    parser.add_argument('--noise_intensity', type=float, default=0.08, help='Standard deviation of Gaussian noise.')
    parser.add_argument('--cluster_scale_divisor', type=int, default=6, help='Controls how tightly vesicles cluster to the center (smaller value = tighter cluster).')
    parser.add_argument('--overlap_factor', type=float, default=1.0, help='Controls how much vesicles can overlap. <1 allows overlap, 1 prevents touching, >1 creates space.')

    args = parser.parse_args()
    generate_dataset_yolo(args)