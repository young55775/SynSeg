import cv2
import numpy as np
import random
import os
import argparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def generate_single_image(
    image_size,
    min_brightness, max_brightness,
    min_radius, max_radius,
    num_spots_min, num_spots_max,
    noise_intensity
):
    """Generates a single synthetic vesicle image and its corresponding mask."""
    
    # Background Generation
    background_base_range = (0.3, random.random() * 0.5 + 0.5)
    background_center_bias_std = image_size // 6
    halo_sigma = random.random() * 20 + 50
    
    gradient = np.linspace(0, np.random.uniform(*background_base_range), image_size)
    background = np.tile(gradient, (image_size, 1))
    
    bg_center_x = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_y = int(np.random.normal(image_size // 2, background_center_bias_std))
    bg_center_x = np.clip(bg_center_x, 0, image_size - 1)
    bg_center_y = np.clip(bg_center_y, 0, image_size - 1)
    
    y, x = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((x - bg_center_x)**2 + (y - bg_center_y)**2)
    halo = np.exp(-distance / (image_size // 2)) * np.random.uniform(0.3, 0.7)
    background += gaussian_filter(halo, sigma=halo_sigma)
    
    noise = np.random.normal(0, noise_intensity, (image_size, image_size))
    background += noise
    background = np.clip(background, 0, 1)

    # Image and Mask Initialization
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    image = background.copy()
    
    # Vesicle (Spot) Generation
    num_spots = random.randint(num_spots_min, num_spots_max)
    for _ in range(num_spots):
        center_x = np.random.randint(0, image_size)
        center_y = np.random.randint(0, image_size)
        radius = np.random.uniform(min_radius, max_radius) # Use uniform for float radii
        brightness = np.random.uniform(min_brightness, max_brightness)
        
        y_grid, x_grid = np.ogrid[:image_size, :image_size]
        mask_circle = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2
        
        spot = np.zeros_like(image)
        spot[mask_circle] = brightness
        blur_sigma = np.random.uniform(0.2, 1.3)
        spot = gaussian_filter(spot, sigma=blur_sigma)
        
        image += spot
        mask[mask_circle] = 1

    image = np.clip(image, 0, 1)
    return image, mask

def generate_dataset(args):
    """Main function to generate and save a dataset based on provided arguments."""
    
    print(f"--- Generating dataset in '{args.output_dir}' ---")
    print(f"Parameters: num_images={args.num_images}, image_size={args.image_size}")

    img_dir = os.path.join(args.output_dir, 'img')
    mask_dir = os.path.join(args.output_dir, 'mask')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in tqdm(range(args.num_images), desc="Generating Images"):
        image, mask = generate_single_image(
            image_size=args.image_size,
            min_brightness=args.min_brightness, max_brightness=args.max_brightness,
            min_radius=args.min_radius, max_radius=args.max_radius,
            num_spots_min=args.num_spots_min, num_spots_max=args.num_spots_max,
            noise_intensity=args.noise_intensity
        )
        
        # Save image as 16-bit TIFF for better dynamic range
        cv2.imwrite(os.path.join(img_dir, f'{i}.tif'), (image * 65535).astype(np.uint16))
        np.save(os.path.join(mask_dir, f'{i}.npy'), mask)

    print("\nDataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic vesicle images for deep learning.")

    # --- Core Arguments ---
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save the dataset.')
    parser.add_argument('--num_images', type=int, default=200, help='Number of images to generate.')
    parser.add_argument('--image_size', type=int, default=512, help='The size (width and height) of the output images.')

    # --- Vesicle Parameters ---
    parser.add_argument('--num_spots_min', type=int, default=300, help='Minimum number of vesicles per image (density control).')
    parser.add_argument('--num_spots_max', type=int, default=500, help='Maximum number of vesicles per image (density control).')
    parser.add_argument('--min_radius', type=float, default=1.0, help='Minimum radius of vesicles in pixels.')
    parser.add_argument('--max_radius', type=float, default=13.0, help='Maximum radius of vesicles in pixels.')
    parser.add_argument('--min_brightness', type=float, default=0.03, help='Minimum brightness of vesicles.')
    parser.add_argument('--max_brightness', type=float, default=0.7, help='Maximum brightness of vesicles.')

    # --- Background & Noise Parameters ---
    parser.add_argument('--noise_intensity', type=float, default=0.15, help='Standard deviation of Gaussian noise (higher value = lower SNR).')
    
    
    args = parser.parse_args()
    generate_dataset(args)