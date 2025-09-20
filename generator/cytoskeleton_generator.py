import cv2
import numpy as np
import random
import os
import argparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- (All your helper functions like is_polygon_overlapping, generate_bezier_curve, etc., go here without any changes) ---
def is_polygon_overlapping(existing_polygons, new_polygon):
    for polygon in existing_polygons:
        center = tuple(new_polygon.mean(axis=0).astype(int))
        center = (int(center[0]), int(center[1]))
        dist = cv2.pointPolygonTest(polygon, center, True)
        if dist > 500:
            return True
    return False


def generate_bezier_curve(p1, p2, p3, num_points=100):
    """Generate points on a quadratic Bezier curve."""
    curve_points = []
    for t in np.linspace(0, 1, num_points):
        x = int((1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * p2[0] + t ** 2 * p3[0])
        y = int((1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * p2[1] + t ** 2 * p3[1])
        curve_points.append((x, y))
    return curve_points


def draw_random_curves_within_polygon(image, polygon_points, num_curves=10):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 1)
    h, w = image.shape[:2]
    curve_masks = []
    yolo_labels = []

    for _ in range(num_curves):
        while True:
            pts = []
            for _ in range(3):
                pt = (random.randint(0, w - 1), random.randint(0, h - 1))
                if mask[pt[1], pt[0]]:
                    pts.append(pt)
            if len(pts) == 3:
                pts = np.array(pts, dtype=np.int32)
                curve_points = generate_bezier_curve(pts[0], pts[1], pts[2])

                curve = np.zeros_like(mask)
                for pt in curve_points:
                    if 0 <= pt[0] < w and 0 <= pt[1] < h and mask[pt[1], pt[0]]:
                        curve[pt[1], pt[0]] = 255

                curve = cv2.dilate(curve, np.ones((3, 3), dtype=np.uint8), iterations=random.randint(1, 2))

                alpha = random.uniform(0.2, 1)
                color = (255, 255, 255)
                overlay = image.copy()

                for i in range(1, len(curve_points)):
                    cv2.line(overlay, tuple(curve_points[i - 1]), tuple(curve_points[i]), color, 2)

                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                if np.any(curve):
                    curve_masks.append(curve)

                    y_coords, x_coords = np.where(curve > 0)
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()

                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min

                    if bbox_w > 0 and bbox_h > 0:
                        bbox = [
                            (x_min + x_max) / 2,
                            (y_min + y_max) / 2,
                            bbox_w,
                            bbox_h,
                        ]
                        bbox_normalized = [
                            bbox[0] / w,
                            bbox[1] / h,
                            bbox[2] / w,
                            bbox[3] / h,
                        ]
                        keypoints_normalized = pts.astype(float) / np.array([w, h])
                        keypoints_with_visibility = []
                        for point in keypoints_normalized:
                            keypoints_with_visibility.extend([point[0], point[1], 1])
                        yolo_labels.append([0] + bbox_normalized + keypoints_with_visibility)

                break
    return curve_masks, yolo_labels


def draw_translucent_spheres(image, polygon_points, num_spheres_min, num_spheres_max):
    image_size = image.shape[0]
    num_spheres = random.randint(num_spheres_min, num_spheres_max)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 1)
    for _ in range(num_spheres):
        while True:
            center = (random.randint(0, image_size - 1), random.randint(0, image_size - 1))
            radius = random.randint(4, 20)
            if mask[center[1], center[0]]:
                overlay = image.copy()
                color = (0, 0, 0)
                alpha = 0.3
                cv2.circle(overlay, center, radius, color, -1)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                break


def draw_white_spheres(image, polygon_points, num_spheres_min, num_spheres_max):
    image_size = image.shape[0]
    num_spheres = random.randint(num_spheres_min, num_spheres_max)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 1)
    for _ in range(num_spheres):
        while True:
            center = (random.randint(0, image_size - 1), random.randint(0, image_size - 1))
            radius = random.randint(4, 20)
            if mask[center[1], center[0]]:
                overlay = image.copy()
                color = (255, 255, 255)
                alpha = random.random() * 0.5
                cv2.circle(overlay, center, radius, color, -1)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                break


def apply_local_blur(image, polygon_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 4)
    blurred_image_with_mask = np.repeat(mask[:, :, None], 3, axis=2)
    np.copyto(image, blurred_image, where=blurred_image_with_mask > 0)


def draw_polygon_with_jitter(image, center, radius, num_sides, alpha):
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    points = np.array([
        (int(center[0] + (radius + random.uniform(-radius * 0.3, radius * 0.3)) * np.cos(angle)),
         int(center[1] + (radius + random.uniform(-radius * 0.3, radius * 0.3)) * np.sin(angle)))
        for angle in angles
    ], dtype=np.int32)

    edge_blurred_image = image.copy()
    cv2.fillPoly(edge_blurred_image, [points],
                 (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200)))
    edge_blurred_image = cv2.GaussianBlur(edge_blurred_image, (21, 21), 5)
    cv2.addWeighted(edge_blurred_image, alpha, image, 1 - alpha, 0, image)
    return points


def generate_single_image(num_curves, noise_intensity, num_spheres_min, num_spheres_max):
    """Generates a single synthetic cell image with specified parameters."""
    image_size = 2048

    background_base_range = (0.3, 0.5)
    background_center_bias_std = image_size // 6
    halo_sigma = 50

    y, x = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((x - image_size // 2) ** 2 + (y - image_size // 2) ** 2)
    halo = np.exp(-distance / (image_size // 2)) * np.random.uniform(0.3, 0.5)

    gradient = np.linspace(0, np.random.uniform(*background_base_range), image_size)
    background = np.tile(gradient, (image_size, 1))
    background += gaussian_filter(halo, sigma=halo_sigma)
    noise = np.random.normal(0, noise_intensity, (image_size, image_size))
    background += noise
    background = np.clip(background, 0, 1)

    image = (background * 100).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    existing_polygons = []
    all_curve_masks = []
    all_yolo_labels = []

    # Generate one cell per image
    while True:
        center = (random.randint(400, image_size - 400), random.randint(400, image_size - 400))
        radius = random.randint(300, 500)
        num_sides = random.randint(14, 18)
        overlay_image = image.copy()
        polygon_points = draw_polygon_with_jitter(overlay_image, center, radius, num_sides, random.random() * 0.5)

        if not is_polygon_overlapping(existing_polygons, polygon_points):
            existing_polygons.append(polygon_points)
            curve_masks, yolo_labels = draw_random_curves_within_polygon(overlay_image, polygon_points, num_curves=num_curves)

            if curve_masks:
                all_curve_masks.extend(curve_masks)
                all_yolo_labels.extend(yolo_labels)
                draw_translucent_spheres(overlay_image, polygon_points, num_spheres_min, num_spheres_max)
                draw_white_spheres(overlay_image, polygon_points, num_spheres_min, num_spheres_max)
                apply_local_blur(overlay_image, polygon_points)
                cv2.addWeighted(overlay_image, 0.8, image, 0.2, 0, image)
            break
    
    return image, all_curve_masks, all_yolo_labels

def generate_dataset(output_dir, num_images, num_curves, noise_intensity, num_spheres_min, num_spheres_max, output_size):
    """
    Main function to generate and save a dataset.
    """
    print(f"--- Generating dataset in '{output_dir}' ---")
    print(f"Parameters: num_images={num_images}, num_curves={num_curves}, noise_intensity={noise_intensity}")
    print(f"num_spheres=[{num_spheres_min}-{num_spheres_max}], output_size={output_size}x{output_size}")

    img_dir = os.path.join(output_dir, 'img')
    mask_dir = os.path.join(output_dir, 'mask')
    label_dir = os.path.join(output_dir, 'label')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc="Generating Images"):
        final_image, curve_masks, yolo_labels = generate_single_image(
            num_curves=num_curves,
            noise_intensity=noise_intensity,
            num_spheres_min=num_spheres_min,
            num_spheres_max=num_spheres_max
        )

        if not curve_masks:
            print(f"Warning: Skipping image {i} due to no filaments being generated.")
            continue

        combined_mask = np.zeros((2048, 2048), dtype=np.uint8)
        for mask in curve_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        combined_mask = np.where(combined_mask > 0, 1, 0).astype('uint8')

        final_image_resized = cv2.resize(final_image, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        combined_mask_resized = cv2.resize(combined_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(img_dir, f'{i}.jpg'), final_image_resized)
        np.save(os.path.join(mask_dir, f'{i}.npy'), combined_mask_resized)

        with open(os.path.join(label_dir, f'{i}.txt'), 'w') as f:
            for label in yolo_labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cell images for deep learning.")

    # --- Core Arguments ---
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save the dataset.')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to generate.')
    
    # --- Arguments from SNR/Density Levels ---
    parser.add_argument('--num_curves', type=int, default=24, help='Number of filaments (curves) to generate in each cell. Controls density.')
    parser.add_argument('--noise_intensity', type=float, default=0.3, help='Standard deviation of Gaussian noise. Controls SNR (higher value = lower SNR).')

    # --- Other Exposed Parameters ---
    parser.add_argument('--num_spheres_min', type=int, default=30, help='Minimum number of black/white spheres (texture).')
    parser.add_argument('--num_spheres_max', type=int, default=80, help='Maximum number of black/white spheres (texture).')
    parser.add_argument('--output_size', type=int, default=1024, help='The final size (width and height) of the output images.')
    
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        num_curves=args.num_curves,
        noise_intensity=args.noise_intensity,
        num_spheres_min=args.num_spheres_min,
        num_spheres_max=args.num_spheres_max,
        output_size=args.output_size
    )