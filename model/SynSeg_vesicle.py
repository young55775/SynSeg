import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.up3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return torch.sigmoid(self.final(dec1))


def predict_large_image(model, image, tile_size=512, overlap=64, device='cpu'):
    model.to(device)
    model.eval()
    h, w = image.shape
    stride = tile_size - overlap
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = padded_image.shape
    prediction_mask = np.zeros((padded_h, padded_w), dtype=np.float32)
    sum_mask = np.zeros((padded_h, padded_w), dtype=np.float32)

    print("Processing image in tiles...")
    for y in tqdm(range(0, padded_h - overlap, stride)):
        for x in range(0, padded_w - overlap, stride):
            tile = padded_image[y:y + tile_size, x:x + tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                tile_prediction = model(tile_tensor)
            tile_prediction_np = tile_prediction.cpu().numpy().squeeze()
            prediction_mask[y:y + tile_size, x:x + tile_size] += tile_prediction_np
            sum_mask[y:y + tile_size, x:x + tile_size] += 1
    
    sum_mask[sum_mask == 0] = 1
    final_mask = prediction_mask / sum_mask
    return final_mask[:h, :w]


def main(args):
    # --- Set Device ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from: {args.model_path}")
    model = UNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully.")

    # --- Load and Pre-process Image ---
    print(f"Loading image from: {args.input_path}")
    img_original = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise FileNotFoundError(f"Could not open image at path: {args.input_path}")
    
    img = img_original.astype(np.float32)

    # --- Apply automatic brightness normalization ---
    if args.median_norm is not None:
        print(f"Normalizing median brightness to target: {args.median_norm}")
        img_median = np.median(img[img > 0])
        if img_median > 0:
            bg_cut = args.median_norm / img_median
            img *= bg_cut

    # --- Run Inference ---
    final_prediction_mask = predict_large_image(model, img, tile_size=args.tile_size, overlap=args.overlap, device=device)

    # --- Post-process and Save ---
    if args.threshold is not None:
        print(f"Applying threshold {args.threshold} to create binary mask.")
        output_mask = (final_prediction_mask > args.threshold).astype(np.uint8) * 255
    else:
        print("Saving raw probability map as 16-bit TIFF.")
        output_mask = (final_prediction_mask * 65535).astype(np.uint16)
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, output_mask)
    print(f"Mask successfully saved to: {args.output_path}")

    # --- Visualize the Results (Optional) ---
    if not args.no_show:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_original, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(output_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a large image using a U-Net with tiling inference.")
    
    # --- Required Arguments ---
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the trained U-Net model weights (.pth).')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output mask.')

    # --- Tiling Parameters ---
    parser.add_argument('--tile_size', type=int, default=512, help='Size of the square tiles for inference. Default: 512.')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between tiles in pixels. Default: 64.')

    # --- Pre- and Post-processing Parameters ---
    parser.add_argument('--median_norm', type=float, default=500.0, help='Target median intensity for brightness normalization. Set to None to disable.')
    parser.add_argument('--threshold', type=float, default=None, help='Optional threshold (0.0-1.0) to create a binary mask. If not set, saves a 16-bit probability map.')

    # --- Execution Parameters ---
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to run on. "auto" uses GPU if available.')
    parser.add_argument('--no_show', action='store_true', help='Do not display the result with matplotlib.')

    args = parser.parse_args()
    main(args)