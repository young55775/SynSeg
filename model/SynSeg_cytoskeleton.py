import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import time
from tqdm import tqdm
import tifffile

class UNet(nn.Module):
    # --- UNet类定义保持不变 ---
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
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
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


def tiling_inference(model, image, tile_size, overlap, device):
    # --- tiling_inference 函数保持不变 ---
    h, w = image.shape
    stride = tile_size - overlap
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = padded_image.shape
    prediction_mask = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum_mask = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_1d = np.concatenate([np.linspace(0, 1, tile_size // 2), np.linspace(1, 0, tile_size - tile_size // 2)])
    weight_window = np.outer(weight_1d, weight_1d)
    print("Running tiling inference with edge blending...")
    for y in tqdm(range(0, padded_h - overlap, stride)):
        for x in range(0, padded_w - overlap, stride):
            tile = padded_image[y:y + tile_size, x:x + tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                tile_prediction = model(tile_tensor).cpu().numpy().squeeze()
            prediction_mask[y:y + tile_size, x:x + tile_size] += tile_prediction * weight_window
            weight_sum_mask[y:y + tile_size, x:x + tile_size] += weight_window
    weight_sum_mask[weight_sum_mask == 0] = 1
    final_mask = prediction_mask / weight_sum_mask
    return final_mask[:h, :w]


def run_inference(args):
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model_path}")
    model = UNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading image from: {args.input_path}")
    image = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not open image at: {args.input_path}")
    
    if image.dtype != np.uint8:
        print(f"Image is not 8-bit ({image.dtype}). Converting to 8-bit.")
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
    image_float = image.astype(np.float32)
    h, w = image_float.shape
    start_time = time.time()
    
    # ===== 最终版推理逻辑 =====
    if args.use_tiling:
        # --- 优先级1: 用户强制使用分片推理 ---
        final_probability_map = tiling_inference(model, image_float, args.tile_size, args.overlap, device)
    
    elif args.resize_size is not None:
        # --- 优先级2: 用户强制使用固定尺寸缩放 ---
        print(f"User override: Resizing input to fixed size ({args.resize_size}x{args.resize_size}).")
        image_resized = cv2.resize(image_float, (args.resize_size, args.resize_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output_mask = model(input_tensor)
        mask_np = output_mask.detach().cpu().squeeze().numpy()
        final_probability_map = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_CUBIC)

    else:
        # --- 优先级3: 默认使用智能自动缩放 ---
        downsample_factor = 8
        is_poolable = (h % downsample_factor == 0) and (w % downsample_factor == 0)
        
        if is_poolable:
            print(f"Default mode: Image size ({h}x{w}) is suitable. Running direct inference.")
            input_image_for_model = image_float
        else:
            new_h = max(downsample_factor, round(h / downsample_factor) * downsample_factor)
            new_w = max(downsample_factor, round(w / downsample_factor) * downsample_factor)
            print(f"Default mode: Image size ({h}x{w}) is not divisible by {downsample_factor}. Auto-resizing to nearest workable size: ({new_h}x{new_w}).")
            input_image_for_model = cv2.resize(image_float, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        input_tensor = torch.from_numpy(input_image_for_model).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output_mask = model(input_tensor)
        
        mask_np = output_mask.detach().cpu().squeeze().numpy()
        
        if not is_poolable:
            final_probability_map = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            final_probability_map = mask_np

    end_time = time.time()
    print(f"Inference complete in {end_time - start_time:.2f} seconds.")

    # --- 保存逻辑保持不变 ---
    if args.threshold is not None:
        print(f"Applying threshold: {args.threshold}")
        final_mask = (final_probability_map > args.threshold).astype(np.uint8) * 255
    else:
        print("Saving raw probability map (32-bit float).")
        final_mask = final_probability_map.astype(np.float32)

    try:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        tifffile.imwrite(args.output_path, final_mask)
        print(f"Mask successfully saved to: {args.output_path}")
    except Exception as e:
        print(f"Error saving with tifffile: {e}. Please ensure tifffile is installed: pip install tifffile")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Inference Script with multiple modes: auto-resize (default), manual resize, or tiling.")
    
    # --- 通用参数 ---
    common_args = parser.add_argument_group('Common Arguments')
    common_args.add_argument('-m', '--model_path', type=str, required=True, help='Path to the trained U-Net model weights (.pth file).')
    common_args.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image file.')
    common_args.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output mask file.')
    common_args.add_argument('--threshold', type=float, default=None, help='Optional. Apply a threshold to create a binary mask (0.0 to 1.0). If not set, saves a 32-bit float probability map.')
    common_args.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Device to run on. Default: auto.")

    # --- 推理模式参数 (按优先级排序) ---
    mode_args = parser.add_argument_group('Inference Mode (Priority: Tiling > Manual Resize > Auto-Resize)')
    mode_args.add_argument('--use_tiling', action='store_true', help='(Priority 1) Force tiling inference. Best for large, non-distortable images.')
    mode_args.add_argument('--resize_size', type=int, default=None, help='(Priority 2) Force resize to a fixed size (e.g., 1024)')
    mode_args.add_argument('--tile_size', type=int, default=1024, help='[Tiling Mode] The size of square tiles. Used only if --use_tiling is specified.')
    mode_args.add_argument('--overlap', type=int, default=128, help='[Tiling Mode] Overlap between tiles. Used only if --use_tiling is specified.')

    args = parser.parse_args()
    
    if args.use_tiling and args.tile_size % 8 != 0:
        raise ValueError("--tile_size must be divisible by 8 for the U-Net architecture.")
    if args.resize_size is not None and args.resize_size % 8 != 0:
        raise ValueError("--resize_size must be divisible by 8 for the U-Net architecture.")
        
    run_inference(args)