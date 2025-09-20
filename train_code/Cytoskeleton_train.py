import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from livelossplot import PlotLosses
import matplotlib.pyplot as plt

# --- Model and Dataset classes remain the same ---

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

class MaskSet(Dataset):
    def __init__(self, data_dir):
        self.img_path = os.path.join(data_dir, 'img')
        self.mask_path = os.path.join(data_dir, 'mask')
        self.img_files = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_base = os.path.splitext(os.path.basename(img_file))[0]
        mask_file = os.path.join(self.mask_path, img_base + '.npy')
        
        mask = np.load(mask_file).astype('float32')
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
        
        img = cv2.imread(os.path.join(self.img_path, img_file), cv2.IMREAD_UNCHANGED).astype('float32')
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        scale_factor = 0.25 + random.random() * 0.75
        img *= scale_factor
        
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

def main(args):
    # --- Setup ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Data Loading ---
    print("Loading datasets...")
    train_set = MaskSet(data_dir=args.train_dir)
    val_set = MaskSet(data_dir=args.val_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")

    # --- Model, Loss, Optimizer ---
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    liveloss = PlotLosses()
    
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        logs = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logs['loss'] = avg_train_loss
        logs['val_loss'] = avg_val_loss
        liveloss.update(logs)
        liveloss.send()
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saved best model to: {model_save_path}")

    # --- Final Visualization ---
    if not args.no_visualize:
        print("Displaying sample prediction from the validation set...")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        model.eval()
        with torch.no_grad():
            sample_image, sample_mask = val_set[0]
            sample_image_dev = sample_image.to(device).unsqueeze(0)
            pred_mask = model(sample_image_dev).squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(sample_mask.squeeze().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Training Script for Image Segmentation.")
    
    # --- Path Arguments ---
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory (should contain img/ and mask/ subfolders).')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory (should contain img/ and mask/ subfolders).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the best model weights.')

    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the Adam optimizer.')

    # --- Execution Arguments ---
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Device to run on. 'auto' uses GPU if available.")
    parser.add_argument('--no_visualize', action='store_true', help='Disable the final visualization plot.')
    
    args = parser.parse_args()
    main(args)