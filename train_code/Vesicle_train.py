import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch.nn.functional as F
import time
import os
import random
from tqdm import tqdm
import argparse

class UNet(nn.Module):
    """The U-Net model for image segmentation."""
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

class VesicleDataset(Dataset):
    """Dataset class for Vesicle Segmentation with wide-range intensity scaling."""
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_files = sorted(os.listdir(self.img_path))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_base = os.path.splitext(img_file)[0]
        mask_file = os.path.join(self.mask_path, img_base + '.npy')
        
        mask = np.load(mask_file)
        mask = mask.astype('float32')

        img = cv2.imread(os.path.join(self.img_path, img_file), cv2.IMREAD_GRAYSCALE).astype('float32')
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {os.path.join(self.img_path, img_file)}")
        
        if random.random() < 0.8:
            scale_factor = 1 + random.random() * 65000
            img *= scale_factor
        else:
            img = img**2
            img = cv2.normalize(img, None, 0, 65000, cv2.NORM_MINMAX)
        
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

def train_model(args):
    """Main function to train the U-Net model."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = VesicleDataset(args.train_img_dir, args.train_mask_dir)
    val_set = VesicleDataset(args.val_img_dir, args.val_mask_dir)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = UNet(input_channels=1, output_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print("----------- Start Training for Vesicle Segmentation -----------")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, masks in train_pbar:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
        epoch_train_loss = running_train_loss / len(train_set)

        model.eval()
        running_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for inputs, masks in val_pbar:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item() * inputs.size(0)
                val_pbar.set_postfix({'loss': loss.item()})
        epoch_val_loss = running_val_loss / len(val_set)
        
        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = os.path.join(args.output_dir, 'best_vesicle_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with validation loss: {best_val_loss:.4f}")

    end_time = time.time()
    print("----------- Training Finished -----------")
    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="U-Net Model Training for Vesicle Segmentation")
    parser.add_argument('--train-img-dir', type=str, required=True, help='Path to the training images directory.')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='Path to the training masks directory (.npy files).')
    parser.add_argument('--val-img-dir', type=str, required=True, help='Path to the validation images directory.')
    parser.add_argument('--val-mask-dir', type=str, required=True, help='Path to the validation masks directory (.npy files).')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda", "cuda:0", "cpu").')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    args = parser.parse_args()
    train_model(args)
