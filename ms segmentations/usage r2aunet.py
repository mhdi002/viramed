import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.nn.init as init

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightweightR2AUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightR2AUNet, self).__init__()
        features = [16, 32, 64, 128]
        
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for i in range(4):
            if i == 0:
                self.encoder.append(self._make_layer(in_channels, features[i]))
            else:
                self.encoder.append(self._make_layer(features[i-1], features[i]))
        
        self.bridge = self._make_layer(features[-1], features[-1]*2)
        
        self.decoder = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for i in range(4):
            self.up_conv.append(nn.ConvTranspose3d(features[-i-1]*2, features[-i-1], kernel_size=2, stride=2))
            self.decoder.append(self._make_layer(features[-i-1]*2, features[-i-1]))
        
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_features = []
        for enc in self.encoder:
            x = enc(x)
            encoder_features.append(x)
            x = self.pool(x)
        
        x = self.bridge(x)
        
        for i in range(4):
            x = self.up_conv[i](x)
            encoder_feature = encoder_features[-i-1]
            if x.shape[2:] != encoder_feature.shape[2:]:
                x = F.interpolate(x, size=encoder_feature.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder[i](x)
        
        x = self.final_conv(x)
        return x

def normalize_flair(flair_data):
    return (flair_data - flair_data.min()) / (flair_data.max() - flair_data.min())

def pad_to_multiple_of_16(data):
    pad_size = [(0, (16 - s % 16) % 16) for s in data.shape]
    return np.pad(data, pad_size, mode='constant')

class MRIDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        flair_path = self.file_paths[idx]
        flair_nifti = nib.load(flair_path)
        flair_data = flair_nifti.get_fdata().astype(np.float32)
        original_shape = flair_data.shape
        flair_data = normalize_flair(flair_data)
        flair_data = pad_to_multiple_of_16(flair_data)
        flair_tensor = torch.from_numpy(flair_data).float().unsqueeze(0)
        return flair_tensor, flair_nifti.affine, flair_path, torch.tensor(original_shape)

def get_file_paths(input_dir):
    file_paths = []
    for file in os.listdir(input_dir):
        if file.endswith("_flair.nii") or file.endswith("_flair.nii.gz"):
            file_paths.append(os.path.join(input_dir, file))
    return file_paths

def predict_mask(model, input_tensor, device):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
    return mask

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set paths
    model_path = r"C:\Users\mahdi\Downloads\best_model.pth"
    input_dir = r"C:\Users\mahdi\Downloads\New folder (4)"
    output_dir = r"C:\Users\mahdi\Downloads\New folder (5)"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = LightweightR2AUNet(in_channels=1, out_channels=5).to(device)  # 5 classes (4 + background)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    # Get file paths
    file_paths = get_file_paths(input_dir)
    dataset = MRIDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process each MRI image
    for flair_tensor, affine, flair_path, original_shape in dataloader:
        # Get the filename
        filename = os.path.basename(flair_path[0])
        logger.info(f"Processing {filename}")

        # Predict mask
        mask = predict_mask(model, flair_tensor, device)

        # Crop the mask back to the original size
        original_shape = original_shape.squeeze().tolist()
        logger.info(f"Original shape: {original_shape}")
        mask = mask[:original_shape[0], :original_shape[1], :original_shape[2]]

        # Load original NIfTI to get header
        original_nifti = nib.load(flair_path[0])

        # Create a new NIfTI image for the mask
        mask_nifti = nib.Nifti1Image(mask.astype(np.uint8), affine[0].numpy(), original_nifti.header)
        mask_nifti.header.set_data_dtype(np.uint8)  # Ensure the datatype is uint8

        # Save the mask
        output_filename = filename.replace("_flair", "_mask")
        output_path = os.path.join(output_dir, output_filename)
        nib.save(mask_nifti, output_path)
        logger.info(f"Mask saved to {output_path}")

        # Log class distribution
        unique, counts = np.unique(mask, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution in the mask: {class_distribution}")

    logger.info("All images processed successfully")

if __name__ == "__main__":
    main()