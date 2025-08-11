import sys, json, argparse, os, io, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

# Optional dependencies for medical imaging
try:
    import nibabel as nib
except Exception:
    nib = None

# -------------------------
# Retina 2D R2UNet (from your eye retina/r2unet.py)
# -------------------------
class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, 1, 1, 0)
    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class R2UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, t=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.encoder = nn.ModuleList([RRCNN_block(in_channels, features[0], t)])
        for i in range(1, len(features)):
            self.encoder.append(RRCNN_block(features[i-1], features[i], t))
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.ConvTranspose2d(features[-1], features[-2], 2, 2))
        self.decoder.append(RRCNN_block(features[-1], features[-2], t))
        for i in range(len(features)-2, 0, -1):
            self.decoder.append(nn.ConvTranspose2d(features[i], features[i-1], 2, 2))
            self.decoder.append(RRCNN_block(features[i], features[i-1], t))
        self.Conv_1x1 = nn.Conv2d(features[0], out_channels, 1, 1, 0)
    def forward(self, x):
        skips = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skips.append(x)
            if i < len(self.encoder) - 1:
                x = self.Maxpool(x)
        skips = skips[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            if x.shape != skips[i//2 + 1].shape:
                diff_y = skips[i//2 + 1].size()[2] - x.size()[2]
                diff_x = skips[i//2 + 1].size()[3] - x.size()[3]
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skips[i//2 + 1], x], dim=1)
            x = self.decoder[i+1](x)
        return self.Conv_1x1(x)

# -------------------------
# LightweightR2AUNet (3D) from your ms segmentations
# -------------------------
class LightweightR2AUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        features = [16, 32, 64, 128]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)
        for i in range(4):
            if i == 0:
                self.encoder.append(self._make_layer(in_channels, features[i]))
            else:
                self.encoder.append(self._make_layer(features[i-1], features[i]))
        self.bridge = self._make_layer(features[-1], features[-1]*2)
        self.decoder = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for i in range(4):
            self.up_conv.append(nn.ConvTranspose3d(features[-i-1]*2, features[-i-1], 2, 2))
            self.decoder.append(self._make_layer(features[-i-1]*2, features[-i-1]))
        self.final_conv = nn.Conv3d(features[0], out_channels, 1)
        self.apply(self._init_weights)
    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        enc_feats = []
        for enc in self.encoder:
            x = enc(x)
            enc_feats.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for i in range(4):
            x = self.up_conv[i](x)
            enc_feat = enc_feats[-i-1]
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i](x)
        return self.final_conv(x)

# -------------------------
# Bone age regressor (from your GUI code)
# -------------------------
class BoneAgeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError(f"timm is required: {e}")
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x.squeeze(1)

# -------------------------
# Helpers
# -------------------------

def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def pad_to_multiple_of_16_3d(data: np.ndarray) -> np.ndarray:
    pad_size = [(0, (16 - s % 16) % 16) for s in data.shape]
    return np.pad(data, pad_size, mode='constant')


def pil_to_data_url(img: Image.Image):
    import base64
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'

# -------------------------
# Tasks
# -------------------------

def run_ms_segmentation(model_path: str, input_path: str, return_image: bool):
    if nib is None:
        return {"error": "nibabel not installed. Install nibabel for NIfTI."}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(np.float32)
    original_shape = data.shape
    data = normalize_minmax(data)
    data = pad_to_multiple_of_16_3d(data)
    data_chw = np.expand_dims(np.transpose(data, (2, 0, 1)), 0)  # 1 x D x H x W
    x = np.expand_dims(data_chw, 0)  # 1 x 1 x D x H x W
    x = torch.from_numpy(x).float().to(device)

    model = LightweightR2AUNet(in_channels=1, out_channels=5).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()  # D x H x W

    pred = pred[:original_shape[2], :original_shape[0], :original_shape[1]]

    image_data = None
    if return_image and Image is not None:
        pred_hwd = np.transpose(pred, (1, 2, 0))
        z = pred_hwd.shape[2] // 2
        slice2d = (normalize_minmax(pred_hwd[:, :, z]) * 255).astype(np.uint8)
        image_data = pil_to_data_url(Image.fromarray(slice2d))

    return {"boxes": [], "image": image_data, "meta": {"task": "segment3d", "classes": 5, "shape": list(pred.shape)}}


def run_retina_segmentation(model_path: str, input_path: str, img_size: int, return_image: bool):
    if Image is None:
        return {"error": "Pillow not installed."}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    img_resized = img.resize((img_size, img_size))
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # simple norm
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().to(device)

    model = R2UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)
        mask = (prob > 0.5).float().squeeze().cpu().numpy()  # HxW (img_size,img_size)

    # Resize mask back to original size
    mask_img = Image.fromarray((mask*255).astype(np.uint8)).resize((w, h))
    image_data = pil_to_data_url(mask_img) if return_image else None

    return {"boxes": [], "image": image_data, "meta": {"task": "segment2d", "shape": [h, w]}}


def run_bone_age(model_path: str, input_path: str, img_size: int):
    if Image is None:
        return {"error": "Pillow not installed."}
    device = torch.device('cpu')  # GUI example uses CPU
    # Compose ensemble: if model_path suggests fold, load all best_model_fold*.pth in same dir
    paths = []
    base_dir = os.path.dirname(model_path)
    if 'best_model_fold' in os.path.basename(model_path):
        paths = sorted(glob.glob(os.path.join(base_dir, 'best_model_fold*.pth')))
        if not paths:
            paths = [model_path]
    else:
        paths = [model_path]

    # Preprocess
    img = Image.open(input_path).convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    arr = (arr - mean) / std
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().to(device)

    # Load models
    models = []
    for p in paths:
        m = BoneAgeRegressor().to(device)
        state = torch.load(p, map_location=device)
        m.load_state_dict(state, strict=False)
        m.eval()
        models.append(m)

    preds = []
    with torch.no_grad():
        for m in models:
            v = float(m(x).item())
            preds.append(v)
    final_pred = float(np.mean(preds)) if preds else 0.0
    final_pred = float(np.clip(final_pred, 0, 240))

    return {"boxes": [], "image": None, "meta": {"task": "regression", "bone_age_months": final_pred, "ensemble": len(models)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to .pth')
    ap.add_argument('--task', default='segment3d', help='segment3d|retina|boneage')
    ap.add_argument('--input', help='Path to input file (.nii/.nii.gz or image)')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--return_image', default='true')
    args = ap.parse_args()

    ret_img = str(args.return_image).lower() in ('1','true','yes','on')

    try:
        if args.task.lower() in ('segment3d', 'seg3d', 'ms'):
            if not args.input or not os.path.exists(args.input):
                print(json.dumps({"error": "--input path required (NIfTI)"})); sys.exit(1)
            out = run_ms_segmentation(args.model, args.input, ret_img)
            print(json.dumps(out)); return
        if args.task.lower() in ('retina', 'segment2d', 'eye'):
            if not args.input or not os.path.exists(args.input):
                print(json.dumps({"error": "--input path required (image)"})); sys.exit(1)
            out = run_retina_segmentation(args.model, args.input, args.img_size, ret_img)
            print(json.dumps(out)); return
        if args.task.lower() in ('boneage', 'bone_age'):
            if not args.input or not os.path.exists(args.input):
                print(json.dumps({"error": "--input path required (image)"})); sys.exit(1)
            out = run_bone_age(args.model, args.input, max(args.img_size, 300))
            print(json.dumps(out)); return
        print(json.dumps({"error": f"Task '{args.task}' not implemented."})); sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()