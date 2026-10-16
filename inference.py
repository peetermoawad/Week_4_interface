"""
inference.py  —  Inference pipeline for the water segmentation model
=====================================================================
Accepts a 12-channel GeoTIFF and returns a binary mask (H x W, uint8).
"""
import io
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from model import build_model, load_weights, INPUT_CHANNELS, IMAGE_SIZE

# ── Load normalisation stats from model_meta.json (Kaggle output) ─────────────
# Fallback to approximate values if the file is not present yet.
_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_meta.json')

if os.path.exists(_META_PATH):
    with open(_META_PATH, 'r') as _f:
        _meta = json.load(_f)
    MEANS = np.array(_meta['normalisation']['means'], dtype=np.float32)
    STDS  = np.array(_meta['normalisation']['stds'],  dtype=np.float32)
    print(f'[inference] Loaded normalisation stats from {_META_PATH}')
else:
    # Approximate fallback (replace with actual values from model_meta.json)
    MEANS = np.array([
        174.1, 166.1, 326.9, 325.8, 1945.2, 1298.2,
        678.8,  66.3,  247.0, 291.1,   20.6,    4.6
    ], dtype=np.float32)
    STDS = np.array([
        120.0, 115.0, 250.0, 240.0, 500.0, 400.0,
        300.0, 120.0, 180.0, 200.0,  30.0,  15.0
    ], dtype=np.float32)
    print('[inference] model_meta.json not found — using fallback normalisation stats')


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


_model_cache = {}


def get_model(weights_path: str):
    """Return cached model (loaded on first call)."""
    if weights_path not in _model_cache:
        device = _get_device()
        m = build_model(in_channels=INPUT_CHANNELS, strategy='weight_avg')
        if os.path.exists(weights_path):
            m = load_weights(m, weights_path, device)
        else:
            # No weights available — run in random-weight demo mode
            m.eval()
        _model_cache[weights_path] = (m.to(device), device)
    return _model_cache[weights_path]


def read_tif_bytes(file_bytes: bytes) -> np.ndarray:
    """Read a 12-band GeoTIFF from raw bytes -> float32 (C,H,W)."""
    if not RASTERIO_AVAILABLE:
        raise RuntimeError('rasterio is required to read GeoTIFF files')
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def read_tif_path(path: str) -> np.ndarray:
    """Read a 12-band GeoTIFF from disk -> float32 (C,H,W)."""
    if not RASTERIO_AVAILABLE:
        raise RuntimeError('rasterio is required to read GeoTIFF files')
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def preprocess(img: np.ndarray, means: np.ndarray = MEANS, stds: np.ndarray = STDS) -> torch.Tensor:
    """
    Normalise and convert to tensor.
    img : (C, H, W) float32
    returns : (1, C, H, W) tensor
    """
    if img.shape[0] != INPUT_CHANNELS:
        raise ValueError(f'Expected {INPUT_CHANNELS} channels, got {img.shape[0]}')
    # Normalise
    img = (img - means[:, None, None]) / (stds[:, None, None] + 1e-8)
    # Resize to model input size if needed
    if img.shape[1] != IMAGE_SIZE or img.shape[2] != IMAGE_SIZE:
        t = torch.from_numpy(img).unsqueeze(0)           # (1, C, H, W)
        t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE),
                          mode='bilinear', align_corners=False)
        return t
    return torch.from_numpy(img).unsqueeze(0)


def run_inference(img: np.ndarray, weights_path: str,
                  threshold: float = 0.5) -> np.ndarray:
    """
    Full inference pipeline.

    Parameters
    ----------
    img          : (12, H, W) float32 array (raw band values)
    weights_path : path to the saved .pth checkpoint
    threshold    : sigmoid probability threshold (default 0.5)

    Returns
    -------
    mask : (H, W) uint8 array — 255 = water, 0 = background
    """
    orig_h, orig_w = img.shape[1], img.shape[2]
    model, device  = get_model(weights_path)

    inp = preprocess(img).to(device)               # (1, C, SIZE, SIZE)
    with torch.no_grad():
        logit = model(inp)                         # (1, 1, SIZE, SIZE)
        prob  = torch.sigmoid(logit)               # probabilities

    prob_np = prob.squeeze().cpu().numpy()         # (SIZE, SIZE)

    # Resize back to original spatial size if needed
    if orig_h != IMAGE_SIZE or orig_w != IMAGE_SIZE:
        prob_pil = Image.fromarray((prob_np * 255).astype(np.uint8))
        prob_pil = prob_pil.resize((orig_w, orig_h), Image.BILINEAR)
        prob_np  = np.array(prob_pil).astype(np.float32) / 255.0

    mask = (prob_np >= threshold).astype(np.uint8) * 255
    return mask


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert (H, W) uint8 mask to PNG bytes."""
    img = Image.fromarray(mask, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def mask_to_rgb_overlay(raw_tif: np.ndarray, mask: np.ndarray) -> bytes:
    """
    Overlay the predicted mask on a false-colour composite for visualisation.
    Returns PNG bytes of the overlay image.
    """
    def norm(band, lo=2, hi=98):
        lo_v, hi_v = np.percentile(band, lo), np.percentile(band, hi)
        return np.clip((band - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)

    # True-colour RGB from bands 3 (R), 2 (G), 1 (B) — indices 3, 2, 1
    r = (norm(raw_tif[3]) * 255).astype(np.uint8)
    g = (norm(raw_tif[2]) * 255).astype(np.uint8)
    b = (norm(raw_tif[1]) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    # Overlay water pixels in blue
    water = mask > 127
    overlay = rgb.copy()
    overlay[water, 0] = 30
    overlay[water, 1] = 100
    overlay[water, 2] = 200

    img = Image.fromarray(overlay, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()
