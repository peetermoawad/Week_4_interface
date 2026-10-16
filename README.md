# Water Segmentation API — README

## Overview

Flask REST API for water body segmentation from 12-channel Sentinel-2 multispectral imagery.

**Best model (Weeks 1–2):** SMP U-Net / ResNet-34 / weight_avg  
**Val IoU: 0.8253 | F1: 0.9043 | Input: 12 Sentinel-2 bands (128×128)**

---

## Project Structure

`
water_segmentation_api/
├── app.py                    # Flask REST API
├── model.py                  # Model architecture (SMP U-Net ResNet-34)
├── inference.py              # Preprocessing + inference pipeline
├── export_weights.py         # Export Kaggle checkpoint -> weights/best_model.pth
├── generate_test_samples.py  # Create synthetic test TIF files
├── test_api.py               # Full API test suite (3 samples × 3 endpoints)
├── requirements.txt          # Python dependencies
├── weights/
│   └── best_model.pth        # Place trained checkpoint here (see below)
├── templates/
│   └── index.html            # HTML frontend (dark UI)
├── test_samples/
│   ├── sample_high_water.tif # Synthetic test: ~69% water
│   ├── sample_mixed.tif      # Synthetic test: ~34% water
│   └── sample_low_water.tif  # Synthetic test: ~9%  water
└── test_results/
    └── test_results.md       # Auto-generated test report
`

---

## Quick Start

### 1. Install dependencies
`ash
pip install -r requirements.txt
`

### 2. Add model weights
Download smp_unet_resnet34_wavg_best.pth from Kaggle output, then:
`ash
python export_weights.py --src /path/to/smp_unet_resnet34_wavg_best.pth
`
> Without weights the API runs in **demo mode** (random weights, masks will not be meaningful).

### 3. Start the server
`ash
python app.py
# → http://localhost:5000
`

### 4. Run tests
`ash
python generate_test_samples.py   # create 3 synthetic TIF files
python test_api.py                # test all endpoints
`

---

## API Endpoints

| Method | Endpoint        | Description                                |
|--------|----------------|--------------------------------------------|
| GET    | /            | HTML frontend                              |
| GET    | /health      | Service health check (JSON)                |
| POST   | /predict     | Upload TIF → binary mask PNG               |
| POST   | /predict_json| Upload TIF → JSON with stats + base64 PNGs|

### POST /predict
`ash
curl -X POST http://localhost:5000/predict \
     -F "file=@my_sentinel2.tif" \
     --output water_mask.png
`
Returns: image/png — binary mask (255=water, 0=land)

### POST /predict_json
`ash
curl -X POST http://localhost:5000/predict_json \
     -F "file=@my_sentinel2.tif"
`
Returns:
`json
{
  "water_percentage": 23.4,
  "water_pixels": 3840,
  "total_pixels": 16384,
  "mask_width": 128,
  "mask_height": 128,
  "mask_png_b64": "<base64 PNG>",
  "overlay_png_b64": "<base64 PNG>"
}
`

---

## Model Details

| Property          | Value                               |
|-------------------|-------------------------------------|
| Architecture      | SMP U-Net                           |
| Encoder           | ResNet-34 (ImageNet pretrained)     |
| Init Strategy     | weight_avg (12→3 ch expansion)      |
| Input             | 12-channel Sentinel-2 TIF (128×128) |
| Output            | Binary mask (water / background)    |
| Loss              | 0.5×BCE + 0.5×Dice                  |
| Val IoU           | **0.8253**                          |
| F1 Score          | **0.9043**                          |
| Training Epochs   | 50 (early stopping)                 |
| LR Schedule       | ReduceLROnPlateau (max IoU)         |

### Band Configuration (Sentinel-2)
| Band | Index | Wavelength | Role        |
|------|-------|-----------|-------------|
| B3   | ch 2  | ~560 nm   | Green (NDWI)|
| B8   | ch 7  | ~842 nm   | NIR         |
| B11  | ch10  | ~1610 nm  | SWIR-1      |
| B1–B12| 0–11 | —         | All 12 bands|

---

## Test Results

All 3 samples × 3 endpoints **passed** locally (demo mode without weights).  
With trained weights, segmentation reflects true water coverage.

| Sample              | /predict | /predict_json | Latency |
|---------------------|----------|---------------|---------|
| sample_high_water   | ✅        | ✅             | ~2.4s   |
| sample_mixed        | ✅        | ✅             | ~2.1s   |
| sample_low_water    | ✅        | ✅             | ~2.2s   |
