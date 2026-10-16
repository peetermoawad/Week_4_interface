# Water Segmentation API — README (FastAPI & Flask)

## Overview

High-performance REST API (available in both **FastAPI** and **Flask**) for water body segmentation from 12-channel Sentinel-2 multispectral satellite imagery.

- **FastAPI with Interactive Swagger UI (`/docs`) & ReDoc (`/redoc`)**
- **Planetary Computer Integration**: Live global coordinates fetch real Sentinel-2 satellite data.
- **Interactive Leaflet Globe / Map Frontend**
- **Best model (Weeks 1–2):** SMP U-Net / ResNet-34 / weight_avg  
- **Val IoU: 0.8253 | F1: 0.9043 | Input: 12 Sentinel-2 bands (128×128)**

---

## Project Structure

```
water_segmentation_api/
├── fastapi_app.py            # FastAPI REST API (Swagger UI /docs & ReDoc)
├── app.py                    # Flask REST API
├── run_fastapi.bat           # 1-click launcher for FastAPI (port 8000)
├── run_server.bat            # 1-click launcher for Flask (port 5000)
├── model.py                  # Model architecture (SMP U-Net ResNet-34)
├── inference.py              # Preprocessing + inference pipeline
├── satellite_fetch.py        # Planetary Computer Sentinel-2 chip fetcher
├── export_weights.py         # Export Kaggle checkpoint -> weights/best_model.pth
├── generate_test_samples.py  # Create synthetic test TIF files
├── test_api.py               # Full API test suite (3 samples × 3 endpoints)
├── requirements.txt          # Python dependencies (FastAPI, Flask, PyTorch, etc.)
├── weights/
│   └── best_model.pth        # Place trained checkpoint here (see below)
├── templates/
│   └── index.html            # Interactive HTML frontend (Map + Upload)
├── test_samples/
│   ├── sample_high_water.tif # Synthetic test: ~69% water
│   ├── sample_mixed.tif      # Synthetic test: ~34% water
│   └── sample_low_water.tif  # Synthetic test: ~9%  water
└── test_results/
    └── test_results.md       # Auto-generated test report
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add model weights
Download `smp_unet_resnet34_wavg_best.pth` from Kaggle output, then:
```bash
python export_weights.py --src /path/to/smp_unet_resnet34_wavg_best.pth
```
> Without weights the API runs in **demo mode** (random weights, masks will not be meaningful).

### 3. Start the server

#### Option A: FastAPI (Recommended — Interactive Swagger UI)
```bash
python fastapi_app.py
# Or double-click run_fastapi.bat
# → Frontend: http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
# → ReDoc: http://localhost:8000/redoc
```

#### Option B: Flask
```bash
python app.py
# Or double-click run_server.bat
# → Frontend: http://localhost:5000
```

### 4. Run tests
```bash
python generate_test_samples.py   # create 3 synthetic TIF files
python test_api.py                # test all endpoints
```

---

## API Endpoints

| Method | Endpoint            | Description                                                       |
|--------|---------------------|-------------------------------------------------------------------|
| GET    | `/`                 | Interactive HTML frontend with Leaflet world map & file upload    |
| GET    | `/health`           | Service health check & weights verification                       |
| GET    | `/docs` (FastAPI)   | Interactive Swagger UI (test endpoints directly in browser)       |
| GET    | `/redoc` (FastAPI)  | OpenAPI ReDoc documentation                                       |
| POST   | `/predict`          | Upload TIF → binary mask PNG                                      |
| POST   | `/predict_json`     | Upload TIF → JSON metrics + base64 binary mask and RGB overlay    |
| POST   | `/predict_location` | Send `{"lat": ..., "lon": ...}` → Fetch Sentinel-2 & run inference|

### POST /predict
```bash
# FastAPI
curl -X POST http://localhost:8000/predict \
     -F "file=@test_samples/sample_high_water.tif" \
     --output water_mask.png

# Flask
curl -X POST http://localhost:5000/predict \
     -F "file=@test_samples/sample_high_water.tif" \
     --output water_mask.png
```
Returns: `image/png` — binary mask (255=water, 0=land)

### POST /predict_json
```bash
curl -X POST http://localhost:8000/predict_json \
     -F "file=@test_samples/sample_high_water.tif"
```
Returns:
```json
{
  "water_percentage": 72.99,
  "water_pixels": 11958,
  "total_pixels": 16384,
  "mask_width": 128,
  "mask_height": 128,
  "mask_png_b64": "<base64 PNG>",
  "overlay_png_b64": "<base64 PNG>",
  "has_water": true,
  "water_level": "High"
}
```

### POST /predict_location
```bash
curl -X POST http://localhost:8000/predict_location \
     -H "Content-Type: application/json" \
     -d '{"lat": 30.56, "lon": 32.28, "days_back": 120, "cloud_max": 30}'
```

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

All 3 samples × 3 endpoints **passed** locally.  
With trained weights, segmentation reflects true water coverage.

| Sample              | /predict | /predict_json | Latency |
|---------------------|----------|---------------|---------|
| sample_high_water   | ✅        | ✅             | ~2.4s   |
| sample_mixed        | ✅        | ✅             | ~2.1s   |
| sample_low_water    | ✅        | ✅             | ~2.2s   |
