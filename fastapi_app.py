"""
fastapi_app.py  —  FastAPI REST API for Water Segmentation
===========================================================
High-performance async-ready API with automatic Swagger UI documentation (/docs).

Endpoints:
  GET  /                   -> Interactive HTML frontend (with Map & Upload)
  GET  /health             -> Service health check & weights status
  POST /predict            -> Upload .TIF -> binary mask PNG directly
  POST /predict_json       -> Upload .TIF -> JSON metrics & Base64 images
  POST /predict_location   -> lat/lon -> fetch Sentinel-2 from Planetary Computer -> JSON
  GET  /test_samples/{fn}  -> Serve test sample files
  GET  /docs               -> Interactive Swagger UI API Docs
  GET  /redoc              -> ReDoc alternative documentation
"""

import os
import io
import base64
import traceback
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import Response, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import read_tif_bytes, run_inference, mask_to_png_bytes, mask_to_rgb_overlay

# Paths and configuration
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'best_model.pth')
THRESHOLD    = 0.5
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR    = os.path.join(BASE_DIR, 'static')
SAMPLES_DIR   = os.path.join(BASE_DIR, 'test_samples')

# Initialize FastAPI App with OpenAPI metadata
app = FastAPI(
    title="AquaSeg — Water Segmentation API",
    description="Multispectral Sentinel-2 Water Body Segmentation API using SMP U-Net ResNet-34 & Microsoft Planetary Computer.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if it exists
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180")
    days_back: int = Field(90, ge=1, le=365, description="Search history in days")
    cloud_max: int = Field(30, ge=0, le=100, description="Maximum allowable cloud cover percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 30.56,
                "lon": 32.28,
                "days_back": 120,
                "cloud_max": 30
            }
        }


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Server status")
    weights_loaded: bool = Field(..., description="Whether trained PyTorch weights exist")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _build_json_response(img: np.ndarray, mask: np.ndarray) -> dict:
    png_b64   = base64.b64encode(mask_to_png_bytes(mask)).decode()
    overlay   = base64.b64encode(mask_to_rgb_overlay(img, mask)).decode()
    water_px  = int((mask > 127).sum())
    total_px  = int(mask.size)
    water_pct = round(100.0 * water_px / total_px, 2)
    return {
        'water_percentage': water_pct,
        'water_pixels':     water_px,
        'total_pixels':     total_px,
        'mask_width':       mask.shape[1],
        'mask_height':      mask.shape[0],
        'mask_png_b64':     png_b64,
        'overlay_png_b64':  overlay,
        'has_water':        water_pct > 5.0,
        'water_level': (
            'High'   if water_pct > 40 else
            'Medium' if water_pct > 10 else
            'Low'    if water_pct >  2 else 'None'
        ),
    }


async def _read_upload_file(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if not file.filename.lower().endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only .tif / .tiff files are accepted.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Frontend"], summary="Interactive UI")
def index():
    """Serves the interactive web interface (Leaflet Map + File Upload)."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html template not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Health Check")
def health():
    """Returns server status and whether trained weights file is available."""
    return HealthResponse(
        status="ok",
        weights_loaded=os.path.exists(WEIGHTS_PATH)
    )


@app.post("/predict", tags=["Segmentation"], summary="Predict Mask PNG")
async def predict(file: UploadFile = File(..., description="12-channel Sentinel-2 TIF image")):
    """
    Accepts a multipart file upload (.tif/.tiff), performs deep learning model inference,
    and directly streams back the binary water mask as a PNG image.
    """
    file_bytes = await _read_upload_file(file)
    try:
        img  = read_tif_bytes(file_bytes)
        mask = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        png_data = mask_to_png_bytes(mask)
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": 'inline; filename="water_mask.png"'}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_json", tags=["Segmentation"], summary="Predict Stats & Images (JSON)")
async def predict_json(file: UploadFile = File(..., description="12-channel Sentinel-2 TIF image")):
    """
    Accepts a multipart file upload and returns detailed JSON metrics along with
    Base64-encoded binary mask and RGB false-color overlay.
    """
    file_bytes = await _read_upload_file(file)
    try:
        img  = read_tif_bytes(file_bytes)
        mask = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        return JSONResponse(content=_build_json_response(img, mask))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_location", tags=["Planetary Computer"], summary="Fetch Sentinel-2 & Run Inference")
def predict_location(req: LocationRequest):
    """
    Given (lat, lon) coordinates, queries Microsoft Planetary Computer STAC catalog,
    streams real Sentinel-2 L2A multispectral data, and runs neural water segmentation.
    """
    try:
        from satellite_fetch import fetch_sentinel2_chip
    except ImportError as ie:
        raise HTTPException(status_code=500, detail=f"satellite_fetch not available: {ie}")

    try:
        print(f"[FastAPI - location] Fetching S2 chip at ({req.lat:.4f}, {req.lon:.4f}) ...")
        img, scene_meta = fetch_sentinel2_chip(
            req.lat,
            req.lon,
            days_back=req.days_back,
            cloud_max=req.cloud_max
        )
        mask   = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        result = _build_json_response(img, mask)
        result.update({
            'scene_id':         scene_meta.get('scene_id'),
            'scene_date':       scene_meta.get('date'),
            'cloud_cover':      scene_meta.get('cloud_cover'),
            'platform':         scene_meta.get('platform'),
            'bbox':             scene_meta.get('bbox'),
            'query_lat':        req.lat,
            'query_lon':        req.lon,
            'rendered_preview': scene_meta.get('rendered_preview', ''),
        })
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_samples/{filename:path}", tags=["Test Samples"], summary="Download Test Samples")
def serve_sample(filename: str):
    """Serves sample TIF files for testing."""
    file_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Sample {filename} not found.")
    return FileResponse(file_path)


# ---------------------------------------------------------------------------
# Direct Execution via Uvicorn
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    print('=' * 65)
    print('  🌊 AquaSeg FastAPI Server (Uvicorn)')
    print('=' * 65)
    print(f'  Weights: {WEIGHTS_PATH} (exists={os.path.exists(WEIGHTS_PATH)})')
    print()
    print('  Endpoints:')
    print('    GET  /                  -> Interactive Frontend (with map)')
    print('    GET  /docs              -> Swagger UI Documentation')
    print('    GET  /redoc             -> ReDoc Documentation')
    print('    GET  /health            -> Health check')
    print('    POST /predict           -> Upload TIF -> mask PNG')
    print('    POST /predict_json      -> Upload TIF -> JSON + Base64')
    print('    POST /predict_location  -> lat/lon -> fetch S2 -> JSON')
    print()
    print('  Starting server on http://localhost:8000 ...')
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
