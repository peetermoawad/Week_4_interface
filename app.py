"""
app.py  Flask REST API for Water Segmentation
Endpoints:
  GET  /                   -> HTML frontend
  POST /predict            -> Upload TIF -> binary mask PNG
  POST /predict_json       -> Upload TIF -> JSON stats + base64 images
  POST /predict_location   -> lat/lon -> fetch real Sentinel-2 from Planetary Computer -> JSON
  GET  /health             -> Service health check
"""
import os
import io
import base64
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template

from inference import read_tif_bytes, run_inference, mask_to_png_bytes, mask_to_rgb_overlay

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'best_model.pth')
THRESHOLD    = 0.5

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB


def _validate_upload(req):
    if 'file' not in req.files:
        return None, (jsonify({'error': 'No file part. Use key "file".'}), 400)
    f = req.files['file']
    if f.filename == '':
        return None, (jsonify({'error': 'No file selected.'}), 400)
    if not f.filename.lower().endswith(('.tif', '.tiff')):
        return None, (jsonify({'error': 'Only .tif/.tiff accepted.'}), 400)
    return f.read(), None


def _build_json_response(img, mask):
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'weights_loaded': os.path.exists(WEIGHTS_PATH)})


@app.route('/predict', methods=['POST'])
def predict():
    file_bytes, err = _validate_upload(request)
    if err:
        return err
    try:
        img  = read_tif_bytes(file_bytes)
        mask = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        return send_file(io.BytesIO(mask_to_png_bytes(mask)), mimetype='image/png',
                         as_attachment=False, download_name='water_mask.png')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_json', methods=['POST'])
def predict_json():
    file_bytes, err = _validate_upload(request)
    if err:
        return err
    try:
        img  = read_tif_bytes(file_bytes)
        mask = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        return jsonify(_build_json_response(img, mask))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_location', methods=['POST'])
def predict_location():
    """
    Fetch real Sentinel-2 from Microsoft Planetary Computer and run inference.
    Body JSON: {"lat": 30.0, "lon": 31.5, "days_back": 90, "cloud_max": 30}
    """
    try:
        from satellite_fetch import fetch_sentinel2_chip
    except ImportError as ie:
        return jsonify({'error': f'satellite_fetch not available: {ie}'}), 500

    data = request.get_json(silent=True) or {}
    lat  = data.get('lat')
    lon  = data.get('lon')
    if lat is None or lon is None:
        return jsonify({'error': 'Provide "lat" and "lon" in JSON body.'}), 400
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return jsonify({'error': '"lat" and "lon" must be numbers.'}), 400
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({'error': f'Invalid coordinates lat={lat}, lon={lon}'}), 400

    days_back = int(data.get('days_back', 90))
    cloud_max = int(data.get('cloud_max', 30))

    try:
        print(f"[location] Fetching S2 chip at ({lat:.4f}, {lon:.4f}) ...")
        img, scene_meta = fetch_sentinel2_chip(lat, lon,
                                               days_back=days_back,
                                               cloud_max=cloud_max)
        mask   = run_inference(img, WEIGHTS_PATH, threshold=THRESHOLD)
        result = _build_json_response(img, mask)
        result.update({
            'scene_id':         scene_meta.get('scene_id'),
            'scene_date':       scene_meta.get('date'),
            'cloud_cover':      scene_meta.get('cloud_cover'),
            'platform':         scene_meta.get('platform'),
            'bbox':             scene_meta.get('bbox'),
            'query_lat':        lat,
            'query_lon':        lon,
            'rendered_preview': scene_meta.get('rendered_preview', ''),
        })
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/test_samples/<path:filename>')
def serve_sample(filename):
    sample_dir = os.path.join(os.path.dirname(__file__), 'test_samples')
    from flask import send_from_directory
    return send_from_directory(sample_dir, filename)



if __name__ == '__main__':
    print('=' * 60)
    print('  Water Segmentation API + Planetary Computer')
    print('=' * 60)
    print(f'  Weights: {WEIGHTS_PATH}  exists={os.path.exists(WEIGHTS_PATH)}')
    print()
    print('  Endpoints:')
    print('    GET  /                  -> Frontend UI (with map)')
    print('    GET  /health            -> Health check')
    print('    POST /predict           -> Upload TIF -> mask PNG')
    print('    POST /predict_json      -> Upload TIF -> JSON')
    print('    POST /predict_location  -> lat/lon -> fetch S2 -> JSON')
    print()
    app.run(host='0.0.0.0', port=5000, debug=False)
