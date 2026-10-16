"""
test_api.py  —  API Test Script
================================
Tests all 3 Flask API endpoints with the 3 generated sample TIF files.
Produces a documented report: test_results.md

Prerequisites:
    1. Start the Flask app: python app.py
    2. Generate samples:    python generate_test_samples.py
    3. Run this script:     python test_api.py

Results are printed to console and saved to test_results.md
"""
import os
import sys
import json
import base64
import time
import io
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print('[ERROR] requests library not installed. Run: pip install requests')
    sys.exit(1)

BASE_URL    = 'http://localhost:5000'
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_samples')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLES = [
    'sample_high_water.tif',
    'sample_mixed.tif',
    'sample_low_water.tif',
]


def test_health():
    print('\n[1/4] Testing /health endpoint...')
    try:
        r = requests.get(f'{BASE_URL}/health', timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data['status'] == 'ok'
        print(f'  ✓ Health OK | weights_loaded={data["weights_loaded"]}')
        return True, data
    except Exception as e:
        print(f'  ✗ FAILED: {e}')
        return False, {}


def test_predict_png(sample_path: str):
    """Test /predict endpoint — returns PNG binary mask."""
    name = os.path.basename(sample_path)
    print(f'  Testing /predict with {name}...')
    try:
        t0 = time.time()
        with open(sample_path, 'rb') as f:
            r = requests.post(f'{BASE_URL}/predict',
                              files={'file': (name, f, 'image/tiff')},
                              timeout=60)
        elapsed = time.time() - t0
        assert r.status_code == 200, f'HTTP {r.status_code}: {r.text[:200]}'
        assert r.headers['Content-Type'].startswith('image/png'), \
            f'Expected image/png, got {r.headers["Content-Type"]}'
        png_bytes = r.content
        assert len(png_bytes) > 0
        # Save output
        out_path = os.path.join(RESULTS_DIR, name.replace('.tif', '_mask.png'))
        with open(out_path, 'wb') as f:
            f.write(png_bytes)
        print(f'    ✓ PNG mask received ({len(png_bytes):,} bytes) in {elapsed:.2f}s')
        print(f'    ✓ Saved: {out_path}')
        return True, {'bytes': len(png_bytes), 'elapsed': elapsed}
    except Exception as e:
        print(f'    ✗ FAILED: {e}')
        return False, {}


def test_predict_json(sample_path: str):
    """Test /predict_json endpoint — returns JSON with stats."""
    name = os.path.basename(sample_path)
    print(f'  Testing /predict_json with {name}...')
    try:
        t0 = time.time()
        with open(sample_path, 'rb') as f:
            r = requests.post(f'{BASE_URL}/predict_json',
                              files={'file': (name, f, 'image/tiff')},
                              timeout=60)
        elapsed = time.time() - t0
        assert r.status_code == 200, f'HTTP {r.status_code}: {r.text[:200]}'
        data = r.json()
        # Validate required keys
        for key in ('water_percentage', 'water_pixels', 'total_pixels',
                    'mask_width', 'mask_height', 'mask_png_b64', 'overlay_png_b64'):
            assert key in data, f'Missing key: {key}'
        # Validate ranges
        assert 0 <= data['water_percentage'] <= 100
        assert data['water_pixels'] <= data['total_pixels']
        # Validate base64
        mask_bytes    = base64.b64decode(data['mask_png_b64'])
        overlay_bytes = base64.b64decode(data['overlay_png_b64'])
        assert mask_bytes[:4] == b'\x89PNG'
        assert overlay_bytes[:4] == b'\x89PNG'
        # Save outputs
        mask_path    = os.path.join(RESULTS_DIR, name.replace('.tif', '_mask_json.png'))
        overlay_path = os.path.join(RESULTS_DIR, name.replace('.tif', '_overlay.png'))
        with open(mask_path, 'wb') as f:
            f.write(mask_bytes)
        with open(overlay_path, 'wb') as f:
            f.write(overlay_bytes)
        print(f'    ✓ JSON response OK in {elapsed:.2f}s')
        print(f'    ✓ Water coverage: {data["water_percentage"]}%  '
              f'({data["water_pixels"]:,} / {data["total_pixels"]:,} px)')
        print(f'    ✓ Mask:    {mask_path}')
        print(f'    ✓ Overlay: {overlay_path}')
        return True, data
    except Exception as e:
        print(f'    ✗ FAILED: {e}')
        import traceback; traceback.print_exc()
        return False, {}


def write_report(results: list):
    """Write a Markdown test report."""
    report_path = os.path.join(RESULTS_DIR, 'test_results.md')
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '# Water Segmentation API — Test Report',
        '',
        f'**Generated:** {ts}  ',
        f'**Base URL:** {BASE_URL}  ',
        '',
        '## Test Results',
        '',
        '| Sample | /predict (PNG) | /predict_json | Water % | Latency |',
        '|--------|---------------|---------------|---------|---------|',
    ]
    for row in results:
        png_ok  = '✅' if row['png_ok']  else '❌'
        json_ok = '✅' if row['json_ok'] else '❌'
        water   = f'{row.get("water_pct", "N/A")}%'
        latency = f'{row.get("latency", 0):.2f}s'
        lines.append(f'| {row["name"]} | {png_ok} | {json_ok} | {water} | {latency} |')

    lines += [
        '',
        '## Output Files',
        '',
        'Saved to 	est_results/:',
        '',
    ]
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(('.png', '.json')):
            lines.append(f'- {f}')

    lines += [
        '',
        '## Model Information',
        '',
        '| Property | Value |',
        '|----------|-------|',
        '| Architecture | SMP U-Net / ResNet-34 |',
        '| Init Strategy | weight_avg |',
        '| Input Channels | 12 (Sentinel-2 bands) |',
        '| Image Size | 128×128 |',
        '| Val IoU | 0.8253 |',
        '| F1 Score | 0.9043 |',
        '| Loss | 0.5×BCE + 0.5×Dice |',
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nTest report saved: {report_path}')
    return report_path


def main():
    print('=' * 60)
    print('  Water Segmentation API — Test Suite')
    print('=' * 60)

    # Check samples exist
    missing = [s for s in SAMPLES if not os.path.exists(os.path.join(SAMPLES_DIR, s))]
    if missing:
        print(f'\n[ERROR] Missing test samples: {missing}')
        print('Run: python generate_test_samples.py')
        sys.exit(1)

    # Test health
    health_ok, health_data = test_health()
    if not health_ok:
        print('\n[ERROR] API not reachable. Is app.py running?')
        sys.exit(1)

    # Test each sample
    print('\n[2/4] Testing /predict (binary PNG mask)...')
    png_results = {}
    for s in SAMPLES:
        ok, info = test_predict_png(os.path.join(SAMPLES_DIR, s))
        png_results[s] = (ok, info)

    print('\n[3/4] Testing /predict_json (JSON with stats)...')
    json_results = {}
    for s in SAMPLES:
        ok, info = test_predict_json(os.path.join(SAMPLES_DIR, s))
        json_results[s] = (ok, info)

    # Summary
    print('\n[4/4] Summary')
    print('-' * 50)
    rows = []
    all_passed = True
    for s in SAMPLES:
        png_ok,  png_info  = png_results[s]
        json_ok, json_info = json_results[s]
        passed = png_ok and json_ok
        all_passed = all_passed and passed
        water_pct = json_info.get('water_percentage', 'N/A') if json_ok else 'N/A'
        latency   = json_info.get('elapsed', png_info.get('elapsed', 0))
        status    = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {status}  {s:40s}  water={water_pct}%  t={latency:.2f}s')
        rows.append({
            'name': s, 'png_ok': png_ok, 'json_ok': json_ok,
            'water_pct': water_pct, 'latency': latency
        })

    write_report(rows)

    print('\n' + ('=' * 60))
    if all_passed:
        print('  ALL TESTS PASSED ✓')
    else:
        print('  SOME TESTS FAILED — check output above')
    print('=' * 60)


if __name__ == '__main__':
    main()
