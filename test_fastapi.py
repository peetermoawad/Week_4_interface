"""
test_fastapi.py  —  Automated Verification Suite for FastAPI Implementation
"""
import os
import sys
import requests

BASE_URL = "http://127.0.0.1:8000"
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_samples')

SAMPLES = [
    'sample_high_water.tif',
    'sample_mixed.tif',
    'sample_low_water.tif'
]

passed_tests = 0
total_tests = 0

def record_test(name, success, details=""):
    global passed_tests, total_tests
    total_tests += 1
    if success:
        passed_tests += 1
        print(f"  [PASS] {name} {details}")
    else:
        print(f"  [FAIL] {name} {details}")
    return success

print("=" * 65)
print("  🚀 Testing AquaSeg FastAPI Server (http://127.0.0.1:8000)")
print("=" * 65)

# 1. Health endpoint
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    data = r.json()
    record_test("GET /health", r.status_code == 200 and data.get("status") == "ok" and data.get("weights_loaded") is True, f"-> weights_loaded={data.get('weights_loaded')}")
except Exception as e:
    record_test("GET /health", False, f"-> Error: {e}")

# 2. Interactive Documentation (Swagger UI)
try:
    r = requests.get(f"{BASE_URL}/docs", timeout=5)
    record_test("GET /docs (Swagger UI)", r.status_code == 200 and "swagger-ui" in r.text.lower())
except Exception as e:
    record_test("GET /docs", False, f"-> Error: {e}")

# 3. OpenAPI Schema
try:
    r = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    data = r.json()
    record_test("GET /openapi.json", r.status_code == 200 and "paths" in data, f"-> Title: '{data.get('info', {}).get('title')}'")
except Exception as e:
    record_test("GET /openapi.json", False, f"-> Error: {e}")

# 4. Frontend UI
try:
    r = requests.get(f"{BASE_URL}/", timeout=5)
    record_test("GET / (Interactive Map UI)", r.status_code == 200 and "AquaSeg" in r.text)
except Exception as e:
    record_test("GET /", False, f"-> Error: {e}")

# 5. POST /predict (Streaming binary PNG)
print("\n--- Testing POST /predict (PNG Streaming) ---")
for s in SAMPLES:
    path = os.path.join(SAMPLES_DIR, s)
    if not os.path.exists(path):
        record_test(f"POST /predict [{s}]", False, "-> File not found")
        continue
    try:
        with open(path, 'rb') as f:
            r = requests.post(f"{BASE_URL}/predict", files={"file": (s, f, "image/tiff")}, timeout=10)
        is_png = r.content.startswith(b'\x89PNG\r\n\x1a\n')
        record_test(f"POST /predict [{s}]", r.status_code == 200 and is_png, f"-> size={len(r.content)} bytes, Content-Type={r.headers.get('content-type')}")
    except Exception as e:
        record_test(f"POST /predict [{s}]", False, f"-> Error: {e}")

# 6. POST /predict_json (Stats + Base64)
print("\n--- Testing POST /predict_json (Inference Metrics) ---")
for s in SAMPLES:
    path = os.path.join(SAMPLES_DIR, s)
    if not os.path.exists(path):
        record_test(f"POST /predict_json [{s}]", False, "-> File not found")
        continue
    try:
        with open(path, 'rb') as f:
            r = requests.post(f"{BASE_URL}/predict_json", files={"file": (s, f, "image/tiff")}, timeout=10)
        data = r.json()
        valid = (
            r.status_code == 200 and
            "water_percentage" in data and
            "mask_png_b64" in data and
            "overlay_png_b64" in data and
            "has_water" in data and
            "water_level" in data
        )
        record_test(
            f"POST /predict_json [{s}]",
            valid,
            f"-> water_pct={data.get('water_percentage')}% | level={data.get('water_level')} | has_water={data.get('has_water')}"
        )
    except Exception as e:
        record_test(f"POST /predict_json [{s}]", False, f"-> Error: {e}")

# 7. POST /predict_location (Microsoft Planetary Computer)
print("\n--- Testing POST /predict_location (Planetary Computer) ---")
try:
    loc_payload = {"lat": 27.50, "lon": 28.50, "days_back": 120, "cloud_max": 30} # Western Desert (Dry)
    r = requests.post(f"{BASE_URL}/predict_location", json=loc_payload, timeout=30)
    data = r.json()
    valid = r.status_code == 200 and "water_percentage" in data and "scene_id" in data
    record_test("POST /predict_location [Western Desert]", valid, f"-> water_pct={data.get('water_percentage')}% | Scene: {data.get('scene_id')}")
except Exception as e:
    record_test("POST /predict_location [Western Desert]", False, f"-> Error: {e}")

# 8. Error Handling Tests
print("\n--- Testing Error Handling & Validation ---")
# Non-tif upload
try:
    r = requests.post(f"{BASE_URL}/predict", files={"file": ("test.txt", b"plain text", "text/plain")}, timeout=5)
    record_test("Invalid file extension handling (400)", r.status_code == 400, f"-> status={r.status_code}")
except Exception as e:
    record_test("Invalid file extension handling", False, f"-> Error: {e}")

# Out-of-bounds coordinates (FastAPI Pydantic auto-validation)
try:
    r = requests.post(f"{BASE_URL}/predict_location", json={"lat": 150.0, "lon": 300.0}, timeout=5)
    record_test("Pydantic invalid coordinate validation (422)", r.status_code == 422, f"-> status={r.status_code}")
except Exception as e:
    record_test("Pydantic invalid coordinate validation", False, f"-> Error: {e}")

print("\n" + "=" * 65)
print(f"  Summary: {passed_tests} / {total_tests} tests PASSED ({round(100 * passed_tests / total_tests)}%)")
print("=" * 65)

if passed_tests == total_tests:
    sys.exit(0)
else:
    sys.exit(1)
