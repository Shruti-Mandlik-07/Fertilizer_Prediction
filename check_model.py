import os
import joblib
import pandas as pd

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fertilizer_model.pkl"
)

info_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_info.pkl"
)

print("Model path:", model_path)
print("File exists:", os.path.isfile(model_path))
print("File size:", f"{os.path.getsize(model_path):,} bytes" if os.path.isfile(model_path) else "")

try:
    model = joblib.load(model_path)
    print("\nSUCCESS: Model loaded!")
    print("Model type:", type(model))

    if os.path.isfile(info_path):
        info = joblib.load(info_path)
        print("\nModel Info:")
        print("  Algorithm:", info.get("model_name"))
        print("  Features:", info.get("features"))
        print("  Target:", info.get("target"))

    if hasattr(model, "classes_"):
        print("\nTarget Classes (Fertilizers):")
        for cls in model.classes_:
            print(f"  - {cls}")

    # Test sample inference
    test_sample = pd.DataFrame([{
        "Nitrogen": 60.0,
        "Phosphorous": 70.0,
        "Potassium": 70.0,
        "Temperature": 26.0,
        "Moisture": 0.60,
        "Rainfall": 110.0,
        "PH": 6.5,
        "Soil": "Loamy Soil",
        "Crop": "wheat"
    }])
    pred = model.predict(test_sample)[0]
    print(f"\nVerification Test Run (Wheat / Loamy Soil): Predicted '{pred}'")

except Exception as e:
    print("ERROR:", repr(e))
