from flask import Flask, render_template, request
import mlflow.pyfunc
import mlflow
import numpy as np
import os

app = Flask(__name__)

# Thiết lập MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Thử load model từ nhiều nguồn (ưu tiên file path cho Docker)
model = None

# Cách 1: Thử load từ file path (ưu tiên cho Docker - model đã được export sẵn)
model_path = "models/ltn_classifier"
if os.path.exists(model_path):
    try:
        model = mlflow.pyfunc.load_model(model_path)
        print("✅ Đã load model từ file path")
    except Exception as e:
        print(f"⚠️ Không thể load từ file path: {e}")

# Cách 2: Nếu không có file path, thử load từ Model Registry
if model is None:
    try:
        MODEL_URI = "models:/ltn_classifier/1"
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Đã load model từ MLflow Model Registry")
    except Exception as e:
        print(f"⚠️ Không thể load từ Registry: {e}")
        
        # Cách 3: Tự train model mới (fallback - chỉ khi không có model nào)
        try:
            print("🔄 Đang train model mới...")
            from mlflow_project.train import train_and_log_model
            
            # Train model với tham số tốt nhất (150, 7)
            acc, f1, run_id = train_and_log_model(150, 7)
            model_uri = f"runs:/{run_id}/ltn_model"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Đã train và load model mới (acc={acc:.4f}, f1={f1:.4f})")
            
            # Đăng ký model vào registry để lần sau dùng
            try:
                mlflow.register_model(model_uri, "ltn_classifier")
                print("✅ Đã đăng ký model vào Registry")
            except:
                pass  # Có thể đã tồn tại
        except Exception as e3:
            print(f"❌ Lỗi khi train model: {e3}")
            model = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Đọc 10 feature đầu vào
            features = [float(request.form[f"f{i}"]) for i in range(1, 11)]
            arr = np.array(features).reshape(1, -1)
            prediction = int(model.predict(arr)[0])
        except Exception as e:
            prediction = f"Lỗi khi dự đoán: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
