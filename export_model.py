"""
Script để export model tốt nhất từ MLflow Registry ra thư mục models/
"""
import mlflow
import shutil
import os

# Thiết lập tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Tên model trong registry
model_name = "ltn_classifier"
export_dir = "models/ltn_classifier"

try:
    # Load model từ registry
    model_uri = f"models:/{model_name}/1"
    print(f"Dang export model tu: {model_uri}")
    
    # Tạo thư mục export nếu chưa có
    os.makedirs(export_dir, exist_ok=True)
    
    # Tìm đường dẫn thực tế của model
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version(model_name, 1)
    run_id = model_version.run_id
    
    # Đường dẫn thực tế trong mlruns
    # Tìm experiment_id
    experiments = mlflow.search_experiments()
    exp_id = None
    for exp in experiments:
        if exp.name == "ltn_classification":
            exp_id = exp.experiment_id
            break
    
    # Thử tìm model từ run_id
    actual_path = None
    if exp_id:
        test_path = f"mlruns/{exp_id}/{run_id}/artifacts/ltn_model"
        if os.path.exists(test_path):
            actual_path = test_path
    
    # Nếu không tìm thấy, tìm trong models/ của experiment
    if not actual_path and exp_id:
        models_dir = f"mlruns/{exp_id}/models"
        if os.path.exists(models_dir):
            # Tìm model có tag mlflow.modelVersions và có artifacts
            for model_dir in os.listdir(models_dir):
                model_full_path = os.path.join(models_dir, model_dir, "artifacts")
                if os.path.exists(model_full_path) and os.path.isdir(model_full_path):
                    # Kiểm tra xem có file MLmodel không
                    if os.path.exists(os.path.join(model_full_path, "MLmodel")):
                        actual_path = model_full_path
                        break
    
    # Copy toàn bộ thư mục model
    if actual_path and os.path.exists(actual_path):
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        shutil.copytree(actual_path, export_dir)
        print(f"Da export model thanh cong vao: {export_dir}")
    else:
        # Thử cách khác: copy từ model artifacts trực tiếp
        print(f"Khong tim thay duong dan, dang tim trong mlruns...")
        # Tìm trong tất cả experiments
        for exp in experiments:
            test_path = f"mlruns/{exp.experiment_id}/models"
            if os.path.exists(test_path):
                # Tìm model có tag mlflow.modelVersions
                for model_dir in os.listdir(test_path):
                    model_full_path = os.path.join(test_path, model_dir, "artifacts")
                    if os.path.exists(model_full_path):
                        # Kiểm tra xem có phải model đã đăng ký không
                        tags_path = os.path.join(test_path, model_dir, "tags", "mlflow.modelVersions")
                        if os.path.exists(tags_path):
                            # Copy model này
                            if os.path.exists(export_dir):
                                shutil.rmtree(export_dir)
                            shutil.copytree(model_full_path, export_dir)
                            print(f"Da export model thanh cong vao: {export_dir}")
                            exit(0)
        
        # Nếu vẫn không tìm thấy, thử load và copy từ run
        print(f"Dang load model tu registry...")
        model = mlflow.pyfunc.load_model(model_uri)
        # Tìm đường dẫn thực tế của model đã load
        # Model đã load có thể có thuộc tính _model_impl
        # Nhưng cách tốt nhất là copy từ source
        print(f"Da load model thanh cong, nhung can copy thu muc artifacts")
        
except Exception as e:
    print(f"Loi khi export model: {e}")
    print(f"Hay chay train.py truoc de tao model")
    # Nếu không có model, train một model mới
    try:
        print("Dang train model moi...")
        from mlflow_project.train import train_and_log_model
        acc, f1, run_id = train_and_log_model(150, 7)
        model_uri = f"runs:/{run_id}/ltn_model"
        model = mlflow.pyfunc.load_model(model_uri)
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        mlflow.pyfunc.save_model(export_dir, python_model=model)
        print(f"Da train va export model moi (acc={acc:.4f}, f1={f1:.4f})")
    except Exception as e2:
        print(f"Loi khi train model: {e2}")

