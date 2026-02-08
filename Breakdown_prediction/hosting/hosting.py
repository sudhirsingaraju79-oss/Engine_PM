from huggingface_hub import HfApi
import os


api = HfApi(token=os.getenv("HF_TOKEN"))

files = ["Breakdown_prediction/deployment/app.py",
         "Breakdown_prediction/model_building/data_register.py",
         "Breakdown_prediction/deployment/Dockerfile",
         "Breakdown_prediction/hosting/hosting.py",
         "Breakdown_prediction/model_building/prep.py",
         "Breakdown_prediction/deployment/requirements.txt",
         "Breakdown_prediction/model_building/train.py",
         "Breakdown_prediction/data/engine_data.csv",
         "Breakdown_prediction/python_env.yaml",
         "Breakdown_prediction/model_building/featureengineer.py",
         "Breakdown_prediction/model_building/outliercapper.py",
         "Breakdown_prediction/best_engine_PM_prediction_v1.joblib"]

for f in files:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=os.path.basename(f),
        repo_id="sudhirpgcmma02/Engine_PM",  # the target repo
        repo_type="space",  # dataset, model, or space
    )
