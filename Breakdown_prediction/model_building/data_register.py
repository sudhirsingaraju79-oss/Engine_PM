from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Huggingface data set repository creation
repo_id = "sudhirpgcmma02/Engine_PM"
repo_type = "dataset"


# Initialize API client token intialistion for pusing data and retrival of data for Huggingface
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

local_file ="/content/Breakdown_prediction/data/engine_data.csv"

folder_path="data/engine_data.csv"
# Register the data directly  to Huggingface
api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
