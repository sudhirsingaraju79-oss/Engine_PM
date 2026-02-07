# for data manipulation
import pandas as pd
import sklearn
## EDA
import matplotlib.pyplot as plt
import seaborn as sns
import math
from xgboost import XGBClassifier
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download
# format for EDA visualisation
sns.set(style="whitegrid", font_scale=1.1)
# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
# read data for Huggingface dataset space
DATASET_PATH = "hf://datasets/sudhirpgcmma02/Engine_PM/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)

#Features naming standardisation for easy handling
df.columns = (df.columns
                   .str.strip()
                   .str.replace(" ","_")
                   .str.replace(r"[^\w]","_",regex=True)
  )

# Targe varaible intialisation
target_col = 'Engine_Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sudhirpgcmma02/Engine_PM",
        repo_type="dataset",
    )
print("Dataset after split  loaded successfully to Huggingface.....")
