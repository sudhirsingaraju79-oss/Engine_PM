import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib
import shap
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from pprint import pprint
from xgboost import XGBClassifier # Added for XGBoost
from sklearn.ensemble import RandomForestClassifier # Added for RandomForest
# custom class inheritance
from featureengineer import FeatureEngineer
from outliercapper import OutlierCapper

api = HfApi()

Xtrain_path = "hf://datasets/sudhirpgcmma02/Engine_PM/Xtrain.csv"
Xtest_path = "hf://datasets/sudhirpgcmma02/Engine_PM/Xtest.csv"
ytrain_path = "hf://datasets/sudhirpgcmma02/Engine_PM/ytrain.csv"
ytest_path = "hf://datasets/sudhirpgcmma02/Engine_PM/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame and copy it.
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # These are the expected column names after initial preprocessing
            # They should be consistent with the features defined in the overall dataset.
            expected_column_names = [
                'Engine_rpm', 'Lub_oil_pressure', 'Fuel_pressure',
                'Coolant_pressure', 'lub_oil_temp', 'Coolant_temp'
            ]
            df = pd.DataFrame(X, columns=expected_column_names)

        df.columns = (df.columns
                           .str.strip()
                           .str.replace(" ","_")
                           .str.replace(r"[^\w]","_",regex=True)
        )

        engine_rpm_col = 'Engine_rpm'
        lub_oil_pressure_col = 'Lub_oil_pressure'
        fuel_pressure_col = 'Fuel_pressure'
        coolant_pressure_col = 'Coolant_pressure'
        lub_oil_temp_col = 'lub_oil_temp'
        coolant_temp_col = 'Coolant_temp'

        core_sensor_cols = [
            engine_rpm_col, lub_oil_pressure_col, fuel_pressure_col,
            coolant_pressure_col, lub_oil_temp_col, coolant_temp_col
        ]

        # ===== diff features
        for col_name in df.select_dtypes(include=np.number).columns:
            df[f"{col_name}_diff"] = df[col_name].diff()

        # ===== rolling mean
        for col_name in core_sensor_cols:
            if col_name in df.columns:
                df[f"{col_name}_roll5"] = df[col_name].rolling(5).mean()

        # ===== anomaly flag (3-sigma)
        for col_name in core_sensor_cols:
            if col_name in df.columns:
                std = df[col_name].std()
                if std > 1e-9: # Use a small epsilon to check for non-zero std
                    df[f"{col_name}_anom"] = (df[col_name].diff().abs() > 3 * std).astype(int)
                else:
                    df[f"{col_name}_anom"] = 0 # No anomaly if data is constant

        # ===== aggregates
        # Corrected: Use actual string column names instead of integer indices
        df["temp_gap"] = df[lub_oil_temp_col] - df[coolant_temp_col]   # oil vs coolant
        df["pressure_sum"] = df[[lub_oil_pressure_col, fuel_pressure_col, coolant_pressure_col]].sum(axis=1)

        df = df.fillna(0)

        # Return DataFrame with new column names for easier debugging and feature name extraction
        return df

class OutlierCapper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.bounds = []

        # If X is a DataFrame, convert to numpy array for percentile calculation to avoid FutureWarning
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        for i in range(X_np.shape[1]):
            Q1 = np.percentile(X_np[:, i], 25)
            Q3 = np.percentile(X_np[:, i], 75)
            IQR = Q3 - Q1
            self.bounds.append((Q1-1.5*IQR, Q3+1.5*IQR))

        return self

    def transform(self, X):

        # If X is a DataFrame, convert to numpy array for manipulation, then back to DataFrame if needed
        X_transformed = X.copy()
        if isinstance(X_transformed, pd.DataFrame):
            column_names = X_transformed.columns
            X_np = X_transformed.values
        else:
            column_names = None # Column names are lost if X is already numpy
            X_np = X_transformed

        for i, (low, high) in enumerate(self.bounds):
            X_np[:, i] = np.clip(X_np[:, i], low, high)

        if column_names is not None:
            return pd.DataFrame(X_np, columns=column_names) # Return DataFrame to preserve column names
        else:
            return X_np # Return numpy array if no original column names

def create_pipe(model):

    return Pipeline([
        ("feat", FeatureEngineer()),      # feature engineering
        ("impute", SimpleImputer(strategy="median")), # SimpleImputer works on numpy arrays
        ("outlier", OutlierCapper()), # OutlierCapper now returns DataFrame if input was DataFrame
        ("scale", RobustScaler()), # RobustScaler outputs numpy arrays
        ("model", model)
    ])

df=X_train.copy()
#renaming columns for easy processing
df.columns = (df.columns
                   .str.strip()
                   .str.replace(" ","_")
                   .str.replace(r"[^\w]","_",regex=True)
  )
print(df.head(10))

# Split into X (features) and y (target)
Xtrain =X_train.copy()
ytrain =y_train.copy()
print("########################### independent, dependent varial split completed ################################")

# Extract column names as lists for the ColumnTransformer
num_feat_cols = Xtrain.select_dtypes(include=[np.number]).columns.tolist()
cat_feat_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()


print("########################### test train split completed ################################")

print("########################### preprocessing creation completed ################################")

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts().get(0, 0) / ytrain.value_counts().get(1, 1) # Added .get to handle potential missing classes gracefully
print("class_weight distribution",class_weight)

#  hyper parameter for DT

def objective_dt(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": 'balanced',
        "random_state": 42
    }

    model = DecisionTreeClassifier(**params)

    pipeline=create_pipe(model)
    score = cross_val_score(
        pipeline, Xtrain, ytrain, # ytrain is a DataFrame, convert to Series if it's 1 column
        cv=5, scoring="recall"
    ).mean()

    return score

study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective_dt, n_trials=25)

best_dt = DecisionTreeClassifier(**study_dt.best_params, class_weight="balanced")
best_dt_pipeline =create_pipe(best_dt)
best_dt_pipeline.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
best_dt = best_dt_pipeline # Assign the fitted pipeline as best_dt
print("Decision Tree best parameters",study_dt.best_params)
# prediction with test data for model preformance
y_pred_dt = best_dt_pipeline.predict(Xtest)
y_pred_proba_dt=best_dt_pipeline.predict_proba(Xtest)[:,1]

acc_dt=accuracy_score(ytest, y_pred_dt)
f1_dt=f1_score(ytest, y_pred_dt)
rec_dt=recall_score(ytest, y_pred_dt)
pre_dt=precision_score(ytest, y_pred_dt)
roc_dt=roc_auc_score(ytest, y_pred_proba_dt)
cl_rep_dt=classification_report(ytest, y_pred_dt)
con_rep_dt=confusion_matrix(ytest, y_pred_dt)


modelperf_dt=pd.DataFrame([{
    "Model":"Decision Tree",
    "Accuracy":acc_dt,
    "f1_score":f1_dt,
    "recall":rec_dt,
    "precision":pre_dt,
    "f1score":f1_dt,
    "roc":roc_dt

}])
print(modelperf_dt)
print("########################### Decision tree completed ################################")

# rf hyper parameter tuning

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)

    pipeline =create_pipe(model)
    score = cross_val_score(
        pipeline, Xtrain, ytrain.iloc[:,0], # Ensure ytrain is a 1D array/Series
        cv=5, scoring="recall"
    ).mean()

    return score

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=25)

best_rf = RandomForestClassifier(**study_rf.best_params, class_weight="balanced")
best_rf_pipeline = create_pipe(best_rf)
best_rf_pipeline.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
best_rf = best_rf_pipeline # Assign the fitted pipeline as best_rf
print("Random Forest best parameters",study_rf.best_params)
# prediction with test data for model preformance
y_pred_rf = best_rf_pipeline.predict(Xtest)
y_pred_proba_rf=best_rf_pipeline.predict_proba(Xtest)[:,1]

acc_rf=accuracy_score(ytest, y_pred_rf)
f1_rf=f1_score(ytest, y_pred_rf)
rec_rf=recall_score(ytest, y_pred_rf)
pre_rf=precision_score(ytest, y_pred_rf)
roc_rf=roc_auc_score(ytest, y_pred_proba_rf)
cl_rep_rf=classification_report(ytest, y_pred_rf)
con_rep_rr=confusion_matrix(ytest, y_pred_rf)

modelperf_rf=pd.DataFrame([{
    "Model":"Random Forest",
    "Accuracy":acc_rf,
    "f1_score":f1_rf,
    "recall":rec_rf,
    "precision":pre_rf,
    "f1score":f1_rf,
    "roc":roc_rf

}])
print(modelperf_rf)

print("########################### RandomForest  completed ################################")

# XGB optuna hyperparameter tuning


def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**params)

    pipeline =create_pipe(model)
    score = cross_val_score(
        pipeline, Xtrain, ytrain.iloc[:,0], # Ensure ytrain is a 1D array/Series
        cv=5, scoring="recall"
    ).mean()

    return score

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=25)

best_xgb = XGBClassifier(**study_xgb.best_params)
best_xgb_pipeline = create_pipe(best_xgb)
best_xgb_pipeline.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
best_xgb = best_xgb_pipeline # Assign the fitted pipeline as best_xgb
print("XGBoost best parameters",study_xgb.best_params)
# prediction with test data for model preformance
y_pred_xgb= best_xgb_pipeline.predict(Xtest)
y_pred_proba_xgb=best_xgb_pipeline.predict_proba(Xtest)[:,1]

acc_xgb=accuracy_score(ytest, y_pred_xgb)
f1_xgb=f1_score(ytest, y_pred_xgb)
rec_xgb=recall_score(ytest, y_pred_xgb)
pre_xgb=precision_score(ytest, y_pred_xgb)
roc_xgb=roc_auc_score(ytest, y_pred_proba_xgb)
cl_rep_xgb=classification_report(ytest, y_pred_xgb)
con_rep_xgb=confusion_matrix(ytest, y_pred_xgb)

modelperf_xgb=pd.DataFrame([{
    "Model":"XGBoost",
    "Accuracy":acc_xgb,
    "f1_score":f1_xgb,
    "recall":rec_xgb,
    "precision":pre_xgb,
    "f1score":f1_xgb,
    "roc":roc_xgb

}])
print(modelperf_xgb)

print("########################### XGboost completed completed ################################")


# voting model
voting_model = VotingClassifier(
    estimators=[
        ("dt", best_dt),
        ("rf", best_rf),
        ("xgb", best_xgb)
    ],
    voting="soft",
    weights=[1, 2, 3]
)

voting_model.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
print("########################### voting  completed ################################")
print("voting score")
# Iterate through estimators to predict and print probabilities
for name, model in voting_model.named_estimators_.items():
  # The estimator in VotingClassifier is the entire pipeline
  # We need to access the actual model within the pipeline for prediction if it's not the final step.
  # However, for voting, the pipeline itself should have a predict_proba method if voting='soft'.
  # Xtest is processed by the full pipeline of the base estimator
  probs = model.predict_proba(Xtest)[:,1]
  print(name,probs)
#evaluation
from sklearn.metrics import classification_report
y_pred = voting_model.predict(Xtest)
acc=accuracy_score(ytest, y_pred)
f1=f1_score(ytest, y_pred,pos_label=1)
rec=recall_score(ytest, y_pred,pos_label=1)
pre=precision_score(ytest, y_pred,pos_label=1)
roc=roc_auc_score(ytest, y_pred)

pref_df=pd.DataFrame([{
    "Accuracy":acc,
    "f1_score":f1,
    "recall":rec,
    "precision":pre
    ,"roc_auc":roc
}])
print("performance\n",pref_df)


stack_model = StackingClassifier(
    estimators=[
        ("dt", best_dt),
        ("rf",best_rf),
        ("xgb",best_xgb)
    ],
    final_estimator=LogisticRegression(),
    passthrough=False,
    cv=5,
    verbose=1
)

stack_model.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
print("########################### stacking  completed ################################")
# prediction with test data for model preformance
y_pred = stack_model.predict(Xtest)
y_pred_proba=stack_model.predict_proba(Xtest)[:,1]

acc=accuracy_score(ytest, y_pred)
f1=f1_score(ytest, y_pred)
rec=recall_score(ytest, y_pred)
pre=precision_score(ytest, y_pred)
roc=roc_auc_score(ytest, y_pred_proba)
cl_rep=classification_report(ytest, y_pred)
con_rep=confusion_matrix(ytest, y_pred)
f1_scr=f1_score(ytest, y_pred)

print("accuracy score",acc)
print("f1 score",f1)
print("recall score",rec)
print("precision score",pre)
print("roc auc score",roc)
print("\n classification_report\n", cl_rep)
print("\nconfusion_matrix\n", con_rep)
print("f1_score",f1_scr)

co_eff=pd.DataFrame(
    stack_model.final_estimator_.coef_,
    columns= [ name for name, _ in stack_model.estimators]
)
print("stack estimator co-err \n",co_eff)

# comparing voiting and stacking
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring={
    "accuracy":"accuracy",
    "f1":"f1",
    "recall":"recall",
    "precision":"precision",
    "roc_auc":"roc_auc"
}
# comparing both voting and stacking through CV and scoring on 5 metrices
vote_cv=cross_validate(voting_model,Xtrain,ytrain.iloc[:,0],cv=cv,scoring=scoring)
stack_cv=cross_validate(stack_model,Xtrain,ytrain.iloc[:,0],cv=cv,scoring=scoring)

results= pd.DataFrame({
    "voting":{
        k: np.mean(vote_cv[f"test_{k}"]) for k in scoring
    },
    "stacking":{
        k: np.mean(stack_cv[f"test_{k}"]) for k in scoring
}}
)

# printing the model results against each indiviual model
print("model evaluation results \n",results)

# primary - recalll , secondary - f1 , tie-break - ,roc-auc, higher score model selected for final deployment
best_model = stack_model if results.loc["recall","stacking"]>results.loc["recall","voting"] else voting_model
best_model_name = "Stacking" if results.loc["recall","stacking"]>results.loc["recall","voting"] else "Voting"

best_model.fit(Xtrain,ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
y_pred=best_model.predict(Xtest)
y_prob=best_model.predict_proba(Xtest)[:,1]
print("selected model: ",best_model_name)
# getting the best model parameters for furture deployment
params=best_model.get_params()
pd.DataFrame(params.items(),columns=['parameter','value'])
for name,model in best_model.named_estimators_.items():
  print(f"\n * Base model - {name}")
  pprint(model.get_params())

# printing the model performance (FP / FN evaluation)
print("best slected model | classification report \n",classification_report(ytest, y_pred))
print("best slected model | confusion matrix \n",confusion_matrix(ytest, y_pred))

### model concludion of feature importance
best_xgb_pipeline.fit(Xtrain, ytrain.iloc[:,0]) # Ensure ytrain is a 1D array/Series
# Corrected: Access the actual XGBoost model from the pipeline
xgb_mdl=best_xgb_pipeline.named_steps["model"]

# Corrected: Transform Xtrain through the pipeline up to the scaler
Xtrain_transformed_df = best_xgb_pipeline.named_steps["feat"].transform(Xtrain) # Feat outputs DF
Xtrain_transformed_df = best_xgb_pipeline.named_steps["impute"].transform(Xtrain_transformed_df)
Xtrain_transformed_df = best_xgb_pipeline.named_steps["outlier"].transform(Xtrain_transformed_df)
Xtrain_transformed = best_xgb_pipeline.named_steps["scale"].transform(Xtrain_transformed_df) # Scaler outputs numpy

# Corrected: Generate feature names explicitly after FeatureEngineer and other steps
def get_feature_names(original_cols):
    feature_names = original_cols[:]
    for col in original_cols:
        feature_names.append(f"{col}_diff")
    for col in original_cols:
        feature_names.append(f"{col}_roll5")
    for col in original_cols:
        feature_names.append(f"{col}_anom")
    feature_names.append("temp_gap")
    feature_names.append("pressure_sum")
    return feature_names

original_feature_cols = Xtrain.columns.tolist()
fea_name = get_feature_names(original_feature_cols)

explain=shap.TreeExplainer(xgb_mdl)
shap_values=explain.shap_values(Xtrain_transformed)

# For summary_plot, it's better to pass the transformed data if shap_values were computed on it
shap.summary_plot(shap_values,
                    pd.DataFrame(Xtrain_transformed, columns=fea_name), # Pass as DataFrame with names
                    feature_names=fea_name)

## summary SHAP plot
shap.summary_plot(shap_values,
                    pd.DataFrame(Xtrain_transformed, columns=fea_name), # Pass as DataFrame with names
                    feature_names=fea_name,
                    plot_type="bar",
                    show=False)
ax= plt.gca()
for p in ax.patches:
    ax.text(
        p.get_width(),
        p.get_y()+p.get_height()/2,
        f"{p.get_width():.2f}",
        va="center",
    )
plt.show()


# Save the model locally
model_path = "Breakdown_prediction/best_engine_PM_prediction_v1.joblib"
joblib.dump(best_model, model_path,compress=("lzma",9))# job lfile > 110 NB |reduce to 20~40 MB

# Log the model artifact
#mlflow.log_artifact(model_path, artifact_path="model")
#print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "sudhirpgcmma02/Engine_PM"
repo_type = "model"

# Step 1: Check if the space exists
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
     path_or_fileobj="Breakdown_prediction/best_engine_PM_prediction_v1.joblib",
     path_in_repo="Breakdown_prediction/best_engine_PM_prediction_v1.joblib",
     repo_id=repo_id,
     repo_type=repo_type,
)
