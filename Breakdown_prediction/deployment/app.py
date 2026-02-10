import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from featureengineer import FeatureEngineer
from outliercapper import OutlierCapper
def load_data():
  try:
    return pd.read_csv("data/engine_data.csv")
  except:
    return pd.DataFrame(columns=[
        "Engine rpm","Lub oil pressure","Fuel pressure","Coolant pressure","lub oil temp","Coolant temp","Engine condition"])


data=load_data()

#renaming columns for easy processing
data.columns = (data.columns
                   .str.strip()
                   .str.replace(" ","_")
                   .str.replace(r"[^\w]","_",regex=True)
                   .str.lower()
                   
  )

# -----------------------------
# Load Model
# -----------------------------
base_dir= os.path.dirname(__file__)
model_path= os.path.join(base_dir,"best_engine_PM_prediction_v1.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="Engine Condition Predictor", layout="centered")
st.title("🔧 Engine Health Monitoring System")
st.write("Enter the engine sensor values below to predict engine condition")

# ---- User Inputs ----
with st.form("engine_input_form"):

    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0,
        max_value=10000,
        value=1500,
        step=50
    )

    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure (bar)",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure (bar)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1
    )

    coolant_pressure = st.number_input(
        "Coolant Pressure (bar)",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.1
    )

    lub_oil_temp = st.number_input(
        "Lub Oil Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=85.0,
        step=1.0
    )

    coolant_temp = st.number_input(
        "Coolant Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=90.0,
        step=1.0
    )
    submit = st.form_submit_button("🚀 Predict Engine Condition")

# -----------------------------
# Predict Button
# -----------------------------
if submit:
  input_df = pd.DataFrame({
            "engine_rpm": [engine_rpm],
            "lub_oil_pressure": [lub_oil_pressure],
            "fuel_pressure": [fuel_pressure],
            "coolant_pressure": [coolant_pressure],
            "lub_oil_temp": [lub_oil_temp],
            "coolant_temp": [coolant_temp]
        })

  st.success("✅ Input captured successfully")
  st.write("### Input Data")
  st.dataframe(input_df)

  # Predict
  prediction = model.predict(input_df)[0]
  prob = model.predict_proba(input_df)[0][1]

  st.subheader("Prediction Result")

  if prob>=0.5:
      label="Maintenance Needed"
      st.warning(f"Engine needs Preventive maintenance. Probability: {prob:.2f}")
  else:
      label="Normal"
      st.success(f"Engine working normal. Probability: {prob:.2f}")

  # Save prediction to dataframe
  input_df['Engine_condition'] = label #'Normal / Preventive maintenance req '
  st.session_state['input_df'] = input_df
  st.dataframe(input_df)
  # -----------------------------
  # SAVE RECORDS SECTION
  # -----------------------------
  if st.button("Save Record"):
    if "input_df" in st.session_state:
      file_path = "records.csv"

      # If file exists → append
    if os.path.exists(file_path):
      existing_df = pd.read_csv(file_path)
      updated_df = pd.concat([existing_df, input_df], ignore_index=True)
    else:
      # Create new CSV
      updated_df = st.session_state['input_df']

      updated_df.to_csv(file_path, index=False)

    st.success("Record saved successfully!")

  else:
    st.error("Record not saved...Thank for analysis")

