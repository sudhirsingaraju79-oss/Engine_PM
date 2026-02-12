import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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

        df.columns = (df.columns
                           .str.strip()
                           .str.replace(" ","_")
                           .str.replace(r"[^\w]","_",regex=True)
                           .str.lower()
        )

        core_sensor_cols = df.columns.tolist()
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

        df["temp_gap"] = df['lub_oil_temp'] - df['coolant_temp']   # oil vs coolant
        df["pressure_sum"] = df[['lub_oil_pressure','fuel_pressure', 'coolant_pressure']].sum(axis=1)

        df = df.fillna(0)

        # Return DataFrame with new column names for easier debugging and feature name extraction
        return df
