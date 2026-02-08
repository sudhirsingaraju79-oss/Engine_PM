from sklearn.base import BaseEstimator, TransformerMixin
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
