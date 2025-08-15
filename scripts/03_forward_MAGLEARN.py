#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample


# ### Config and paths

# In[ ]:


training_csv = "core_top_data.csv"     # training data
new_data_csv = "core_top_data.csv"     # File path for new data


pred_xlsx = "forward_predictions.xlsx"


RANDOM_STATE = 42


target = "mgca"
numerical_features = ["pH", "t_annual", "s_annual", "omega_inv_sq"]
categorical_features = ["species", "clean_method"]


N_BOOTSTRAP = 500


# ### Ensemble Regressor

# In[ ]:


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Trains N bootstrap replicates
    """
    def __init__(self, models, n_bootstrap=500, random_state=42):
        self.models = models
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = []
        rng = np.random.RandomState(self.random_state)
        for _ in tqdm(range(self.n_bootstrap), desc="Training"):
            X_b, y_b = resample(X, y, random_state=rng.randint(0, 1_000_000_000))
            fitted = [clone(m).fit(X_b, y_b) for m in self.models]
            self.models_.append(fitted)
        return self

    def predict(self, X):
        all_means = []
        for fitted in tqdm(self.models_, desc="Predicting"):
            preds = np.column_stack([m.predict(X) for m in fitted])  
            all_means.append(preds.mean(axis=1))                     
        all_means = np.vstack(all_means)                             
        mean_pred = all_means.mean(axis=0)
        std_pred  = all_means.std(axis=0)
        lower     = mean_pred - std_pred
        upper     = mean_pred + std_pred
        return mean_pred, std_pred, lower, upper


# ### Preprocessor and pipeline

# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
    ]
)

rf = RandomForestRegressor(
    n_estimators=100, max_features="sqrt", max_depth=None,
    min_samples_split=2, min_samples_leaf=1, random_state=RANDOM_STATE
)

gbr = GradientBoostingRegressor(
    learning_rate=0.1, max_depth=4, max_features="sqrt",
    min_samples_split=2, min_samples_leaf=1,
    n_estimators=200, subsample=0.9, random_state=RANDOM_STATE
)

svr = SVR(C=10, degree=2, epsilon=0.1, gamma="scale", kernel="rbf")

ensemble = CustomEnsembleRegressor(
    models=[rf, gbr, svr],
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("ensemble", ensemble),
])


# ### Training data and Train/Test split

# In[ ]:


df = pd.read_csv(training_csv)

required = set(numerical_features + categorical_features + [target])
missing = required - set(df.columns)
assert not missing, f"Missing required columns: {missing}"

X = df[numerical_features + categorical_features]
y = df[target].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)


# ### Fit

# In[ ]:


pipeline.fit(X_train, y_train)


# ### Predictions

# In[ ]:


new_data = pd.read_csv(new_data_csv)

req_pred = set(numerical_features + categorical_features)
missing_pred = req_pred - set(new_data.columns)
assert not missing_pred, f"Missing columns in new data: {missing_pred}"

X_new = new_data[numerical_features + categorical_features]
pred_mean, pred_sd, lower, upper = pipeline.predict(X_new)


out_df = pd.DataFrame({
    "Predicted_MgCa": pred_mean,
    "Predicted_MgCa_1SD_Lower": lower,
    "Predicted_MgCa_1SD_Upper": upper,
})


out_df = out_df.round(3)


with pd.ExcelWriter(pred_xlsx) as writer:
    out_df.to_excel(writer, sheet_name="Forward_Predictions", index=False)

print(f"Wrote {pred_xlsx} with columns: {list(out_df.columns)}")


# In[ ]:




