#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
import joblib


sns.set(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ### Config & Feature Lists

# In[ ]:


training_csv = "core_top_data.csv"     # training data file
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

pipeline_path = os.path.join(artifacts_dir, "maglearn_residual_pipeline.joblib")
paramT_csv_path = os.path.join(artifacts_dir, "core_top_with_paramT.csv")


RANDOM_STATE = 42


target_col = "t_annual"
num_features_residual = ["mgca", "omega_inv_sq"]      
cat_features_residual = ["species", "clean_method"]


PARAM_COEFFS = dict(a_mean=0.036, a_std=0.006,
                    b_mean=0.061, b_std=0.005,
                    c_mean=-0.73, c_std=0.07,
                    d_mean=0.0,   d_std=0.0)

N_MC = 1_000_000 


# ### Parametric T

# In[ ]:


def compute_parametric_T(df, coeffs, n_mc=1_000_000, seed=42):
    """
    Vectorized per-row Monte Carlo for parametric T:
      Mg/Ca = exp(a*(S-35) + b*T + c*(pH-8) + d)
      => T = [ln(MgCa) - a*(S-35) - c*(pH-8) - d] / b
    Returns: mean (np.ndarray), std (np.ndarray)
    """
    rng = np.random.default_rng(seed)
    a_s = rng.normal(coeffs["a_mean"], coeffs["a_std"], n_mc)
    b_s = rng.normal(coeffs["b_mean"], coeffs["b_std"], n_mc)
    c_s = rng.normal(coeffs["c_mean"], coeffs["c_std"], n_mc)
    d_s = rng.normal(coeffs["d_mean"], coeffs["d_std"], n_mc)

    means = np.empty(len(df), dtype=float)
    stds  = np.empty(len(df), dtype=float)

    for i, row in df.iterrows():
        mgca = row["mgca"]
        S    = row["s_annual"]
        pH   = row["pH"]

        T_samples = (np.log(mgca)
                     - a_s * (S - 35.0)
                     - c_s * (pH - 8.0)
                     - d_s) / b_s

        means[i] = T_samples.mean()
        stds[i]  = T_samples.std(ddof=0)

    return means, stds


# ### Custom Ensemble (Bootstrap Averaging of Base Models)

# In[ ]:


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Bootstraps the training set n_bootstrap times.
    """
    def __init__(self, models, n_bootstrap=500, random_state=42, save_bootstrap_predictions=False, bootstrap_csv_path=None):
        self.models = models
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.save_bootstrap_predictions = save_bootstrap_predictions
        self.bootstrap_csv_path = bootstrap_csv_path

    def fit(self, X, y):
        self.models_ = []
        rng = np.random.RandomState(self.random_state)

        for _ in tqdm(range(self.n_bootstrap), desc="Training"):
           
            X_b, y_b = resample(X, y, random_state=rng.randint(0, 1_000_000_000))
            fitted = [clone(m).fit(X_b, y_b) for m in self.models]
            self.models_.append(fitted)
        return self

    def predict(self, X):
        all_preds = []
        for fitted in tqdm(self.models_, desc="Predicting"):
            preds = np.column_stack([m.predict(X) for m in fitted]) 
            mean_per_boot = preds.mean(axis=1)                        
            all_preds.append(mean_per_boot)

        all_preds = np.vstack(all_preds)                
        mean_pred = all_preds.mean(axis=0)              
        sigma     = all_preds.std(axis=0)               
        lower     = mean_pred - sigma
        upper     = mean_pred + sigma

        if self.save_bootstrap_predictions and self.bootstrap_csv_path:
            pd.DataFrame(all_preds.T,
                         columns=[f"bootstrap_{i+1}" for i in range(self.n_bootstrap)]
                        ).to_csv(self.bootstrap_csv_path, index=False)

        return mean_pred, sigma, lower, upper


# ### Load Training Data

# In[ ]:


df = pd.read_csv(training_csv)

required_cols = {"mgca", "omega_inv_sq", "s_annual", "pH", "t_annual", "species", "clean_method"}
missing = required_cols - set(df.columns)
assert not missing, f"Missing required columns: {missing}"


assert (df["mgca"] > 0).all(), "Non-positive mgca values detected."


# ### Box-plots of Key Features

# In[ ]:


num_cols = ["mgca", "omega_inv_sq", "s_annual", "pH", "t_annual"]


melted = df[num_cols].melt(var_name="feature", value_name="value")

plt.figure(figsize=(12, 5))
ax = sns.boxplot(data=melted, x="feature", y="value")
ax.set_title("Box-plots of Numeric Features (Training)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()


# ### Compute Parametric T

# In[ ]:


param_T, param_T_unc = compute_parametric_T(df, PARAM_COEFFS, n_mc=N_MC, seed=RANDOM_STATE)

df["param_T"] = param_T
df["param_T_uncertainty"] = param_T_unc



residual = df[target_col] - param_T


# ### Train/Test Split & Preprocessor

# In[ ]:


X = df[num_features_residual + cat_features_residual]
y = residual
param_T_all = param_T  

X_train, X_test, y_train_res, y_test_res, param_T_train, param_T_test = train_test_split(
    X, y, param_T_all, test_size=0.30, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features_residual),
        ("cat", OrdinalEncoder(),  cat_features_residual)
    ]
)


# ### Define Ensemble Pipeline

# In[ ]:


rf = RandomForestRegressor(
    n_estimators=200, max_features="sqrt", max_depth=20,
    min_samples_split=2, min_samples_leaf=1, random_state=RANDOM_STATE
)

gbr = GradientBoostingRegressor(
    learning_rate=0.1, max_depth=4, max_features="sqrt", min_samples_split=2,
    min_samples_leaf=1, n_estimators=200, subsample=0.9, random_state=RANDOM_STATE
)

svr = SVR(C=100, degree=2, epsilon=0.1, gamma="scale", kernel="rbf")

ensemble = CustomEnsembleRegressor(
    models=[rf, gbr, svr],
    n_bootstrap=500,
    random_state=RANDOM_STATE,
    save_bootstrap_predictions=False, 
    bootstrap_csv_path=os.path.join(artifacts_dir, "bootstrap_predictions.csv")
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("ensemble", ensemble)
])


# ### Model fit

# In[ ]:


pipeline.fit(X_train, y_train_res)


# ### Predict

# In[ ]:


ytr_res_pred, ytr_sigma, ytr_low, ytr_up = pipeline.predict(X_train)
yte_res_pred, yte_sigma, yte_low, yte_up = pipeline.predict(X_test)

y_train_final = param_T_train + ytr_res_pred
y_test_final  = param_T_test  + yte_res_pred


# ###  Evaluate

# In[ ]:


rmse_train = np.sqrt(mean_squared_error(param_T_train + y_train_res, y_train_final))
r2_train   = r2_score(param_T_train + y_train_res, y_train_final)

rmse_test = np.sqrt(mean_squared_error(param_T_test + y_test_res, y_test_final))
r2_test   = r2_score(param_T_test + y_test_res, y_test_final)

print(f"Training RMSE: {rmse_train:.3f}")
print(f"Training R²:   {r2_train:.3f}")
print(f"Test RMSE:     {rmse_test:.3f}")
print(f"Test R²:       {r2_test:.3f}")


# ### Save fitted pipeline

# In[ ]:


# Save fitted pipeline for reuse in inference
joblib.dump(pipeline, pipeline_path)


with open(os.path.join(artifacts_dir, "metrics.txt"), "w") as f:
    f.write(f"Training RMSE: {rmse_train:.6f}\n")
    f.write(f"Training R2:   {r2_train:.6f}\n")
    f.write(f"Test RMSE:     {rmse_test:.6f}\n")
    f.write(f"Test R2:       {r2_test:.6f}\n")


# In[ ]:




