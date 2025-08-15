#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample
from tqdm import tqdm



artifacts_dir = "artifacts"
pipeline_path = os.path.join(artifacts_dir, "maglearn_residual_pipeline.joblib")


RANDOM_STATE = 42


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


# ### Load Fitted Pipeline

# In[ ]:


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models, n_bootstrap=500, random_state=42,
                 save_bootstrap_predictions=False, bootstrap_csv_path=None):
        self.models = models
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.save_bootstrap_predictions = save_bootstrap_predictions
        self.bootstrap_csv_path = bootstrap_csv_path

    def fit(self, X, y):
        self.models_ = []
        rng = np.random.RandomState(self.random_state)
        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrapping"):
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


import __main__
__main__.CustomEnsembleRegressor = CustomEnsembleRegressor


pipeline = joblib.load(os.path.join("artifacts","maglearn_residual_pipeline.joblib"))


# ### Load New Data & Compute Parametric T

# In[ ]:


# Replace with your file
new_csv = "tr163.csv"
new_data = pd.read_csv(new_csv)

req_cols = {"mgca", "omega_inv_sq", "s_annual", "pH", "species", "clean_method"}
missing = req_cols - set(new_data.columns)
assert not missing, f"Missing required columns in new data: {missing}"

param_T_new, param_T_new_unc = compute_parametric_T(new_data, PARAM_COEFFS, n_mc=N_MC, seed=RANDOM_STATE)


# ### Predict and save outputs

# In[ ]:


X_new = new_data[num_features_residual + cat_features_residual]

resid_mean, resid_sigma, resid_low, resid_up = pipeline.predict(X_new)

final_mean_T = param_T_new + resid_mean


total_sigma = np.sqrt(param_T_new_unc**2 + resid_sigma**2)
lower_total = final_mean_T - total_sigma
upper_total = final_mean_T + total_sigma


# In[ ]:


summary_df = pd.DataFrame({
    "Predicted_T": final_mean_T,
    "Predicted_T_1SD_Lower": lower_total,   
    "Predicted_T_1SD_Upper": upper_total,  
})


summary_df = summary_df.round(3)


out_xlsx = "predictions.xlsx"
with pd.ExcelWriter(out_xlsx) as writer:
    summary_df.to_excel(writer, sheet_name="T_and_bounds", index=False)

print(f"Wrote {out_xlsx} with columns: {', '.join(summary_df.columns)}")


# In[ ]:




