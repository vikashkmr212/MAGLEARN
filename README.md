# MAGLEARN — Calibration framework for Mg/Ca Paleothermometry

MAGLEARN is a global calibration scheme for Mg/Ca paleothermometry of planktic foraminifera employed in past climate research. MAGLEARN incorporates various non-thermal influences (seawater carbonate chemistry, salinity, dissolution, species identity, and shell cleaning methodology) in its forward and inverse implementations, and is validated on independent (out-of-sample) data that improve accuracy and generalizability.

This a repository for **Kumar et al. (2025)** manuscript in *Paleoceanography and Paleoclimatology* entitled **"MAGLEARN: A Global Calibration Framework for Mg/Ca Paleothermometry of Planktic Foraminifera Based on Machine Learning"**.

---

## Notebooks

```
notebooks/
├─ 01_downcore_train_MAGLEARN_inverse.ipynb
├─ 02_downcore_inference_MAGLEARN_inverse.ipynb
└─ 03_forward_MAGLEARN.ipynb
```

> **Important:** CSV column names must **exactly match** those used in the training and example files (case‑sensitive).

**Column name meanings (used across notebooks)**  
- `mgca` — shell Mg/Ca (mmol/mol)  
- `t_annual` — sea‑surface temperature (°C)  
- `s_annual` — surface salinity (psu)  
- `pH` — seawater pH 
- `omega_inv_sq` — 1/Ω² (calcite saturation state)  
- `species` — planktic foraminifera species code (`ruber_w`, `ruber_p`, `sacculifer`, `pachy`, `incompta`, `bulloides`)  
- `clean_method` — shell cleaning method category (reductive or non_reductive)

**01_downcore_train_MAGLEARN_inverse.ipynb**  
- **Purpose:** Train the inverse model to learn temperature from Mg/Ca and other predictors.  
- **Target variable:** `surface temperature`  
- **Input features (CSV column names):**  
  - Numerical: `mgca`, `omega_inv_sq`, `s_annual`, `pH`  
  - Categorical: `species`, `clean_method`  
- **Training data file:** `core_top_data.csv` (placed in `notebooks/`)  
- **Writes artifact:** `notebooks/artifacts/maglearn_residual_pipeline.joblib`

**02_downcore_inference_MAGLEARN_inverse.ipynb**  
- **Purpose:** Predict temperature on new downcore data using the trained inverse model.  
- **Inputs required (CSV column names):** `mgca`, `omega_inv_sq`, `s_annual`, `pH`, `species`, `clean_method`  
- **Example new‑data file for prediction:** `tr163.csv` (placed in `notebooks/`)  
- **Output file:** `predictions.xlsx` with three columns — `Predicted_T`, `Predicted_T_1SD_Lower`, `Predicted_T_1SD_Upper`

> **Important:** Make sure `notebooks/artifacts/maglearn_residual_pipeline.joblib` exists by running all the cells of **01_downcore_train_MAGLEARN_inverse.ipynb**  

**03_forward_MAGLEARN.ipynb**  
- **Purpose:** Train/apply the forward model to predict Mg/Ca from environmental and taxonomic predictors.  
- **Target column:** `mgca`  
- **Input features (CSV column names):**  
  - Numerical: `pH`, `t_annual`, `s_annual`, `omega_inv_sq`  
  - Categorical: `species`, `clean_method`  
- **Training data file:** `core_top_data.csv` (placed in `notebooks/`)  
- **Output file:** `forward_predictions.xlsx` with three columns — `Predicted_MgCa`, `Predicted_MgCa_1SD_Lower`, `Predicted_MgCa_1SD_Upper`

---

## Data

- Place input CSVs in **`notebooks/`** (same directory as the notebooks).  


## Environment

```txt
joblib==1.2.0
matplotlib==3.7.2
numpy==1.24.0
openpyxl==3.0.10
pandas==2.0.3
scikit-learn==1.3.2
seaborn==0.12.2
tqdm==4.65.0
```


---

## How to run

1) Run `01_downcore_train_MAGLEARN_inverse.ipynb` to train and write `artifacts/maglearn_residual_pipeline.joblib`.  
2) Run `02_downcore_inference_MAGLEARN_inverse.ipynb` to produce `predictions.xlsx`.  
3) Run `03_forward_MAGLEARN.ipynb` to produce `forward_predictions.xlsx`.

---

## Python scripts (alternative to notebooks)

Plain Python scripts placed in the `scripts/` folder can be used in place of notebooks

```
scripts/
├─ 01_downcore_train_inverse.py        # mirrors 01_ notebook
├─ 02_downcore_inference_inverse.py    # mirrors 02_ notebook
└─ 03_forward.py                       # mirrors 03_ notebook
```



## Outputs & artifacts

- **Artifacts (large file size)** is generated at runtime in `notebooks/artifacts/`.  
  - Main file: `maglearn_residual_pipeline.joblib`.
- **Excel outputs** are saved in the working directory (notebooks folder):
  - `predictions.xlsx` (inverse)
  - `forward_predictions.xlsx` (forward)


---


## Citation

If you use MAGLEARN, please cite the manuscript (in review):

> **Kumar, V.*** and **Tiwari, M.**, *MAGLEARN: A Global Calibration Framework for Mg/Ca Paleothermometry of Planktic Foraminifera Based on Machine Learning*, **Paleoceanography and Paleoclimatology**, in review.

