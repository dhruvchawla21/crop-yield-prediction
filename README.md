# 🌾 AgriIntel Challenge — Predicting Agricultural Income

A full machine learning pipeline to predict **farmer agricultural income** (Total Income − Non-Agricultural Income) using seasonal scores, rainfall, land holdings, village infrastructure, and external market data.

---

## 📌 Problem Statement

Given a dataset of ~43,000 Indian farmers, predict each farmer's **total income** as:

```
Predicted Total Income = exp(predicted log_agri_income) + Non_Agriculture_Income
```

Model evaluation metric: **MAPE (Mean Absolute Percentage Error)** on total income.

---

## 🗂️ Project Structure

```
AgriIntel/
│
├── ML_project.ipynb              # Main notebook (all 4 phases)
├── train_data.csv              # Raw training data 
├── test_data.csv               # Raw test data 
│
├── train_processed.csv         # Output: cleaned + engineered features
├── train_processed_scaled.csv  # Output: scaled version for Ridge baseline
├── train_enriched.csv          # Output: with external MSP + productivity data
├── TeamName_predictions.csv    # Output: final submission file
│
└── figures/                    # All EDA and analysis plots (auto-generated)
    ├── fig1_target_distribution.png
    ├── fig2_land_distribution.png
    ├── fig3_agri_scores.png
    ├── fig4_skewness_table.png
    ├── fig5_income_region_state.png
    ├── fig6_income_demographics_village.png
    ├── fig7_income_by_soil.png
    ├── fig8_correlation_matrix.png
    ├── fig9_correlations_ranked.png
    ├── fig10_scatter_top_predictors.png
    ├── fig11_seasonal_trends.png
    ├── fig12_volatility_state.png
    ├── bonus_a_shap_intervention.png
    ├── bonus_a_shap_beeswarm.png
    ├── bonus_b_volatility_analysis.png
    ├── bonus_b_panel_trajectories.png
    └── bonus_b_credit_risk_state.png
```

---

## Dataset

Download the dataset from Google Drive:

https://drive.google.com/drive/u/0/folders/19Xxo4LCxDfBkRan-Sk2ku8BLJn1rTHnZ

## 🔄 Pipeline Overview

The notebook is organized into **4 sequential phases** + **2 bonus tasks**. Run the cells in order.

### Phase 1 — Exploratory Data Analysis
- Target variable analysis (raw + log-transformed distribution)
- Land holding distribution and outlier detection
- Kharif & Rabi agricultural score distributions (2020–2022)
- Skewness and kurtosis summary table
- Income breakdown by region, state, gender, village category, and soil type
- Correlation matrix across 22 numerical features
- Seasonal score trends and rainfall analysis
- Volatility vs income at state level

### Phase 2 — Data Preparation & Feature Engineering
- **Missing value imputation** (bureau features → 0 + flag; living index → district median; land → state median)
- **Outlier treatment** (land capped at 99th percentile; income: log transform handles it)
- **Categorical encoding** (ordinal for village categories; target encoding for State/Region; one-hot for gender/marital status; label encoding for soil type and agro-ecological zone)
- **Feature engineering**: score volatility, rainfall deltas, market access score, groundwater features, village infra composite score, land × score interactions, KCC credit stress
- **External data enrichment**: MSP (Minimum Support Price) by soil/crop type + state-level agricultural productivity index
- Outputs: `train_processed.csv`, `train_enriched.csv`

### Phase 3 — Modelling
| Model | Description |
|---|---|
| Ridge (baseline) | L2 regression on StandardScaler-normalized features |
| LightGBM | MAE objective, 5-fold CV, early stopping |
| XGBoost | Absolute error objective, 5-fold CV, early stopping |
| Stacked Ensemble | Ridge meta-learner on OOF predictions of all 3 base models |
| Weighted Average | Inverse-MAPE weighted blend of all 3 models |

- OOF target encoding for State, Region, District (no leakage)
- Zero-income farmers (225 rows) excluded from log-target training but flagged
- Final prediction: `exp(log_agri_pred) + non_agri_income`
- Output: `TeamName_predictions.csv`

### Bonus A — SHAP Intervention Recommender
- Identifies bottom 25% farmers by predicted income
- Computes SHAP values using a full-data LightGBM model
- Separates **controllable** features (land use, market access, credit) from **fixed** features (rainfall, geography, soil)
- Runs counterfactual analysis: shifting the top 2 controllable features to the 75th percentile of the training distribution
- Reports projected income uplift in ₹ and %

### Bonus B — Seasonal Volatility Modelling
- Reshapes wide farmer data into **panel format** (one row per farmer × season × year)
- Engineers volatility features: score std dev, coefficient of variation, drought proxy, kharif/rabi split
- Defines high-risk farmers as: high volatility + below-median income
- Demonstrates statistically significant negative correlation between volatility and income
- Produces state-level credit risk heatmap

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/dhruvchawla21/crop-yield-prediction


# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Untitled.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
lightgbm
xgboost
shap
jupyter
```

> Install all at once: `pip install -r requirements.txt`

---

## 🚀 How to Reproduce

1. Place `train_data.csv` and `test_data.csv` in the project root.
2. Open `ML_project.ipynb` in Jupyter.
3. Run all cells **in order** (Kernel → Restart & Run All).
4. All intermediate CSVs and figures will be saved automatically.
5. Final predictions are saved to `TeamName_predictions.csv`.

> ⚠️ The phases are **sequential** — each phase reads the CSV output from the previous one. Do not skip phases.

---

## 🔑 Key Features Engineered

| Feature | Description |
|---|---|
| `avg_kharif_score` / `avg_rabi_score` | Mean agricultural score across 3 years per season |
| `kharif_score_std` | Volatility in Kharif scores — credit risk proxy |
| `kharif_score_trend` | Score change from 2020 → 2022 (improvement indicator) |
| `market_access_score` | Inverse distance to nearest mandi + railway station |
| `village_infra_score` | Composite of electricity access, sanitation, pucca housing |
| `income_potential_score` | MSP × land × average Kharif score |
| `state_agri_productivity_idx` | External state-level normalised agri GDP per farmer |
| `land_x_kharif_score` | Interaction: land holding × Kharif productivity |
| `credit_stress` | % households without KCC credit limit of ₹50k |

---

## 📊 Model Results (5-Fold CV MAPE on Total Income)

| Model | CV MAPE |
|---|---|
| Ridge (baseline) | ~26.86% |
| LightGBM | ~23.98% |
| XGBoost | ~24.45% |
| Stacked Ensemble | ~24.31% |
| Weighted Average | ~24.38% |


---

## 📝 Notes

- The notebook uses `income_per_acre` only inside Phase 2 feature engineering; the final model feature list is defined separately in Phase 3's `preprocess()` function.
- Target encoding is done **out-of-fold** to prevent data leakage.
- 225 farmers with zero agricultural income are excluded from log-target training but retain a `zero_agri_income` flag so the model can distinguish them at inference.
- External MSP prices are sourced from [DACFW (Kharif 2022-23)](https://agricoop.nic.in/en/msp); the state productivity index is approximated from NABARD/MoA state reports.

---
