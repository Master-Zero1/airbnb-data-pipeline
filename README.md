<div align="center">

# 🏙️ NYC Airbnb Price Prediction Pipeline

**End-to-end data engineering pipeline on 12M+ rows of NYC Airbnb data**  
*From raw messy CSVs → production-ready ML model in 7 notebooks*

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-red?style=flat-square)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🎯 What This Project Does

Takes three raw, messy Airbnb CSV files and builds a complete ML pipeline to predict nightly listing prices across New York City — cleaning 90 raw columns down to 43 engineered features, training three models, and achieving **76.1% R² with $189 average prediction error** on unseen data.

---

## 📊 Results

| Model | Val R² | Test R² | Avg Dollar Error |
|-------|--------|---------|-----------------|
| Ridge Baseline | 0.613 | — | $246 |
| Random Forest | 0.718 | — | $246 |
| **XGBoost ✓** | **0.769** | **0.761** | **$189** |

> Trained on 16,416 listings · Validated on 2,052 · Tested on 2,053  
> Stratified 80/10/10 split by NYC borough

---

## ⚠️ Leakage Identified and Fixed

During self-review, I identified **target leakage** in an earlier version of this project.

`price_per_person` and `price_per_bedroom` were engineered from the target variable `price` and incorrectly included in X — allowing the model to mathematically reconstruct the answer rather than learn genuine patterns.

**Before fix (inflated):** XGBoost R² = 0.991  
**After fix (honest):** XGBoost R² = 0.761

The fix was removing both columns from X before splitting. They remain in the processed dataset for reference but are excluded from model training. Identifying and fixing this independently demonstrates understanding of one of the most common and dangerous mistakes in real-world ML pipelines.

---

## 🗺️ Feature Importance

The top features driving price predictions after the leakage fix:

![Feature Importance](plots/feature_importance.png)

Key findings:
- `accommodates` — dominant signal, size drives price most
- `bedrooms` — second strongest
- `neighbourhood_cleansed` — location (target encoded, 200+ neighbourhoods)
- `room_type_Entire home/apt` — entire homes command significant premium
- `property_type` — hotel rooms and rental units priced differently

---

## 📈 Model Performance

![Actual vs Predicted](plots/06_actual_vs_predicted.png)

Points cluster tightly around the perfect prediction line especially in the $50–$250 range. Spread increases at higher prices — expected since luxury listings ($400+) are rarer and harder to generalize from.

---

## 🗂️ Project Structure

```
nyc-airbnb-price-predictor/
│
├── 📓 notebooks/
│   ├── 01_cleaning_listing_data.ipynb   # clean 90-column listings file
│   ├── 02_cleaning_reviews_data.ipynb   # aggregate 700k review rows
│   ├── 03_cleaning_calendar_data.ipynb  # aggregate 12M calendar rows
│   ├── 04_feature_engineering.ipynb     # merge, engineer features, outlier capping
│   ├── 05_ml_pipeline.ipynb             # train/val/test split (leakage fix here)
│   ├── 06_ml_encoding.ipynb             # encoding, scaling, train 3 models (Colab)
│   └── 07_eda_plots.ipynb               # all visualizations
│
├── 📊 plots/
│   ├── 01_price_distribution.png        # before/after log transform
│   ├── 02_price_by_borough.png          # price by NYC borough
│   ├── 03_price_by_room_type.png        # entire home vs private room
│   ├── 04_price_map.png                 # geographic price heatmap
│   ├── 05_correlation_heatmap.png       # feature correlations
│   ├── 06_actual_vs_predicted.png       # model performance
│   └── feature_importance.png           # XGBoost top 15 features
│
├── 🤖 models/
│   └── xgboost_airbnb_model.pkl         # trained XGBoost model (2MB)
│
├── 📁 data/
│   ├── raw/                             # original CSVs (not in repo)
│   ├── processed/                       # cleaned intermediate files
│   └── output/                          # final train/val/test splits
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔍 Pipeline Overview

```
Raw Data                    Cleaning                  Features
─────────                   ────────                  ────────
listings.csv.gz   ──►  drop 47 useless cols  ──►  host_experience_level
reviews.csv.gz    ──►  parse $1,200 → 1200   ──►  total_availability_score
calendar.csv.gz   ──►  fill nulls smartly    ──►  amenity binary flags (12)
                         fix t/f booleans         review_score_avg
35,036 listings   ──►  parse amenity JSON    ──►  days_since_last_review
90 raw columns         IQR outlier capping        neighbourhood target enc.
                                                        │
                                                        ▼
                                              20,521 clean rows
                                              39 training features
                                                        │
                                              ┌─────────┼─────────┐
                                           Train      Val       Test
                                          16,416     2,052     2,053
                                                        │
                                              XGBoost Regressor
                                                        │
                                              R² = 0.761
                                           $189 avg error
```

---

## 🛠️ Key Challenges Solved

**1. Price column entirely NULL in latest scrape**  
Inside Airbnb removed the static price column from recent datasets. Solved by extracting `price_per_night` from the `price_quote_raw` JSON blob in the new scrape format.

**2. 12M row calendar file**  
Aggregated per listing using `groupby().agg()` — reducing 12M rows to one row per listing with `availability_rate` and `total_days`.

**3. Amenities stored as JSON-like strings**  
`'["Wifi", "Kitchen", "TV"]'` parsed with `json.loads()` + loop to create 12 binary feature columns for high-signal amenities.

**4. neighbourhood_cleansed had 200+ unique values**  
One-hot encoding would create 200+ columns. Used **target encoding** — replaced each neighbourhood with its mean `log_price` computed from the train set only, with global mean fallback for unseen neighbourhoods.

**5. Target leakage identified and fixed**  
`price_per_person` and `price_per_bedroom` were derived from the target variable and incorrectly included in X. Identified during self-review, removed before splitting. See leakage section above.

**6. Train/val/test leakage prevention**  
All encoders and scalers fit exclusively on X_train, then applied to X_val and X_test. Target encoding computed from y_train only.

---

## 📦 Data

This project uses the [Inside Airbnb](http://insideairbnb.com/get-the-data/) dataset for **New York City** (April 2026 scrape).

To reproduce locally:

1. Go to http://insideairbnb.com/get-the-data/
2. Scroll to **New York City**
3. Download:
   - `listings.csv.gz` — Detailed listings data (~50MB)
   - `reviews.csv.gz` — Detailed review data
   - `calendar.csv.gz` — Daily availability data (~300MB)
4. Place all three in `data/raw/`

> Raw data files are not included in this repo due to file size.

---

## 🚀 How to Run

```bash
# 1. clone the repo
git clone https://github.com/master-zero1/nyc-airbnb-price-predictor.git
cd nyc-airbnb-price-predictor

# 2. install dependencies
pip install -r requirements.txt

# 3. download data (see above) and place in data/raw/

# 4. run notebooks in order (locally)
# 01 → 02 → 03 → 04 → 05

# 5. run 06_ml_encoding.ipynb in Google Colab for model training
# (upload data/output/ CSVs to Colab)
```

---

## 🔮 Use the Trained Model

```python
import joblib
import numpy as np

model = joblib.load('models/xgboost_airbnb_model.pkl')

# model outputs log_price — convert back to dollars
prediction = model.predict(X_encoded)
price_dollars = np.expm1(prediction)
print(f"Predicted price: ${price_dollars[0]:.2f}/night")
```

---

## 📚 Tech Stack

| Tool | Purpose |
|------|---------|
| Pandas | Data loading, cleaning, transformation |
| NumPy | Numerical operations, log transforms |
| Scikit-learn | Encoding, scaling, Ridge, Random Forest |
| XGBoost | Final production model |
| Matplotlib | All visualizations |
| Joblib | Model serialization |

---

## 📖 What I Learned

This project was my first complete end-to-end data pipeline. Key lessons:

- **Target leakage** — engineered features derived from the target variable inflate scores dramatically (0.991 → 0.761 after fix). Always audit feature origins before training.
- **IQR capping** vs dropping outliers — capping preserves data while limiting damage from extreme values
- **Flag before filling** — null values in review scores carry information (no reviews ≠ bad reviews)
- **Target encoding** for high-cardinality categoricals avoids the column explosion of one-hot encoding
- **Fit on train only** — the single most important rule in ML preprocessing. Fitting scalers/encoders on full data leaks test distribution into training.
- **log transform** on skewed targets dramatically improves model learning
- **Honest scores matter** — a 0.761 R² with no leakage is more credible and useful than 0.991 with leakage

---

## 🔭 What's Next

- [ ] Add 6 new engineered features (listing_age_years, reviews_per_year, bathrooms_per_guest etc.)
- [ ] Build a Streamlit web app for live price predictions
- [ ] Hyperparameter tuning with Optuna
- [ ] Add SHAP values for model explainability
- [ ] NLP sentiment analysis on review text

---

<div align="center">

**Built by [master-zero1](https://github.com/master-zero1)**  
*NYC Airbnb Data · April 2026 Scrape · XGBoost · R² = 0.761*

</div>
