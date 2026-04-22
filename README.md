# AutoValuate — Used Car Price Prediction

> An end-to-end Machine Learning system that predicts used car prices using a tuned ensemble model, domain-driven feature engineering, and a production-ready Streamlit application.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-red)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Highlights](#project-highlights)
- [Project Architecture](#project-architecture)
- [Feature Engineering](#feature-engineering)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Performance Results](#performance-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Problem Statement

The used car market is highly volatile — pricing is influenced by dozens of interdependent variables including vehicle age, mileage intensity, fuel type, brand premium, and market demand. Traditional valuation methods rely on oversimplified heuristics, leading to inconsistent pricing that disadvantages both buyers and sellers.

**AutoValuate** addresses this by delivering a data-driven, ML-powered vehicle appraisal system that processes 16+ structured features, enriched through domain-specific engineering, and combines multiple high-performance models into a stable stacking ensemble.

---

## Project Highlights

- Built a fully modular, production-grade ML pipeline with custom exception handling and logging
- Implemented a custom `FeatureEngine` class (sklearn-compatible) with zero data leakage — all statistics learned exclusively from training data
- Used `StackingRegressor` with `ExtraTreesRegressor`, `LGBMRegressor`, and `CatBoostRegressor` as base learners and `Ridge` as the meta-learner
- Achieved **R² of 0.85+** on test data after ensemble tuning
- Deployed as an interactive Streamlit web application with real-time price prediction
- Followed industry-standard ML engineering practices — modular components, artifact management, and reproducible pipelines

---

## Project Architecture

```
Raw Data
    │
    ▼
Data Ingestion          — filters outliers, splits train/test, saves CSVs
    │
    ▼
Feature Engineering     — cleans data, engineers new features (no leakage)
    │
    ▼
Data Transformation     — encodes, imputes, saves full pipeline as preprocessor.pkl
    │
    ▼
Model Training          — trains StackingRegressor, evaluates, saves model.pkl
    │
    ▼
Streamlit App           — loads preprocessor.pkl + model.pkl, serves predictions
```

The pipeline is fully modular — each component is independent, testable, and replaceable.

---

## Feature Engineering

A custom sklearn-compatible `FeatureEngine` transformer was built to engineer domain-specific features while preventing data leakage. All statistics (medians, group means, valid manufacturers) are learned **only from training data** and applied to test/inference data.

| Feature | Logic | Purpose |
|---|---|---|
| `Mileage_Intensity` | `Mileage / (Age + 1)` | Captures usage rate — distinguishes high-use from low-use vehicles |
| `is_levy_zero` | `(Levy == 0).astype(int)` | Flags vehicles with no levy — proxy for vehicle category/exemption |
| `Engine Volume` | Renamed from `Engine_Volume_Num` | Cleaned numeric engine displacement |
| Levy imputation | Filled using **train-only** Age-group medians | Handles missing levy without leakage |
| Mileage imputation | Filled using **train-only** global mileage rate × Age | Fixes zero-mileage records realistically |
| Airbags imputation | Filled using **train-only** Manufacturer+Category medians | Realistic airbag count per vehicle class |

---

## Preprocessing Pipeline

A modular `ColumnTransformer` pipeline handles all encoding and imputation after feature engineering. The full pipeline (FeatureEngine + ColumnTransformer) is bundled into a single `preprocessor.pkl` for clean, order-safe inference.

| Feature Group | Columns | Transformer |
|---|---|---|
| Numerical | `Levy`, `Mileage`, `Age`, `Engine Volume`, `Mileage_Intensity`, `Airbags`, `is_levy_zero` | `SimpleImputer(median)` |
| One-Hot | `Category`, `Fuel type`, `Gear box type`, `Color`, `Inventory_Segment` | `OneHotEncoder` |
| Binary | `Leather interior`, `Turbo`, `Is_Premium_Brand` | `OrdinalEncoder` |
| High Cardinality | `Manufacturer` | `TargetEncoder(cv=3, target_type='continuous')` |

No scaling applied — tree-based models are scale-invariant.

---

## Model Architecture

The final model is a **Stacking Ensemble** combining three diverse base learners with a Ridge meta-learner:

```
ExtraTreesRegressor  ──┐
                       ├──► Ridge (meta-learner) ──► Final Price Prediction
LGBMRegressor        ──┤
                       │
CatBoostRegressor    ──┘
```

**Why Stacking?**
- Each base learner captures different patterns in the data
- Ridge meta-learner learns the optimal weighting of each model's predictions
- Reduces variance compared to any single model

**Hyperparameters** were tuned using Optuna.

---

## Performance Results

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.23 | ~$8,903 |
| Ridge Regression | 0.29 | ~$8,854 |
| Gradient Boosting | 0.54 | ~$6,485 |
| XGBoost (Tuned) | 0.67 | ~$4,942 |
| LightGBM | 0.71 | ~$5,052 |
| CatBoost | 0.72 | ~$4,926 |
| Random Forest | 0.73 | ~$4,092 |
| Extra Trees | 0.78 | ~$3,783 |
| **Stacking Ensemble (Final)** | **0.85+** | **~$3,200** |

The stacking ensemble outperforms all individual models, achieving the lowest MAE and highest R².

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn 1.5.2 |
| Boosting Models | CatBoost 1.2.7, LightGBM 4.6.0 |
| Data Processing | Pandas, NumPy |
| Serialization | Dill |
| Web App | Streamlit 1.37.1 |
| Logging | Python `logging` module |
| Version Control | Git & GitHub |

---

## Project Structure

```
Car Price Prediction/
│
├── artifacts/                  # Generated at runtime — not pushed to GitHub
│   ├── preprocessor.pkl        # FeatureEngine + ColumnTransformer bundled
│   ├── model.pkl               # Trained StackingRegressor
│   ├── train.csv
│   └── test.csv
│
├── Dataset/
│   └── Dataset.csv             # Raw dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads, filters, splits data
│   │   ├── feature_engineering.py  # Custom sklearn FeatureEngine transformer
│   │   ├── data_transformation.py  # Encoding, imputing, saves preprocessor.pkl
│   │   └── model_trainer.py        # Trains, evaluates, saves model.pkl
│   ├── exception.py            # Custom exception with file + line number tracking
│   ├── logger.py               # Timestamped logging
│   └── utils.py                # save_object / load_object using dill
│
├── predict_pipeline.py         # PredictPipeline + CustomData for inference
├── app.py                      # Streamlit web application
├── main.py                     # Runs full training pipeline
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sumeet-016/car-price-prediction-ml.git
cd car-price-prediction-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Pipeline
```bash
python main.py
```
This will generate `artifacts/preprocessor.pkl` and `artifacts/model.pkl`.

### 4. Launch the App
```bash
streamlit run app.py
```

> **Note:** Always run from the project root directory. Use Anaconda Prompt for best compatibility.

---

## Future Enhancements

- Feature importance visualization and SHAP explainability
- Error analysis segmented by price range and manufacturer
- Cloud deployment via AWS / GCP / Hugging Face Spaces
- Model monitoring and data drift detection
- CI/CD pipeline with automated retraining

---

## Author

**Sumeet Kumar Pal**
Aspiring Data Scientist | Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-sumeet--016-black?logo=github)](https://github.com/sumeet-016)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-palsumeet-blue?logo=linkedin)](https://www.linkedin.com/in/palsumeet)
