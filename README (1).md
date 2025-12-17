# 🚗 Used Car Price Prediction – End-to-End Machine Learning Project

## 📌 Project Overview
This project focuses on building a **robust machine learning pipeline** to predict used car prices based on vehicle specifications such as age, mileage, brand, engine details, and other attributes.

The goal was not just to achieve high accuracy, but to follow an **industry-grade ML workflow**, including:
- Advanced preprocessing
- Model comparison
- Hyperparameter tuning
- Cross-validation
- Honest metric interpretation

---

## 🧠 Problem Statement
Used car prices are influenced by multiple non-linear factors such as:
- Vehicle age
- Mileage
- Manufacturer & model popularity
- Engine specifications
- Gear type and drivetrain

The challenge lies in handling:
- Mixed data types (numerical, categorical, high-cardinality)
- Skewed price distribution
- Noisy real-world data

---

## 📊 Dataset Description
The dataset contains vehicle-level information with the following key columns:

- **Numerical:** Price, Levy, Mileage, Cylinders, Airbags, Age  
- **Categorical:** Manufacturer, Model, Category, Fuel type, Gear box type, Drive wheels, Color  
- **Binary / Ordinal:** Leather interior, Wheel  

**Target Variable:**  
- `Price` (log-transformed during training)

---

## 🛠️ Feature Engineering & Preprocessing
All preprocessing was implemented using **`Pipeline`** and **`ColumnTransformer`** to avoid data leakage and ensure reproducibility.

### Preprocessing Strategy
- **Numerical Features:** Median imputation  
- **Categorical (Low Cardinality):** One-Hot Encoding  
- **High-Cardinality (Manufacturer, Model):** Frequency Encoding  
- **Binary & Ordinal Features:** Ordinal Encoding  
- **Target Transformation:** `log1p(Price)` to handle skewness  

---

## 🤖 Models Trained & Compared
- Decision Tree Regressor  
- Random Forest Regressor  
- Extra Trees Regressor  
- Gradient Boosting Regressor  
- HistGradientBoosting Regressor  
- XGBoost Regressor  
- Support Vector Regressor (SVR)  

---

## 🔍 Model Selection & Hyperparameter Tuning
- **RandomizedSearchCV** used for efficient hyperparameter exploration  
- Model selection based on **cross-validated R²**, not a single train–test split  

---

## 📊 Model Performance Comparison

Multiple regression models were trained and evaluated using a consistent preprocessing pipeline.  
Model selection was based on **5-fold cross-validated R²**, prioritizing generalization and stability over single train–test results.

| Model | Cross-Validated R² | Notes |
|------|-------------------|------|
| Decision Tree | ~0.40 | Baseline model, prone to overfitting |
| SVR (RBF Kernel) | ~0.44 | Captures non-linearity but sensitive to scaling |
| Gradient Boosting | ~0.34 | Underfit on this dataset |
| HistGradientBoosting | ~0.49 | Faster boosting, moderate performance |
| XGBoost | ~0.40 | Under-tuned, requires further optimization |
| Extra Trees | ~0.61 | Strong performance, slightly higher variance |
| **Random Forest (Final Model)** | **~0.66** | **Best generalization with low variance** |

**Final model selection** was based on:
- Highest cross-validated R²
- Low variance across folds
- Robust performance on noisy, real-world data

---
## 🏆 Final Model
### ✅ Random Forest Regressor

**Why Random Forest?**
- Highest cross-validated performance  
- Low variance across folds  
- Robust to noisy, real-world tabular data  

**Cross-Validation Performance:**
- Mean CV R²: ~0.66  
- Standard Deviation: ~0.01  

---

## 📈 Evaluation Metrics
This is a **regression problem**, so classification accuracy is not applicable.

### Metrics Used
- **R² Score:** Measures variance explained  
- **MAE (Mean Absolute Error):** Measures absolute price error  

> MAE appears relatively large due to the presence of low-priced, older vehicles where small feature deviations cause large absolute price differences. This is expected behavior in real-world pricing data.

Percentage-based metrics (raw MAPE) were avoided due to distortion from very low-priced vehicles.

---

## 🔎 Model Explainability
Feature importance analysis was performed using the Random Forest model to identify key price drivers such as:
- Vehicle age  
- Mileage  
- Manufacturer / model influence  
- Engine specifications  

---

## 💾 Model Artifact Note
The trained model file exceeds GitHub's 100MB limit due to:
- 800-tree Random Forest  
- Full preprocessing pipeline  

Therefore, the model file is **not stored directly** in the repository and can be recreated by running the training scripts.

---

## 🚀 Future Work & Deployment
### Planned Enhancements
- Error analysis by price segments  
- Further hyperparameter tuning  
- Enhanced feature importance visualization  

### Deployment (Planned)
- Deployment using **Streamlit**
- Interactive UI for price prediction  
- This section will be updated once deployed  

---

## 🧰 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Git & GitHub  

---

## 📌 Key Takeaways
- Built an end-to-end ML pipeline  
- Followed industry-standard validation practices  
- Prioritized stability and interpretability  

---

## 👤 Author
**Sumeet Kumar Pal**  
Aspiring Data Analyst / Machine Learning Enthusiast
