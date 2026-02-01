# 🚗 AutoValuate: Advanced Ensemble Machine Learning for Used Car Pricing

## 📌 Project Overview & Problem Statement

The used car market is inherently volatile, with pricing driven by a complex interaction of vehicle condition, usage intensity, technical specifications, and market demand. Traditional valuation methods often rely on oversimplified heuristics, failing to capture the true wear and economic value of a vehicle. This leads to inconsistent pricing, creating inefficiencies for both buyers and sellers.

This project addresses that gap by delivering an advanced end-to-end Machine Learning system that predicts used car prices with high precision. The system processes 17+ structured variables, enhanced through domain-driven feature engineering, and combines multiple high-performance models into a tuned ensemble. The result is a production-ready, data-driven vehicle appraisal tool designed to reduce uncertainty and improve pricing transparency.

---

## 🧠 Smart Feature Engineering

To better capture real-world pricing behavior, the dataset was enriched with domain-specific engineered features:

- **Mileage Intensity**  
  `Mileage / (Age + 1)`  
  Differentiates high-usage vehicles from low-usage vehicles with identical mileage but different age profiles.

- **Log Target Transformation**  
  Training on `log1p(Price)` reduces skew in the target variable and improves performance on high-value vehicles.

- **Frequency Encoding**  
  High-cardinality categorical features (Manufacturer, Model) encoded by market frequency to avoid dimensional explosion.

These transformations allow the model to learn economically meaningful patterns.

---

## 🛠️ Pipeline & Preprocessing Architecture

A modular preprocessing pipeline was implemented using Scikit-Learn’s `ColumnTransformer`.

### Feature Groups

**Numerical**
- Age
- Mileage
- Engine Volume
- Levy
- Cylinders
- Airbags
- Mileage Intensity

**Categorical**
- Manufacturer
- Model
- Category
- Fuel Type
- Gearbox
- Drive Wheels
- Color

**Binary / Ordinal**
- Leather Interior
- Turbo Engine

Custom transformers ensure reproducibility and production compatibility.

---

## 🤖 Model Architecture: Tuned Ensemble System

The final prediction is generated using a **Voting Regressor Ensemble**, combining three optimized models:

1. **Extra Trees Regressor** – Handles noisy tabular data effectively  
2. **Random Forest Regressor** – Robust and stable baseline  
3. **CatBoost Regressor** – Learns complex categorical relationships

This ensemble balances bias and variance to produce stable predictions.

---

## 📊 Performance Evaluation

| Model | MAE | R² Score |
|------|------|------|
| Linear Regression | ~$8,903 | 0.23 |
| Lasso Regression | ~$11,375 | -0.137 |
| Ridge Regression | ~$8,854 | 0.285 |
| Gradient Boosting | ~$6,485 | 0.535 |
| XGBoost (Tuned) | ~$4,942 | 0.67 |
| LightGBM | ~$5,052 | 0.706 |
| CatBoost | ~$4,926 | 0.720 |
| Random Forest | ~$4,092 | 0.731 |
| Extra Trees | ~$3,783 | 0.776 |
| **Final Tuned Voting Ensemble** | **~$3,930** | **0.765** |

The ensemble achieves strong predictive accuracy while maintaining stability across price ranges.

---

## 💻 Deployment & Application Layer

The model is deployed through an interactive **Streamlit** web application featuring:

- Cascading dropdown filters (Manufacturer → Category → Model)
- Real-time prediction feedback
- Structured UI reflecting real vehicle listings

This simulates a production-grade automated pricing system.

---

## 🧰 Tech Stack

- Python 3.13  
- Pandas, NumPy  
- Scikit-Learn (v1.5.2+)  
- CatBoost, XGBoost  
- Streamlit  
- Joblib / Pickle  
- Git & GitHub  

A custom `CatBoostRegressorWrapper` resolves compatibility issues with modern Scikit-Learn versions.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/sumeet-016/Car-Price-Prediction.git
cd Car-Price-Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Enhancements

- Error analysis by price segments  
- Advanced hyperparameter tuning  
- Feature importance visualization  
- Model monitoring & drift detection  
- Cloud deployment pipeline  

---

## 📌 Key Takeaways

- Built a fully modular ML pipeline  
- Applied domain-driven feature engineering  
- Leveraged ensemble learning for stability  
- Designed a deployment-ready prediction system  
- Followed industry-standard validation practices  

---

## 👤 Author

**Sumeet Kumar Pal**  
Aspiring Data Analyst | Machine Learning Enthusiast  

GitHub: https://github.com/sumeet-016  
LinkedIn: https://www.linkedin.com/in/palsumeet  

---
