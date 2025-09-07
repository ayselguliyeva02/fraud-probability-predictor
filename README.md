# 🚨 Fraud Detection Model

This project builds and optimizes machine learning models to detect fraudulent financial transactions. The dataset (~10K rows) includes transaction type, amount, balances, account type, time of day, and a fraud indicator (`isFraud`).

---

## 🔍 Workflow
1. **Data Preprocessing**
   - Removed identifiers and extracted date features.
   - Encoded categorical variables (`type`, `Acct type`, `Time of day`).
   - Verified no missing values.

2. **Model Training**
   - Baseline models: XGBoost, LightGBM, Random Forest, CatBoost.
   - Initial CatBoost + Optuna tuning → strong performance but ~0.10 train–test gap (overfitting).

3. **Feature Selection with SHAP**
   - Applied SHAP values to identify the most important predictors.
   - Reduced feature space to retain only impactful variables.

4. **Retraining & Tuning**
   - Re-trained with selected features.
   - Re-ran **Optuna hyperparameter tuning** on the reduced dataset.
   - Achieved a stable model with only **~5% train–test gap**.

5. **Deployment**
   - Final CatBoost (Optuna-tuned) model exported and deployed for fraud prediction.

---

## ✅ Final Model
- **CatBoost with Optuna tuning + SHAP-based feature selection**  
- Best generalization (only 5% train–test gap).  
- Robust performance and ready for deployment.  

---

## ⚙️ Tech Stack
- Python 3.8+
- pandas, numpy, scikit-learn
- catboost, xgboost, lightgbm, randomforest
- optuna, shap

---

pip install -r requirements.txt
python fraud_model.py
