# ✈️ Airline Passenger Satisfaction Predictor

A machine learning web application that predicts whether an airline passenger will be **satisfied** or **neutral/dissatisfied** based on their flight experience, service ratings, and demographics.

🔗 **Live App:** https://rohitkathe76-del-airline-satisfaction-app-app-wetpk3.streamlit.app/

## 📌 Overview

Airlines want to know what drives passenger satisfaction so they can prioritize investment in the right areas. This project builds and compares 7 classification models to predict satisfaction, identifies the best-performing model through systematic hyperparameter tuning, and deploys it as an interactive web app.

## 📊 Dataset

- **Source:** [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets)
- **Size:** ~103,904 rows, 24 features
- **Target:** `satisfaction` (satisfied / neutral or dissatisfied)
- **Features:** Passenger demographics, trip details, flight delays, and 14 service quality ratings (0–5 scale)

## 🧠 Models Compared

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **LightGBM (Final)** | **96.4%** | **0.995** |
| XGBoost | 96.3% | 0.995 |
| Random Forest | 96.0% | 0.993 |
| Decision Tree | 95.2% | 0.983 |
| KNN | 93.9% | 0.980 |
| AdaBoost | 88.1% | 0.957 |
| Logistic Regression | 87.7% | 0.927 |

## ⚙️ Methodology

1. **EDA** — explored satisfaction patterns across travel class, customer type, delays, and service ratings
2. **Data Cleaning** — handled missing values, removed non-predictive columns
3. **Feature Engineering** — one-hot encoding for categorical features, scaling for distance/linear-based models
4. **Model Training** — compared 7 classification algorithms across linear, distance-based, tree, and boosting families
5. **Hyperparameter Tuning** — two-stage tuning (RandomizedSearchCV → focused GridSearchCV) on top-performing models
6. **Evaluation** — Accuracy, ROC-AUC, PR-AUC, F1-score, and train/test gap analysis to confirm generalization
7. **Deployment** — final LightGBM model served via a Streamlit web app

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, LightGBM, XGBoost
- **Data Handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit, Streamlit Community Cloud

## 🚀 Running Locally

```bash
git clone https://github.com/rohitkathe76/airline-satisfaction-app.git
cd airline-satisfaction-app
pip install -r requirements.txt
streamlit run app.py
```


## 📈 Results

The final tuned **LightGBM** model achieved **96.4% test accuracy** with only a ~2% train-test gap, confirming strong generalization rather than overfitting. Service quality ratings (wifi, boarding process, seat comfort) emerged as the strongest predictors of overall satisfaction.

## 🔮 Future Improvements

- SHAP-based feature importance for deeper interpretability
- Model monitoring for data drift in production
- A/B testing framework to validate business impact of predictions

