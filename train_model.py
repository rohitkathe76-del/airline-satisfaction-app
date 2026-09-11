import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Load data ---
df = pd.read_csv('train.csv')
df = df.drop(columns=[col for col in ['Unnamed: 0', 'id'] if col in df.columns])

# --- Clean data ---
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
    df['Arrival Delay in Minutes'].median()
)
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# --- Encode categorical features ---
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Split features and target ---
X = df_encoded.drop(columns=['satisfaction'])
y = df_encoded['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale (saved for consistency, not required by LightGBM) ---
scaler = StandardScaler()
scaler.fit(X_train)

# --- Train final LightGBM model with your actual Round 2 best params ---
final_model = LGBMClassifier(
    n_estimators=400,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=30,
    subsample=0.8,
    random_state=42
)
final_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = final_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# --- Save model artifacts ---
joblib.dump(final_model, 'satisfaction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("\nModel and artifacts saved successfully!")
print("Columns used:", X.columns.tolist())