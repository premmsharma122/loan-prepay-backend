import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# 1. Load data
df = pd.read_csv("loan_data.csv")

# 2. Create a synthetic 'prepaid_early'
df["prepaid_early"] = df["Total Received Interest"].apply(
    lambda x: 1 if x < 150 else 0
)

# 3. Select feature columns
feature_cols = [
    "Loan Amount",
    "Funded Amount",
    "Interest Rate",
    "Term",
    "Debit to Income",
    "Open Account",
    "Total Accounts",
    "Revolving Balance",
    "Revolving Utilities",
    "Total Received Interest",
    "Total Current Balance",
    "Total Revolving Credit Limit"
]

# 4. Remove missing rows for selected features
df = df.dropna(subset=feature_cols + ["prepaid_early"])

# 5. Prepare X and y
X = df[feature_cols]
y = df["prepaid_early"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 9. Evaluate
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", auc)

# 10. Save model, scaler, features
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "features.pkl")

print("Model training completed successfully!")
