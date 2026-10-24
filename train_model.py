# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING NIFTYJR STOCK PREDICTION MODEL")
print("="*60)

# 1. Load Data
df = pd.read_csv("NSE_Daily_indices_data_3000.csv")
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d-%m-%Y')
df = df.sort_values('DATETIME').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")

# 2. Preprocessing
df['PREVIOUS_CLOSE'] = df['PREVIOUS_CLOSE'].bfill()
df['CHANGE'] = df['CHANGE'].fillna(0)
df['CHANGE_PERCENT'] = df['CHANGE_PERCENT'].fillna(0)

# Target: Next day direction
df['Target'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)
df = df.dropna(subset=['Target'])

# Feature Engineering
df['Return'] = df['CLOSE'].pct_change()
df['SMA_5'] = df['CLOSE'].rolling(5).mean()
df['SMA_10'] = df['CLOSE'].rolling(10).mean()
df['SMA_20'] = df['CLOSE'].rolling(20).mean()
df['EMA_12'] = df['CLOSE'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['CLOSE'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Volatility'] = df['Return'].rolling(10).std()
df['Momentum'] = df['CLOSE'] - df['CLOSE'].shift(5)
df['HL_Spread'] = df['HIGH'] - df['LOW']
df['OC_Spread'] = df['CLOSE'] - df['OPEN']

for lag in [1, 2, 3, 5]:
    df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    df[f'Close_lag{lag}'] = df['CLOSE'].shift(lag)

delta = df['CLOSE'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df = df.dropna().reset_index(drop=True)
print(f"Shape after feature engineering: {df.shape}")
print(f"Target distribution:\n{df['Target'].value_counts()}")

# Features
feature_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'CHANGE', 'CHANGE_PERCENT',
                'Return', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
                'MACD', 'Volatility', 'Momentum', 'HL_Spread', 'OC_Spread',
                'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
                'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'RSI']

X = df[feature_cols]
y = df['Target']

# Time-based split
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=15)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print("Selected Features:", selected_features)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
}

results = []
best_acc = 0
best_model = None
best_name = ""

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc*100, 2),
        "Precision": round(prec*100, 2),
        "Recall": round(rec*100, 2),
        "F1-Score": round(f1*100, 2)
    })
    
    print(f"{name:22} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print("\n" + "="*60)
print(f"BEST MODEL: {best_name} with Accuracy = {best_acc*100:.2f}%")
print("="*60)

# Save everything
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(selected_features, "selected_features.pkl")
joblib.dump(feature_cols, "all_features.pkl")

pd.DataFrame(results).to_csv("model_comparison.csv", index=False)
print("\nSaved files: best_model.pkl, scaler.pkl, selector.pkl, selected_features.pkl, all_features.pkl, model_comparison.csv")
print("Training Completed Successfully!")
