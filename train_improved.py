# ============================================================
# IMPROVED NIFTYJR STOCK PREDICTION - Higher Performance
# Models: Decision Tree, Naive Bayes, Logistic Regression, SVM, Random Forest
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("="*70)
print("IMPROVED TRAINING - NIFTYJR STOCK DIRECTION PREDICTION")
print("="*70)

# -------------------- 1. LOAD DATA --------------------
df = pd.read_csv("NSE_Daily_indices_data_3000.csv")
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d-%m-%Y')
df = df.sort_values('DATETIME').reset_index(drop=True)
print(f"Original shape: {df.shape}")
print(f"Date range: {df['DATETIME'].min().date()} → {df['DATETIME'].max().date()}")

# -------------------- 2. PREPROCESSING --------------------
df['PREVIOUS_CLOSE'] = df['PREVIOUS_CLOSE'].bfill()
df['CHANGE'] = df['CHANGE'].fillna(0)
df['CHANGE_PERCENT'] = df['CHANGE_PERCENT'].fillna(0)

# Target: next day direction
df['Target'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)

# -------------------- 3. STRONGER FEATURE ENGINEERING --------------------
# Returns & basic
df['Return'] = df['CLOSE'].pct_change()
df['Log_Return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))

# Moving averages
df['SMA_5']  = df['CLOSE'].rolling(5).mean()
df['SMA_10'] = df['CLOSE'].rolling(10).mean()
df['SMA_20'] = df['CLOSE'].rolling(20).mean()
df['SMA_50'] = df['CLOSE'].rolling(50).mean()
df['EMA_12'] = df['CLOSE'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['CLOSE'].ewm(span=26, adjust=False).mean()

# MACD
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# Bollinger Bands
df['BB_Mid'] = df['CLOSE'].rolling(20).mean()
df['BB_Std'] = df['CLOSE'].rolling(20).std()
df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
df['BB_Position'] = (df['CLOSE'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)

# RSI
delta = df['CLOSE'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-8)
df['RSI'] = 100 - (100 / (1 + rs))

# Stochastic Oscillator
low_14 = df['LOW'].rolling(14).min()
high_14 = df['HIGH'].rolling(14).max()
df['Stoch_K'] = 100 * (df['CLOSE'] - low_14) / (high_14 - low_14 + 1e-8)
df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

# Volatility & Momentum
df['Volatility'] = df['Return'].rolling(10).std()
df['Volatility_20'] = df['Return'].rolling(20).std()
df['Momentum_5'] = df['CLOSE'] - df['CLOSE'].shift(5)
df['Momentum_10'] = df['CLOSE'] - df['CLOSE'].shift(10)
df['ROC_5'] = df['CLOSE'].pct_change(5)
df['ROC_10'] = df['CLOSE'].pct_change(10)

# Price spreads
df['HL_Spread'] = (df['HIGH'] - df['LOW']) / df['CLOSE']
df['OC_Spread'] = (df['CLOSE'] - df['OPEN']) / df['CLOSE']
df['Gap'] = (df['OPEN'] - df['PREVIOUS_CLOSE']) / df['PREVIOUS_CLOSE']

# Lag features
for lag in [1, 2, 3, 5, 10]:
    df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)

# Moving average crossovers (binary signals)
df['SMA_5_10_cross'] = (df['SMA_5'] > df['SMA_10']).astype(int)
df['SMA_10_20_cross'] = (df['SMA_10'] > df['SMA_20']).astype(int)
df['Price_above_SMA20'] = (df['CLOSE'] > df['SMA_20']).astype(int)

df = df.dropna().reset_index(drop=True)
print(f"Shape after feature engineering: {df.shape}")
print(f"Target balance:\n{df['Target'].value_counts(normalize=True).round(3)*100}")

# -------------------- 4. FEATURES & SPLIT --------------------
exclude = ['DATETIME', 'SYMBOL', 'VOLUME', 'PREVIOUS_CLOSE', 'INTERVAL', 'GAP',
           'LOW_CLOSE', 'HIGH_CLOSE', 'TOTAL', 'Target']
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32']]

X = df[feature_cols]
y = df['Target']

print(f"Total features: {len(feature_cols)}")

# Time-based split (no shuffle)
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection - top 20
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected top 20 features: {selected_features}")

# -------------------- 5. MODELS WITH TUNING --------------------
print("\n" + "="*70)
print("TRAINING & EVALUATING MODELS (with hyperparameter tuning)")
print("="*70)

tscv = TimeSeriesSplit(n_splits=3)

# Decision Tree
dt_params = {'max_depth': [5, 8, 12], 'min_samples_leaf': [5, 10]}
dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=tscv, scoring='accuracy', n_jobs=-1)
dt.fit(X_train_sel, y_train)
best_dt = dt.best_estimator_

# Naive Bayes (no major params)
nb = GaussianNB()
nb.fit(X_train_sel, y_train)

# Logistic Regression
lr_params = {'C': [0.1, 1.0, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), lr_params, cv=tscv, scoring='accuracy', n_jobs=-1)
lr.fit(X_train_sel, y_train)
best_lr = lr.best_estimator_

# SVM
svm_params = {'C': [0.5, 1, 5], 'gamma': ['scale', 0.01]}
svm = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), svm_params, cv=tscv, scoring='accuracy', n_jobs=-1)
svm.fit(X_train_sel, y_train)
best_svm = svm.best_estimator_

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [6, 10], 'min_samples_leaf': [5]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=tscv, scoring='accuracy', n_jobs=-1)
rf.fit(X_train_sel, y_train)
best_rf = rf.best_estimator_

# Voting Ensemble (soft)
voting = VotingClassifier(
    estimators=[
        ('lr', best_lr),
        ('svm', best_svm),
        ('rf', best_rf),
        ('nb', nb)
    ],
    voting='soft'
)
voting.fit(X_train_sel, y_train)

models = {
    "Decision Tree": best_dt,
    "Naive Bayes": nb,
    "Logistic Regression": best_lr,
    "SVM": best_svm,
    "Random Forest": best_rf,
    "Voting Ensemble": voting
}

# -------------------- 6. EVALUATION --------------------
results = []
best_acc = 0
best_model = None
best_name = ""

print(f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*65)

for name, model in models.items():
    y_pred = model.predict(X_test_sel)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc*100, 2),
        "Precision": round(prec*100, 2),
        "Recall": round(rec*100, 2),
        "F1-Score": round(f1*100, 2)
    })
    
    print(f"{name:<22} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print("\n" + "="*70)
print(f"BEST MODEL: {best_name}  |  Accuracy = {best_acc*100:.2f}%")
print("="*70)

# -------------------- 7. SAVE ARTIFACTS --------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(selected_features, "selected_features.pkl")
joblib.dump(feature_cols, "all_features.pkl")
pd.DataFrame(results).to_csv("model_comparison.csv", index=False)

print("\nSaved: best_model.pkl, scaler.pkl, selector.pkl, selected_features.pkl, all_features.pkl, model_comparison.csv")
print("Improved training completed successfully!")
