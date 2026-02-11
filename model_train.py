import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip().str.lower()

# Strip spaces from all string columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Encode target FIRST
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop rows where target is missing
data = data.dropna(subset=['loan_status'])

# Drop ID column if it exists
if 'loan_id' in data.columns:
    data.drop('loan_id', axis=1, inplace=True)

# Encode categorical columns
if 'education' in data.columns:
    data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['education'] = data['education'].fillna(data['education'].median())

if 'self_employed' in data.columns:
    data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
    data['self_employed'] = data['self_employed'].fillna(data['self_employed'].median())

if 'gender' in data.columns:
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['gender'] = data['gender'].fillna(data['gender'].median())

if 'married' in data.columns:
    data['married'] = data['married'].map({'Yes': 1, 'No': 0})
    data['married'] = data['married'].fillna(data['married'].median())

if 'dependents' in data.columns:
    data['dependents'] = data['dependents'].replace('3+', 3)
    data['dependents'] = pd.to_numeric(data['dependents'], errors='coerce')
    data['dependents'] = data['dependents'].fillna(data['dependents'].median())

if 'property_area' in data.columns:
    data['property_area'] = data['property_area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    data['property_area'] = data['property_area'].fillna(data['property_area'].median())

# Fill numeric columns
num_cols = data.select_dtypes(include='number').columns
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Drop any remaining rows with NaN
data = data.dropna()

# Check if we have data left
if len(data) == 0:
    print("ERROR: No data left after preprocessing!")
    exit()

# Split input and output
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("="*70)

# Dictionary to store results
results = {}

# ==================== LOGISTIC REGRESSION ====================
print("\n📊 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
lr_pred = lr_model.predict(X_test)

# Metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

results['Logistic Regression'] = {
    'model': lr_model,
    'accuracy': lr_accuracy,
    'precision': lr_precision,
    'recall': lr_recall,
    'f1': lr_f1
}

print("\n📈 Logistic Regression Results:")
print(f"   Accuracy:  {lr_accuracy:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall:    {lr_recall:.4f}")
print(f"   F1-Score:  {lr_f1:.4f}")

# ==================== RANDOM FOREST ====================
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

results['Random Forest'] = {
    'model': rf_model,
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1': rf_f1
}

print("\n📈 Random Forest Results:")
print(f"   Accuracy:  {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall:    {rf_recall:.4f}")
print(f"   F1-Score:  {rf_f1:.4f}")

# ==================== XGBOOST ====================
print("\n🚀 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,      # Number of boosting rounds
    max_depth=6,           # Maximum depth of trees
    learning_rate=0.1,     # Step size shrinkage
    random_state=42,
    eval_metric='logloss'  # Suppress warning
)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_test)

# Metrics
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

results['XGBoost'] = {
    'model': xgb_model,
    'accuracy': xgb_accuracy,
    'precision': xgb_precision,
    'recall': xgb_recall,
    'f1': xgb_f1
}

print("\n📈 XGBoost Results:")
print(f"   Accuracy:  {xgb_accuracy:.4f}")
print(f"   Precision: {xgb_precision:.4f}")
print(f"   Recall:    {xgb_recall:.4f}")
print(f"   F1-Score:  {xgb_f1:.4f}")

# ==================== COMPARISON ====================
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)
print(f"{'Metric':<15} {'Logistic Reg':<18} {'Random Forest':<18} {'XGBoost':<18}")
print("-"*70)
print(f"{'Accuracy':<15} {lr_accuracy:<18.4f} {rf_accuracy:<18.4f} {xgb_accuracy:<18.4f}")
print(f"{'Precision':<15} {lr_precision:<18.4f} {rf_precision:<18.4f} {xgb_precision:<18.4f}")
print(f"{'Recall':<15} {lr_recall:<18.4f} {rf_recall:<18.4f} {xgb_recall:<18.4f}")
print(f"{'F1-Score':<15} {lr_f1:<18.4f} {rf_f1:<18.4f} {xgb_f1:<18.4f}")

# Find best model based on F1-score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_f1 = results[best_model_name]['f1']

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
print("="*70)

# Save all models
joblib.dump(lr_model, "loan_model_lr.pkl")
joblib.dump(rf_model, "loan_model_rf.pkl")
joblib.dump(xgb_model, "loan_model_xgb.pkl")
joblib.dump(best_model, "loan_model_best.pkl")

print("\n✅ Models saved:")
print("   - loan_model_lr.pkl (Logistic Regression)")
print("   - loan_model_rf.pkl (Random Forest)")
print("   - loan_model_xgb.pkl (XGBoost)")
print(f"   - loan_model_best.pkl ({best_model_name})")

# Feature Importance (only for tree-based models)
if best_model_name in ['Random Forest', 'XGBoost']:
    print(f"\n🔍 Top 5 Important Features ({best_model_name}):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']:<20} {row['importance']:.4f}")
else:
    # For Logistic Regression, show coefficients
    print(f"\n🔍 Top 5 Important Features (Coefficients):")
    feature_coef = pd.DataFrame({
        'feature': X.columns,
        'coefficient': abs(best_model.coef_[0])
    }).sort_values('coefficient', ascending=False)
    
    for idx, row in feature_coef.head(5).iterrows():
        print(f"   {row['feature']:<20} {row['coefficient']:.4f}")

print("\n" + "="*70)
print("✨ Training Complete!")
print("="*70)