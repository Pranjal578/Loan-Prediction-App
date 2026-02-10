import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip().str.lower()

# print("Original shape:", data.shape)
# print("Columns:", data.columns.tolist())
# print("\nFirst few rows:\n", data.head())

# Right after loading the CSV and fixing column names, add:
data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip().str.lower()

# Strip spaces from all string columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Encode target FIRST
# print("Unique loan_status values:", data['loan_status'].unique())
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop rows where target is missing (do this early)
data = data.dropna(subset=['loan_status'])
# print("Shape after dropping missing loan_status:", data.shape)

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

# Encode other common categorical columns in loan datasets
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

# print("\nData types after encoding:\n", data.dtypes)
# print("Shape before dropping NaN:", data.shape)

# Drop any remaining rows with NaN (should be very few or none)
data = data.dropna()
# print("Shape after dropping all NaN:", data.shape)

# Check if we have data left
if len(data) == 0:
    # print("ERROR: No data left after preprocessing!")
    # print("Check your CSV file and column values")
    exit()

# Split input and output
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# print("\nX shape:", X.shape)
# print("y shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "loan_model.pkl")

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel trained successfully!")
print(f"Test accuracy: {accuracy:.4f}")