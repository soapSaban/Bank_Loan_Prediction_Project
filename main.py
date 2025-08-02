import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox

# ============================
# 1. Load and Preprocess Data
# ============================
file_path = "Loan_Granting_Binary_Classification.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df.drop(['Loan_ID', 'Customer_ID'], axis=1, inplace=True)

# Convert target variable
df['Loan_Status'] = df['Loan_Status'].map({'Fully Paid': 1, 'Charged Off': 0})

# Clean Monthly_Debt
if df['Monthly_Debt'].dtype == 'object':
    df['Monthly_Debt'] = df['Monthly_Debt'].str.replace(',', '').astype(float)

# Separate features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# 2. Train Random Forest Model
# ============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "loan_approval_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(num_cols, "num_cols.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")

# ============================
# Visualization Functions
# ============================
def plot_feature_importance():
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_credit_heatmap():
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.pivot_table(values='Loan_Status', index='Credit_Score', aggfunc='mean').fillna(0), cmap='coolwarm')
    plt.title('Credit Score vs Loan Status')
    plt.show()

# ============================
# 3. Tkinter GUI
# ============================
model = joblib.load("loan_approval_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

root = tk.Tk()
root.title("Loan Approval Prediction System")
root.geometry("700x600")
root.config(bg="#f0f0f0")

# Title
title = tk.Label(root, text="Loan Approval Prediction", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
title.pack(pady=10)

# Frame for form
form_frame = tk.Frame(root, bg="#f0f0f0")
form_frame.pack(pady=10)

entries = {}

for idx, col in enumerate(X.columns):
    lbl = tk.Label(form_frame, text=col, font=("Arial", 10), bg="#f0f0f0")
    lbl.grid(row=idx, column=0, sticky='w', padx=10, pady=5)
    if col in cat_cols:
        combo = ttk.Combobox(form_frame, values=sorted([str(v) for v in df[col].unique().tolist()]))
        combo.grid(row=idx, column=1, padx=10, pady=5)
        entries[col] = combo
    else:
        ent = tk.Entry(form_frame)
        ent.grid(row=idx, column=1, padx=10, pady=5)
        entries[col] = ent

def predict():
    try:
        input_data = {}
        for col in X.columns:
            val = entries[col].get()
            if col in cat_cols:
                val = label_encoders[col].transform([val])[0]
            else:
                val = float(val)
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]

        result_text = "Approved" if prediction == 1 else "Rejected"
        messagebox.showinfo("Prediction Result", f"Loan Status: {result_text}\nConfidence: {proba*100:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_feature_importance():
    plot_feature_importance()

def show_heatmap():
    plot_credit_heatmap()

# Buttons
btn_frame = tk.Frame(root, bg="#f0f0f0")
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Predict", font=("Arial", 12), bg="#4CAF50", fg="white", command=predict).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Feature Importance", font=("Arial", 12), bg="#2196F3", fg="white", command=show_feature_importance).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Credit Heatmap", font=("Arial", 12), bg="#FF5722", fg="white", command=show_heatmap).grid(row=0, column=2, padx=10)

root.mainloop()
