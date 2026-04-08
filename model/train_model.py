import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_excel("dataset/career_data.xlsx")

print("Original Shape:", df.shape)


# ==============================
# 2. CLEAN DATA
# ==============================

# Remove unnecessary columns
df = df.drop(columns=["Sr.No.", "Course", "Student", "s/p"], errors='ignore')

# Clean Job profession column
df["Job profession"] = df["Job profession"].astype(str).str.strip()

# Drop missing values
df = df.dropna()

print("After Cleaning Shape:", df.shape)


# ==============================
# 3. CONVERT BEST/AVG/POOR → NUMBERS
# ==============================

mapping = {
    "BEST": 2,
    "AVG": 1,
    "POOR": 0
}

performance_cols = ["P1","P2","P3","P4","P5","P6","P7","P8"]

for col in performance_cols:
    df[col] = df[col].map(mapping)

print("Converted Performance Columns ✅")


# ==============================
# 4. SPLIT INPUT / OUTPUT
# ==============================

X = df.drop("Job profession", axis=1)
y = df["Job profession"]


# ==============================
# 5. ENCODE TARGET
# ==============================

le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ==============================
# 6. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# ==============================
# 7. TRAIN MODEL (Random Forest)
# ==============================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained ✅")


# ==============================
# 8. CHECK ACCURACY
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ==============================
# 9. SAVE MODEL
# ==============================

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(le, open("model/encoder.pkl", "wb"))

print("Model Saved ✅")