import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


# ============================================
# STEP 1: Load Dataset
# ============================================

data = pd.read_excel("dataset/dataset_project.xlsx")

print("Original Dataset Shape:", data.shape)


# ============================================
# STEP 2: Remove Unnecessary Columns
# ============================================

data = data.drop(columns=["Sr.No.", "Course", "Student", "s/p"])

print("After Dropping Columns:", data.shape)


# ============================================
# STEP 3: Remove Missing Values
# ============================================

data = data.dropna()

print("After Removing Missing Values:", data.shape)


# ============================================
# STEP 4: Clean Text Columns
# ============================================

# remove extra spaces and newline from job profession
data["Job profession"] = data["Job profession"].str.strip()


# ============================================
# STEP 5: Convert BEST / AVG / POOR → Numbers
# ============================================

performance_map = {
    "POOR": 0,
    "AVG": 1,
    "BEST": 2
}

for col in ["P1","P2","P3","P4","P5","P6","P7","P8"]:
    data[col] = data[col].map(performance_map)


print("\nEncoded Dataset Preview:\n")
print(data.head())


# ============================================
# STEP 6: Separate Features and Target
# ============================================

X = data.drop("Job profession", axis=1)
y = data["Job profession"]


# ============================================
# STEP 7: Encode Career Names
# ============================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ============================================
# STEP 8: Train Random Forest Model
# ============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)


# ============================================
# STEP 9: Save Model
# ============================================

joblib.dump(model, "model/career_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")


print("\nModel Training Completed Successfully!")
print("Model saved in model/career_model.pkl")
print("Encoder saved in model/label_encoder.pkl")