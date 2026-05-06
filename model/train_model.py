import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("dataset/career_data.xlsx")

print("Original Shape:", df.shape)


df = df.drop(columns=["Sr.No.", "Course", "Student", "s/p"], errors='ignore')

df["Job profession"] = df["Job profession"].astype(str).str.strip()

df = df.dropna()

print("After Cleaning Shape:", df.shape)


mapping = {
    "BEST": 2,
    "AVG": 1,
    "POOR": 0
}

performance_cols = ["P1","P2","P3","P4","P5","P6","P7","P8"]

for col in performance_cols:
    df[col] = df[col].map(mapping)

print("Converted Performance Columns ✅")


X = df.drop("Job profession", axis=1)
y = df["Job profession"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        batch_size=16,
        max_iter=900,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    ))
])

model.fit(X_train, y_train)

print("Model Trained ✅")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/encoder.pkl")

print("Model Saved ✅")
