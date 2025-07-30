import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load training data
with open("training_data.json", "r") as f:
    data = json.load(f)

X = [entry["features"] for entry in data]
y = [entry["label"] for entry in data]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"🧠 Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Save model
joblib.dump(clf, "game_classifier.joblib")
print("✅ Saved model as game_classifier.joblib")
