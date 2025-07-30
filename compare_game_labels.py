import json
import os

# Paths
HUMAN_LABELS_FILE = "chunk_feedback.json"  # from GUI
PREDICTIONS_FILE = "logs/run_log.json"  # from run_all.py

# Load human labels
def load_human_labels():
    with open(HUMAN_LABELS_FILE, "r") as f:
        return json.load(f)

# Load model predictions
def load_predictions():
    with open(PREDICTIONS_FILE, "r") as f:
        data = json.load(f)
        return data.get("game_classifications", {})

# Main comparison
def compare_labels():
    human_labels = load_human_labels()
    predictions = load_predictions()

    # Normalize keys (remove folder paths)
    normalized_predictions = {os.path.basename(k): v for k, v in predictions.items()}
    normalized_human_labels = {os.path.basename(k): v for k, v in human_labels.items()}

    correct = 0
    mismatches = 0
    missing = []

    for filename, human_label in normalized_human_labels.items():
        predicted_label = normalized_predictions.get(filename)
        if predicted_label is None:
            missing.append((filename, human_label))
        elif predicted_label == human_label:
            correct += 1
        else:
            mismatches += 1
            print(f"❌ Mismatch: {filename}: Human = {human_label}, Predicted = {predicted_label}")

    print("✅ Correct Matches:", correct)
    print("❌ Mismatches:", mismatches)
    print("🤷 Missing Predictions:", len(missing))
    print("-" * 50)
    if missing:
        print("\n🤷 Missing Predictions:")
        for filename, label in missing:
            print(f"  {filename}: Human = {label}")

    accuracy = (correct / len(normalized_human_labels)) * 100 if normalized_human_labels else 0
    print(f"\n🧠 Accuracy: {accuracy:.2f}%")

# Run it
if __name__ == "__main__":
    compare_labels()
