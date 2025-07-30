import json
from pathlib import Path

LOG_FILE = "logs/run_log.json"
FEEDBACK_FILE = "chunk_feedback.json"

def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file}")
        return {}

def compare_labels():
    predicted_data = load_json(LOG_FILE)
    human_labels = load_json(FEEDBACK_FILE)

    correct = []
    mismatches = []
    missing_preds = []

    for chunk, human_label in human_labels.items():
        pred_label = predicted_data.get(chunk, {}).get("game_classification")
        if pred_label is None:
            missing_preds.append((chunk, human_label))
        elif pred_label == human_label:
            correct.append((chunk, human_label))
        else:
            mismatches.append((chunk, human_label, pred_label))

    print("✅ Correct Matches:", len(correct))
    print("❌ Mismatches:", len(mismatches))
    print("🤷 Missing Predictions:", len(missing_preds))
    print("-" * 50)

    if mismatches:
        print("\n❌ Mismatched Examples:")
        for chunk, human, pred in mismatches:
            print(f"  {Path(chunk).name}: Human = {human} | Predicted = {pred}")

    if missing_preds:
        print("\n🤷 Missing Predictions:")
        for chunk, human in missing_preds:
            print(f"  {Path(chunk).name}: Human = {human}")

    print("\n🧠 Accuracy:", f"{len(correct) / len(human_labels) * 100:.2f}%" if human_labels else "N/A")

if __name__ == "__main__":
    compare_labels()
