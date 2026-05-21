import sys

from src.features import build_processed_datasets
from src.predict import make_submission
from src.train import train


def run_all():
    print("1. Building features...")
    train_df, _, feature_info = build_processed_datasets()

    print("2. Training model...")
    result = train(train_df, feature_info)

    print("3. Creating submission...")
    submission = make_submission()

    best_model = result["best_model"]
    accuracy = result["models"][best_model]["summary"]["accuracy"]["mean"]

    print("\nDone.")
    print(f"Best model: {best_model}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Submission rows: {len(submission)}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"

    if command == "features":
        build_processed_datasets()
        print("Processed datasets saved.")
    elif command == "train":
        result = train()
        print(f"Model trained. Best model: {result['best_model']}")
    elif command == "predict":
        submission = make_submission()
        print(f"Saved {len(submission)} predictions to submission.csv")
    elif command == "all":
        run_all()
    else:
        print("Use: python run.py [all|features|train|predict]")
