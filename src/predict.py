import joblib
import pandas as pd

from src.config import ROOT_DIR, load_config
from src.features import RAW_DIR, prepare_features


def make_submission():
    config = load_config()

    model_path = ROOT_DIR / config["paths"]["models_dir"] / config["paths"]["model_file"]
    submission_path = ROOT_DIR / config["paths"]["submission_file"]

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_info = artifact["feature_info"]

    test_df = pd.read_csv(RAW_DIR / "test.csv")
    x_test = prepare_features(test_df, feature_info)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": model.predict(x_test).astype(int),
        }
    )

    submission.to_csv(submission_path, index=False)
    return submission


if __name__ == "__main__":
    submission = make_submission()
    print(f"Saved {len(submission)} predictions to submission.csv")
