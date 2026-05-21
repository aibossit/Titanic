from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


INITIAL_REPLACEMENTS = {
    "Mlle": "Miss",
    "Mme": "Miss",
    "Ms": "Miss",
    "Dr": "Other",
    "Major": "Other",
    "Lady": "Other",
    "Countess": "Other",
    "Jonkheer": "Other",
    "Col": "Other",
    "Rev": "Other",
    "Capt": "Other",
    "Sir": "Mr",
    "Don": "Mr",
}

INITIAL_MAP = {
    "Mr": 0,
    "Mrs": 1,
    "Miss": 2,
    "Master": 3,
    "Other": 4,
}


def make_feature_info(train_df):
    age = train_df["Age"].fillna(train_df["Age"].median())
    fare = train_df["Fare"].fillna(train_df["Fare"].median())

    _, age_bins = pd.cut(age, 10, retbins=True, labels=False)
    _, fare_bins = pd.qcut(fare, 13, retbins=True, labels=False, duplicates="drop")

    age_bins[0] = float("-inf")
    age_bins[-1] = float("inf")
    fare_bins[0] = float("-inf")
    fare_bins[-1] = float("inf")

    return {
        "age_median": train_df["Age"].median(),
        "fare_median": train_df["Fare"].median(),
        "embarked_mode": train_df["Embarked"].mode()[0],
        "age_bins": age_bins,
        "fare_bins": fare_bins,
        "columns": None,
    }


def prepare_features(df, feature_info, is_train=False):
    data = df.copy()

    data["Age"] = data["Age"].fillna(feature_info["age_median"])
    data["Fare"] = data["Fare"].fillna(feature_info["fare_median"])
    data["Embarked"] = data["Embarked"].fillna(feature_info["embarked_mode"])

    data["Age_bin"] = pd.cut(data["Age"], feature_info["age_bins"], labels=False)
    data["Fare_bin"] = pd.cut(data["Fare"], feature_info["fare_bins"], labels=False)
    data["Ticket_group"] = data.groupby("Ticket")["Ticket"].transform("count")

    data["Initial"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    data["Initial"] = data["Initial"].replace(INITIAL_REPLACEMENTS)
    data["Initial"] = data["Initial"].map(INITIAL_MAP).fillna(INITIAL_MAP["Other"])

    data["Family_size"] = data["SibSp"] + data["Parch"] + 1
    data["Is_alone"] = (data["Family_size"] == 1).astype(int)
    data["Deck"] = data["Cabin"].str[0].fillna("Unknown")

    data = pd.get_dummies(data, columns=["Embarked", "Sex", "Deck"], drop_first=True)
    data = data.drop(
        columns=["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"],
        errors="ignore",
    )

    if is_train:
        y = data["Survived"]
        x = data.drop(columns="Survived")
        feature_info["columns"] = list(x.columns)
        return pd.concat([y, x], axis=1)

    return data.reindex(columns=feature_info["columns"], fill_value=0)


def build_processed_datasets():
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")

    feature_info = make_feature_info(train_df)
    train_processed = prepare_features(train_df, feature_info, is_train=True)
    test_processed = prepare_features(test_df, feature_info)

    PROCESSED_DIR.mkdir(exist_ok=True)
    train_processed.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test.csv", index=False)

    return train_processed, test_processed, feature_info


if __name__ == "__main__":
    build_processed_datasets()
    print("Processed datasets saved.")
