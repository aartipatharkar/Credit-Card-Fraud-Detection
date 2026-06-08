from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df):

    df.columns = df.columns.str.lower()

    # Faster training using sample
    df = df.sample(
        n=10000,
        random_state=42
    )

    target = "class"

    features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    features.remove(target)

    X = df[features]
    y = df[target]

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save both model and features
    joblib.dump(
        {
            "model": model,
            "features": features
        },
        "fraud_model.pkl"
    )

    return model, features, target
