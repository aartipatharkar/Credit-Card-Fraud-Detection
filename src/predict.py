import joblib

def predict_fraud(user_inputs):

    data = joblib.load(
        "fraud_model.pkl"
    )

    model = data["model"]
    features = data["features"]

    # Fill all features with 0
    full_input = [0] * len(features)

    # Replace first few values with user input
    for i in range(len(user_inputs)):

        full_input[i] = user_inputs[i]

    result = model.predict(
        [full_input]
    )

    return result[0]
