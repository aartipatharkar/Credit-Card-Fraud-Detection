from sklearn.metrics import accuracy_score

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    return accuracy
