from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

def predict_single(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    return model.predict(input_scaled)[0]