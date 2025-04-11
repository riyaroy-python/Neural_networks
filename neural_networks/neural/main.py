from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.predict import evaluate_model
from src.utils import setup_logging

logger = setup_logging()

try:
    df = load_data("Data/raw/Admission(in).csv")
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    logger.info(f"Model MSE: {mse:.4f}, R2 Score: {r2:.4f}")
except Exception as e:
    logger.error(f"Error during execution: {e}")