from sklearn.neural_network import MLPRegressor

def train_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
