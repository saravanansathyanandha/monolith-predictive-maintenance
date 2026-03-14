import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class PredictiveModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)

    def generate_synthetic_data(self):
        np.random.seed(42)
        rows = 1000
        time = np.arange(rows)
        sensor_1 = 500 + time * -0.1 + np.random.normal(0, 5, rows)
        rul = rows - time
        return pd.DataFrame({'sensor_1': sensor_1, 'RUL': rul})

    def train(self):
        df = self.generate_synthetic_data()
        X = df[['sensor_1']]
        y = df['RUL']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(f"Model Trained. RMSE: {np.sqrt(mean_squared_error(y_test, preds))}")

if __name__ == "__main__":
    pm = PredictiveModel()
    pm.train()
