from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib
import pandas as pd

class RacingNet:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
        self.is_trained = False
        try:
            self.model = joblib.load('racing_net.pkl')
            self.is_trained = True
        except FileNotFoundError:
            pass
    
    def predict(self, state):
        if not self.is_trained:
            # Return default actions until trained
            return np.array([0.0, 0.5, 0.0])  # Neutral steer, moderate accel, no brake
        return self.model.predict(state.reshape(1, -1))[0]
    
    def train(self):
        try:
            # Load data from CSV
            df = pd.read_csv('telemetry_log.csv')
            if len(df) < 100:  # Ensure enough data
                print("Not enough data to train model")
                return
            
            # Prepare state (inputs)
            state_cols = [f'track_{i}' for i in range(19)] + [f'opponents_{i}' for i in range(36)] + ['angle', 'speedX', 'trackPos', 'rpm']
            X = df[state_cols].values
            
            # Prepare actions (outputs)
            action_cols = ['steer', 'accel', 'brake']
            y = df[action_cols].values
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            print("Model trained successfully")
        except Exception as e:
            print(f"Error training model: {e}")
    
    def save(self):
        if self.is_trained:
            joblib.dump(self.model, 'racing_net.pkl')
            print("Model saved to racing_net.pkl")