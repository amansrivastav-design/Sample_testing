import joblib
import numpy as np
import warnings
from fastapi import FastAPI, Body
from typing import Dict, Any
import uvicorn

warnings.filterwarnings('ignore')

app = FastAPI()

CURRENT_CONDITIONS_STATE = {
    "Fan Power meter (KW)": 10.509812355041504,
    "OA Flow": 819.8904418945312,
    "OA Humid": 55.93263626098633,
    "OA Temp": 35.93263626098633,
    "RA temperature setpoint": 24.5,
    "RA CO2": 500,
    "RA CO2 setpoint": 750.0,
    "RA Damper feedback": 15.0,
    "RA Temp": 23.376264572143555,
    "RA Temp control( Valve Feedback)": 1.6030502319335938,
    "RA damper control": 10.0,
    "SA Fan Speed control": 100.0,
    "SA Fan Speed feedback": 100.0,
    "SA Pressure setpoint": 750.0,
    "SA pressure": 297.919921875,
    "SA temp": 16.363384246826172,
    "Trip status": 1.0,
    'date': 22,
    'month' : 5,
    'year' : 2025,
    'hour' : 6,
    'minute': 20
}

try:
    loaded_model = joblib.load("C:/Users/KER LPTP- 31/Desktop/BMS Ground Floor/Data Pipeline/ElasticNet.joblib")
    loaded_scaler = joblib.load("C:/Users/KER LPTP- 31/Desktop/BMS Ground Floor/Data Pipeline/minmaxScaler")
except FileNotFoundError:
    class DummyModel:
        def predict(self, X):
            return np.array([[10.0, 50.0, 80.0, 10.0] for _ in range(X.shape[0])])
    class DummyScaler:
        def transform(self, X):
            return X
    loaded_model = DummyModel()
    loaded_scaler = DummyScaler()

def create_input_array(conditions: Dict[str, Any], setpoints: Dict[str, float]):
    input_list = [
        conditions['OA Flow'], conditions['OA Humid'], conditions['OA Temp'],
        setpoints['RA temperature setpoint'], conditions['RA CO2 setpoint'], 
        conditions['RA CO2'], conditions['RA Damper feedback'], conditions['RA Temp'],
        conditions['SA Fan Speed feedback'], setpoints['SA Pressure setpoint'],
        conditions['SA pressure'], conditions['SA temp'], conditions['Trip status'],
        conditions['date'], conditions['month'], conditions['year'], conditions['hour'], conditions['minute']
    ]
    return np.array([input_list])

def optimize_setpoints(model, conditions: Dict[str, Any], x_scaler, n_iterations=1000):
    search_space = {
        'RA temperature setpoint': (20.0, 27.0),
        'RA CO2 setpoint': (500.0, 800.0),
        'SA Pressure setpoint': (500.0, 1200.0)
    }

    current_setpoints = {
        'RA temperature setpoint': conditions['RA temperature setpoint'],
        'RA CO2 setpoint': conditions['RA CO2 setpoint'],
        'SA Pressure setpoint': conditions['SA Pressure setpoint']
    }
    
    current_input_data = create_input_array(conditions, current_setpoints)
    current_input_data_scaled = x_scaler.transform(current_input_data)
    current_predictions = model.predict(current_input_data_scaled)
    current_power = current_predictions[0][3]

    best_power = current_power
    best_setpoints = current_setpoints.copy()

    for _ in range(n_iterations):
        random_setpoints = {
            'RA temperature setpoint': np.random.uniform(*search_space['RA temperature setpoint']),
            'RA CO2 setpoint': np.random.uniform(*search_space['RA CO2 setpoint']),
            'SA Pressure setpoint': np.random.uniform(*search_space['SA Pressure setpoint'])
        }

        input_data = create_input_array(conditions, random_setpoints)
        input_data_scaled = x_scaler.transform(input_data)
        predictions = model.predict(input_data_scaled)
        predicted_power = predictions[0][3]

        if predicted_power < best_power:
            best_power = predicted_power
            best_setpoints = random_setpoints

    return best_setpoints, best_power

def prediction_with_optimization(current_conditions: Dict[str, Any], model, X_scaler):
    optimal_setpoints, minimal_power = optimize_setpoints(model, current_conditions, X_scaler)

    optimal_input = create_input_array(current_conditions, optimal_setpoints)
    optimal_input_scaled = X_scaler.transform(optimal_input)
    optimal_commands = model.predict(optimal_input_scaled)

    optimal_commands[0][0] = np.clip(optimal_commands[0][0], 0, 10)
    optimal_commands[0][1] = np.clip(optimal_commands[0][1], 0, 100)
    optimal_commands[0][2] = np.clip(optimal_commands[0][2], 0, 100)

    return {
        "Optimal setpoints to minimize Fan Power": {
            "RA Temp Setpoint": f"{optimal_setpoints['RA temperature setpoint']:.2f} °C",
            "RA CO2 Setpoint": f"{optimal_setpoints['RA CO2 setpoint']:.2f} ppm",
            "SA Pressure Setpoint": f"{optimal_setpoints['SA Pressure setpoint']:.2f} Pa",
            "Predicted Minimal Fan Power": f"{minimal_power:.2f} KW"
        },
        "Corresponding optimal control commands": {
            "RA Damper Control": f"{optimal_commands[0][0]:.2f}",
            "RA Temp control (Valve Feedback)": f"{optimal_commands[0][1]:.2f}",
            "SA Fan Speed Control": f"{optimal_commands[0][2]:.2f}"
        }
    }

@app.post('/predict_new_data')
def predict_with_new_data(
    current_conditions: Dict[str, Any] = Body(default=CURRENT_CONDITIONS_STATE),
    model=loaded_model,
    X_scaler=loaded_scaler
):
    return prediction_with_optimization(current_conditions, model, X_scaler)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)