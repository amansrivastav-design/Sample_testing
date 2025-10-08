import joblib    
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_current_conditions():
        return {
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

def create_input_array(conditions, setpoints):
        input_list = [
            conditions['OA Flow'], conditions['OA Humid'], conditions['OA Temp'],
            setpoints['RA temperature setpoint'],
            setpoints['RA CO2 setpoint'], conditions['RA CO2'],
            conditions['RA Damper feedback'], conditions['RA Temp'],
            conditions['SA Fan Speed feedback'], setpoints['SA Pressure setpoint'],
            conditions['SA pressure'], conditions['SA temp'],
            conditions['Trip status'],
            conditions['date'], conditions['month'], conditions['year'], conditions['hour'], conditions['minute']
        ]
        return np.array([input_list])

def optimize_setpoints(model, conditions, x_scaler, n_iterations=1000):
        search_space = {
            'RA temperature setpoint': (20.0, 27.0),
            'RA CO2 setpoint': (500.0, 800.0),
            'SA Pressure setpoint': (500.0, 1200.0)
        }

        # Initialize with current conditions
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


            # Predict all outputs, including fan power
            # The model is expected to output: [RA_damper_control, RA_temp_control, SA_fan_speed_control, Fan_Power]
            # Adjust index if your multioutput_model's outputs are in a different order
            predictions = model.predict(input_data_scaled)
            predicted_power = predictions[0][3]

            # print(best_power)

            if predicted_power < best_power:
                best_power = predicted_power
                best_setpoints = random_setpoints

        return best_setpoints, best_power

def model_prediction(model, current_conditions, X_scaler):
    optimal_setpoints, minimal_power = optimize_setpoints(model, current_conditions, X_scaler)

    print("\nOptimal setpoints to minimize Fan Power:")
    print(f"RA Temp Setpoint: {optimal_setpoints['RA temperature setpoint']:.2f} °C")
    print(f"RA CO2 Setpoint: {optimal_setpoints['RA CO2 setpoint']:.2f} ppm")
    print(f"SA Pressure Setpoint: {optimal_setpoints['SA Pressure setpoint']:.2f} Pa")
    print(f"\nPredicted minimal Fan Power: {minimal_power:.2f} KW")

    optimal_input = create_input_array(current_conditions, optimal_setpoints)
    optimal_input_scaled = X_scaler.transform(optimal_input)
    optimal_commands = model.predict(optimal_input_scaled)

    optimal_commands[0][0] = np.clip(optimal_commands[0][0], 0, 10)
    optimal_commands[0][1] = np.clip(optimal_commands[0][1], 0, 100)
    optimal_commands[0][2] = np.clip(optimal_commands[0][2], 0, 100)

    print("\nCorresponding optimal control commands:")
    print(f"RA Damper Control: {optimal_commands[0][0]:.2f}")
    print(f"RA Temp control( Valve Feedback): {optimal_commands[0][1]:.2f}")
    print(f"SA Fan Speed Control: {optimal_commands[0][2]:.2f}")

if __name__ == "__main__":
    loaded_model = joblib.load("C:/Users/KER LPTP- 31/Desktop/BMS Ground Floor/Data Pipeline/ElasticNet.joblib")
    print("Model loaded successfully.")

    loaded_scaler = joblib.load("C:/Users/KER LPTP- 31/Desktop/BMS Ground Floor/Data Pipeline/minmaxScaler")
    print("Scaler Loaded Successfully.")
    
    model_prediction(model=loaded_model, current_conditions=get_current_conditions(), X_scaler=loaded_scaler)