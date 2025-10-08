import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xg
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from scipy.stats import uniform, loguniform
from skopt import gp_minimize
import joblib

def data_preprocessor(data):
    Feature_set = list(data.columns[1:])
    if 'RA Temp control( Valve Feedback)' in Feature_set:
        Feature_set.remove('RA Temp control( Valve Feedback)')
    if 'RA damper control' in Feature_set:
        Feature_set.remove('RA damper control')
    if 'SA Fan Speed control' in Feature_set:
        Feature_set.remove('SA Fan Speed control')
    if 'Fan Power meter (KW)' in Feature_set:
        Feature_set.remove('Fan Power meter (KW)')

    x = data[Feature_set]
    y = data[["RA damper control", "RA Temp control( Valve Feedback)", "SA Fan Speed control", "Fan Power meter (KW)"]]

    x_scaler = MinMaxScaler()
    x_scaled = x_scaler.fit_transform(x)

    # save_model(x_scaler, "minmaxScaler")

    X_train_scaled, temp_X_scaled, y_train, temp_y = train_test_split(x_scaled, y, test_size = 0.3, random_state = 123)

    val_X_scaled, X_test_scaled, val_y, y_test = train_test_split(temp_X_scaled, temp_y, test_size=0.5, random_state=42)

    if np.isnan(X_train_scaled).sum() > 0:
        X_train_scaled = pd.DataFrame(X_train_scaled).fillna(method='ffill').values

    if y_train.isnull().sum().sum() > 0:
        y_train.fillna(method='ffill', inplace=True)

    if np.isnan(X_test_scaled).sum() > 0:
        X_test_scaled = pd.DataFrame(X_test_scaled).fillna(method='ffill').values

    if np.isnan(val_X_scaled).sum() > 0:
        val_X_scaled = pd.DataFrame(val_X_scaled).fillna(method='ffill').values

    if y_test.isnull().sum().sum() > 0:
        y_test.fillna(method='ffill', inplace=True)

    if val_y.isnull().sum().sum() > 0:
        val_y.fillna(method='ffill', inplace=True)

    return {'X_scaler' : x_scaler, 'X_train' : X_train_scaled, 'X_test' : X_test_scaled, 'y_train' : y_train, 'y_test' : y_test, 'X_val' : val_X_scaled, 'y_val' : val_y}

"""# Finding Mutual Correlation between target and features"""
def mutual_correlation(data, site_types, target):
  for site_type in site_types:
      Feature_set = list(data.columns[1:])

      x = data[Feature_set]
      y = data[target]

      if x.isnull().sum().sum() > 0:
        x.fillna(method='ffill', inplace=True)

      if y.isnull().sum() > 0:
        y.fillna(method='ffill', inplace=True)

      mi_scores = mutual_info_regression(x, y)

      mi_scores_series = pd.Series(mi_scores, index=x.columns)

      mi_scores_series_sorted = mi_scores_series.sort_values(ascending=False)

      mi_scores_series_sorted

      print(f"\n{site_type}Mutual Information Scores (Regression):")
      print(mi_scores_series_sorted)

"""# Heatmap Visualization for coorelation"""
def heatMapVisualization(data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Data Columns')
    plt.show()

"""# Defining Utility functions for MultiTarget Prediction"""
def get_current_conditions():
        # return {
        #     "Bag filter dirty status": 0.0,
        #     "Fan Power meter (KW)": 10.544381141662598,
        #     "OA Flow": 0.0,
        #     "OA Humid": 56.90983200073242,
        #     "OA Temp": 35.93263626098633,
        #     "Plant enable": 1.0,
        #     "RA temperature setpoint": 25.0,
        #     "RA CO2": 1474.063720703125,
        #     "RA CO2 setpoint": 850.0,
        #     "RA Damper feedback": 0.0,
        #     "RA Temp": 23.376264572143555,
        #     "RA Temp control( Valve Feedback)": 0.0,
        #     "RA damper control": 0.0,
        #     "SA Fan Speed control": 0.0,
        #     "SA Fan Speed feedback": 0.0,
        #     "SA Pressure setpoint": 500.0,
        #     "SA pressure": 300.8293151855469,
        #     "SA temp": 16.580490112304688,
        #     "Sup fan cmd": 0.0,
        #     "Trip status": 1.0,
        #     "airflow Status": 1.0,
        #     "auto Status": 0.0,
        #     "pre Filter dirty staus": 0.0
        # }

        # return {
        #     "Bag filter dirty status": 0.0,
        #     "Fan Power meter (KW)": 9.500351905822754,
        #     "OA Flow": 0.0,
        #     "OA Humid": 67.43144989013672,
        #     "OA Temp": 46.691307067871094,
        #     "Plant enable": 1.0,
        #     "RA temperature setpoint": 20.0,
        #     "RA CO2": 1442.705078125,
        #     "RA CO2 setpoint": 850.0,
        #     "RA Damper feedback": 0.0,
        #     "RA Temp": 26.427051544189453,
        #     "RA Temp control( Valve Feedback)": 0.0,
        #     "RA damper control": 0.0,
        #     "SA Fan Speed control": 0.0,
        #     "SA Fan Speed feedback": 0.0,
        #     "SA Pressure setpoint": 500.0,
        #     "SA pressure": 294.4295959472656,
        #     "SA temp": 18.49893569946289,
        #     "Sup fan cmd": 0.0,
        #     "Trip status": 1.0,
        #     "airflow Status": 1.0,
        #     "auto Status": 0.0,
        #     "pre Filter dirty staus": 0.0
        # }

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

    #   return {
    #     "Fan Power meter (KW)": 9.500351905822754,
    #     "OA Flow": 0.0,
    #     "OA Humid": 67.43144989013672,
    #     "OA Temp": 46.691307067871094,
    #     "RA temperature setpoint": 20.0,
    #     "RA CO2": 1422.9434814453125,
    #     "RA CO2 setpoint": 850.0,
    #     "RA Damper feedback": 0.0,
    #     "RA Temp": 26.427051544189453,
    #     "RA Temp control( Valve Feedback)": 0.0,
    #     "RA damper control": 0.0,
    #     "SA Fan Speed control": 0.0,
    #     "SA Fan Speed feedback": 0.0,
    #     "SA Pressure setpoint": 500.0,
    #     "SA pressure": 290.3966369628906,
    #     "SA temp": 18.61865234375,
    #     "Trip status": 1.0,
    #     "date": 23,
    #     "month": 5,
    #     "year": 2025,
    #     "hour": 13,
    #     "minute": 52.65775
    # }

"""# Getting the important values to compare with the predicted values"""
def check_imp_values():
    focused_values_to_check_after_prediction = ["Fan Power meter (KW)","RA CO2","RA CO2 setpoint","RA damper control","RA Temp","RA temperature setpoint","RA Temp control( Valve Feedback)","SA pressure","SA Pressure setpoint","SA Fan Speed control"]

    print("\nActual Values : ")
    for feature in focused_values_to_check_after_prediction:
        print(f"{feature} : {get_current_conditions()[feature]}")

"""# Creating a input array with conditions and setpoints"""
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

"""# Optimizing the Setpoints"""
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

"""# Define the objective function to minimize (Fan Power) , This function will be used by the optimizer"""
def objective_function(setpoint_values, model, conditions, x_scaler):
    setpoints = {
        'RA temperature setpoint': setpoint_values[0],
        'RA CO2 setpoint': setpoint_values[1],
        'SA Pressure setpoint': setpoint_values[2]
    }
    input_data = create_input_array(conditions, setpoints)
    input_data_scaled = x_scaler.transform(input_data)

    predictions = model.predict(input_data_scaled)
    predicted_power = predictions[0][3]

    return predicted_power

"""# Finding the best tuning parameter of alpha and l1_ratio for elastic_net model using RandomSearchCV"""
def best_parameter_find(data, param_dict, model):

    X_scaler, X_train, X_test, y_train, y_test, X_val, y_val = data_preprocessor(data).values()

    random_search = RandomizedSearchCV(
      estimator=model,  # Pass the instance here
      param_distributions=param_dict,
      n_iter=50,
      cv=5,
      random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_

    return best_params
    # print(f"Ground Floor Best Hyperparameters: {best_params}")

"""# Training ElasticNet Model with best parameter as estimator model for multioutput Regressor Model"""
def multioutputRegressorModelTraining(data, prediction_arr, alpha_value = 0.5, l1_value=0.5):
    global model
    
    X_scaler, X_train, X_test, y_train, y_test, X_val, y_val = data_preprocessor(data).values()

    model = MultiOutputRegressor(
        ElasticNet(alpha=alpha_value, l1_ratio=l1_value), n_jobs=5)
    model.fit(X_train, y_train)

    # save_model(model,'ElasticNet.joblib')

    model_Evaluation(model, prediction_arr, X_scaler, X_test=X_test, X_val=X_val, y_test=y_test, y_val=y_val)

"""# Training XGBoostRegressor Model with best parameters as estimator model for multioutput Regressor Model"""
def xgBoostModelTraining(data, prediction_arr, best_params):
    global model
   
    X_scaler, X_train, X_test, y_train, y_test, X_val, y_val = data_preprocessor(data).values()

    multioutput_model_xg = xg.XGBRegressor(**best_params)

    model = multioutput_model_xg
    model.fit(X_train, y_train)

    model_Evaluation(model, prediction_arr, X_scaler, X_test=X_test, X_val=X_val, y_test=y_test, y_val=y_val)

"""# Experimenting with Bayesian Optimization gp_minimize function"""
def bayesian_optimization_model(data , prediction_arr, alpha_value = 0.5, l1_value=0.5):
    
    search_space_skopt = [
        (20.0, 27.0),
        (500.0, 800.0),
        (500.0, 1200.0)
    ]
    
    X_scaler, X_train, X_test, y_train, y_test, X_val, y_val = data_preprocessor(data).values()
    
    multioutput_model_bayes = MultiOutputRegressor(
        ElasticNet(alpha=alpha_value, l1_ratio=l1_value), n_jobs=5)
    multioutput_model_bayes.fit(X_train, y_train)
    
    res_gp = gp_minimize(lambda setpoint_values: objective_function(setpoint_values, multioutput_model_bayes, prediction_arr, X_scaler),
                         search_space_skopt,
                         n_calls=50,
                         random_state=42)
    
    best_setpoints_bayes_gp = res_gp.x
    predicted_minimal_power_bayes_gp = res_gp.fun
    
    print("\nBayesian Optimization (gp_minimize) Results:")
    print(f"Optimal setpoints: RA Temp Setpoint: {best_setpoints_bayes_gp[0]:.2f} °C, RA CO2 Setpoint: {best_setpoints_bayes_gp[1]:.2f} ppm, SA Pressure Setpoint: {best_setpoints_bayes_gp[2]:.2f} Pa")
    print(f"Predicted minimal Fan Power: {predicted_minimal_power_bayes_gp:.2f} KW")
    
    optimal_ra_temp_setpoint_rs = get_current_conditions()['RA temperature setpoint']
    optimal_ra_co2_setpoint_rs = get_current_conditions()['RA CO2 setpoint']
    optimal_sa_pressure_setpoint_rs = get_current_conditions()['SA Pressure setpoint']
    predicted_minimal_power_rs = get_current_conditions()['Fan Power meter (KW)']
    
    print("\nComparison with given datapoint:")
    print(f"Provided Setpoints: RA Temp Setpoint: {optimal_ra_temp_setpoint_rs:.2f} °C, RA CO2 Setpoint: {optimal_ra_co2_setpoint_rs:.2f} ppm, SA Pressure Setpoint: {optimal_sa_pressure_setpoint_rs:.2f} Pa")
    print(f"Given Fan Power: {predicted_minimal_power_rs:.2f} KW")
    
    # # Extract values from the prediction_arr dictionary
    # current_fan_power = prediction_arr['Fan Power meter (KW)']
    # current_ra_temp_setpoint = prediction_arr['RA temperature setpoint']
    # current_ra_co2_setpoint = prediction_arr['RA CO2 setpoint']
    # current_sa_pressure_setpoint = prediction_arr['SA Pressure setpoint']
    
    # # Extract values from the printed output (assuming the last execution output is available)
    # # This is a placeholder and needs to be replaced with actual parsing of the output
    # # For now, I will manually extract the values based on the last successful run output
    # optimal_ra_temp_setpoint = best_setpoints_bayes_gp[0]
    # optimal_ra_co2_setpoint = best_setpoints_bayes_gp[1]
    # optimal_sa_pressure_setpoint = best_setpoints_bayes_gp[2]
    # predicted_minimal_power = predicted_minimal_power_bayes_gp
    
    # # Create bar plot for Fan Power
    # plt.figure(figsize=(6, 4))
    # fan_power_labels = ['Current Fan Power', 'Predicted Minimal Fan Power']
    # fan_power_values = [current_fan_power, predicted_minimal_power]
    # plt.bar(fan_power_labels, fan_power_values, color=['blue', 'green'])
    # plt.ylabel('Fan Power (KW)')
    # plt.title('Current vs. Predicted Minimal Fan Power')
    # plt.show()
    
    # # Create bar plot for Setpoints
    # plt.figure(figsize=(10, 6))
    # setpoint_labels = ['RA Temp Setpoint', 'RA CO2 Setpoint', 'SA Pressure Setpoint']
    # current_setpoint_values = [current_ra_temp_setpoint, current_ra_co2_setpoint, current_sa_pressure_setpoint]
    # optimal_setpoint_values = [optimal_ra_temp_setpoint, optimal_ra_co2_setpoint, optimal_sa_pressure_setpoint]
    
    # x = np.arange(len(setpoint_labels))
    # width = 0.35
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # rects1 = ax.bar(x - width/2, current_setpoint_values, width, label='Current')
    # rects2 = ax.bar(x + width/2, optimal_setpoint_values, width, label='Optimal')
    
    # ax.set_ylabel('Setpoint Value')
    # ax.set_title('Current vs. Optimal Setpoints')
    # ax.set_xticks(x)
    # ax.set_xticklabels(setpoint_labels)
    # ax.legend()
    
    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(round(height, 2)),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    
    # autolabel(rects1)
    # autolabel(rects2)
    
    # fig.tight_layout()
    # plt.show()

"""# Training Voting Regressor Model"""
def voting_regressor_model(data):
    global model
   
    X_train, X_test, y_train, y_test, X_val, y_val = data_preprocessor(data).values()

    linear_reg = LinearRegression()
    svr_reg = SVR(kernel='sigmoid')  # You can choose different kernels if needed
    xgb_reg = xg.XGBRegressor(random_state=42)
    rf_reg = RandomForestRegressor(random_state=42)

    multioutput_model = MultiOutputRegressor(
        VotingRegressor(estimators=[('linear', linear_reg),  ('svr', svr_reg), ('xgb', xgb_reg), ('rf', rf_reg)],
                                n_jobs=5), n_jobs=5)
    multioutput_model.fit(X_train, y_train)
    model = multioutput_model

    multioutput_test_pred = model.predict(X_test)
    multioutput_val_pred = model.predict(X_val)

    multioutput_test_mse = mean_squared_error(y_test, multioutput_test_pred)
    multioutput_test_mae = mean_absolute_error(y_test, multioutput_test_pred)

    multioutput_validation_mse = mean_squared_error(y_val, multioutput_val_pred)
    multioutput_validation_mae = mean_absolute_error(y_val, multioutput_val_pred)

    print("Multioutput Voting Regressor Model- Mean Squared Error - Testing Data:", multioutput_test_mse)
    print("Multioutput Voting Regressor Model - Mean Absolute Error: - Testing Data", multioutput_test_mae)

    print("\nMultioutput Voting Regressor Model - Mean Squared Error - Validation Data:", multioutput_validation_mse)
    print("Multioutput Voting Regressor Model - Mean Absolute Error - Validation Data:", multioutput_validation_mae)

"""# Prediction Function"""
def model_Evaluation(model, current_conditions, X_scaler, X_test, X_val, y_test, y_val):
    model_test_pred = model.predict(X_test)
    model_val_pred = model.predict(X_val)

    model_test_mse = mean_squared_error(y_test, model_test_pred)
    model_test_mae = mean_absolute_error(y_test, model_test_pred)

    model_validation_mse = mean_squared_error(y_val, model_val_pred)
    model_validation_mae = mean_absolute_error(y_val, model_val_pred)

    print("Multioutput ElasticNet Model - Mean Squared Error - Testing Data:", model_test_mse)
    print("Multioutput ElasticNet Model - Mean Absolute Error: - Testing Data", model_test_mae)

    print("\nMultioutput ElasticNet Model - Mean Squared Error - Validation Data:", model_validation_mse)
    print("Multioutput ElasticNet Model - Mean Absolute Error - Validation Data:", model_validation_mae)

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

"""# Saving a Model"""
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    data = pd.read_csv('C:/Users/KER LPTP- 31/Desktop/Dataset/AHU_Data_processing/Data Pipeline/Cleaned_data.csv')

    data['data_received_on'] = pd.to_datetime(data['data_received_on'])
    # print(data["data_received_on"].dtypes)

    data['date'] = data['data_received_on'].dt.day
    data['month'] = data['data_received_on'].dt.month
    data['year'] = data['data_received_on'].dt.year
    data['hour'] = data['data_received_on'].dt.hour
    data['minute'] = data['data_received_on'].dt.minute

    data.set_index('data_received_on',inplace=True)

    # mutual_correlation(data, ["Ground Floor"], "Fan Power meter (KW)")

    """# Feature Selection : Dropping features with very less correaltion with 'Fan Power meter (KW)' feature"""
    data.drop(columns=["Unnamed: 0","Bag filter dirty status", "auto Status","pre Filter dirty staus","Plant enable","Sup fan cmd","airflow Status"], inplace=True)

    new_data_for_visulaizing_correlation = data.copy()
    new_data_for_visulaizing_correlation.drop(columns=["date","month","year","hour","minute","Trip status"], inplace=True)

    # heatMapVisualization(new_data_for_visulaizing_correlation)
    # check_imp_values()

    elasticNet_param_dict = {
      'alpha': loguniform(0.1, 1),
      'l1_ratio': loguniform(0.1, 1)
    }

    best_ElasticNet_params = best_parameter_find(data, param_dict=elasticNet_param_dict, model=ElasticNet(alpha=0.5, l1_ratio=0.5))
    # print(best_ElasticNet_params)

    multioutputRegressorModelTraining( data, get_current_conditions(), alpha_value=best_ElasticNet_params['alpha'] , l1_value=best_ElasticNet_params['l1_ratio'])
    # bayesian_optimization_model(data, get_current_conditions(), alpha_value=best_ElasticNet_params['alpha'] , l1_value=best_ElasticNet_params['l1_ratio'])

    # xgb_params = {
    #   'estimator__objective': ['reg:squarederror', 'reg:linear'],
    #   'estimator__n_estimators': [10, 50, 100, 200],
    #   'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    #   'estimator__max_depth': [3, 4, 5, 6, 7, 8],
    #   'estimator__min_child_weight': [1, 3, 5, 7],
    #   'estimator__gamma': [0, 0.1, 0.2, 0.3, 0.4],
    #   'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    #   'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    #   'estimator__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    #   'estimator__reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    # }

    # best_Xgb_params = best_parameter_find(data,xgb_params,xg.XGBRegressor(seed=123))
    # print(best_Xgb_params)

    # xgBoostModelTraining(data, get_current_conditions(), best_Xgb_params)

    # voting_regressor_model(data)