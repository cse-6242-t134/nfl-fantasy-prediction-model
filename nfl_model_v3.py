import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class NFLModel:
    def __init__(self, data, target_variable, test_size=0.2, random_state=42):
        """
        Initializes the NFLModel class.

        Parameters:
        - data: pandas DataFrame containing the dataset.
        - target_variable: string, name of the target variable column.
        - test_size: float, proportion of the dataset to include in the test split.
        - random_state: int, random seed for reproducibility.
        """
        self.data = data
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state

        # Placeholders for data splits and models
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = {'Model': [], 'Data': [], 'MAE': [], 'MSE': [], 'R2': []}

    def preprocess_data(self):
        """
        Preprocesses the data by splitting into training and testing sets,
        converting categorical variables to dummy variables, and scaling the features.
        """
        # Separate features and target
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]

        # Split data into training and testing sets
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Align the training and testing data
        self.X_train, self.X_test = self.X_train_raw.align(self.X_test_raw, join='left', axis=1, fill_value=0)

        # Convert y_train and y_test to 1D arrays if necessary
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()

        # Standardize data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("Data preprocessing completed.")

    def train_all_models(self):

        rf_model = (RandomForestRegressor(random_state=self.random_state, max_depth=5)
                    .fit(self.X_train_scaled, self.y_train))
        self.evaluate_model('Random Forest', rf_model)


    def evaluate_model(self, name, model):
        # model.fit(self.X_train_scaled, self.y_train)
        for data in ['Train', 'Test']:
            y_actual = self.y_train if data == 'Train' else self.y_test
            x_actual = self.X_train_scaled if data == 'Train' else self.X_test_scaled

            y_pred = model.predict(x_actual)
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)

            self.results['Model'].append(name)
            self.results['Data'].append(data)
            self.results['MAE'].append(mae)
            self.results['MSE'].append(mse)
            self.results['R2'].append(r2)

    def get_results(self):
        """
        Returns the evaluation results as a pandas DataFrame.
        """
        return pd.DataFrame(self.results)