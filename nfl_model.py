import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

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
        self.rf_model = None
        self.lstm_model = None
        self.results = {}

    def preprocess_data(self):
        """
        Preprocesses the data by splitting into training and testing sets,
        and scaling the features.
        """
        # Separate features and target
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Standardize data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Reshape data for LSTM input (samples, time steps, features)
        self.X_train_lstm = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1]))
        self.X_test_lstm = self.X_test_scaled.reshape((self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1]))

        print("Data preprocessing completed.")

    def feature_selection(self):
        """
        Performs feature selection using Lasso and Elastic Net methods.
        """
        # Initialize models
        lasso_selector = Lasso(alpha=0.1, random_state=self.random_state)
        elastic_net_selector = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)

        # Lasso Feature Selection
        lasso_selector.fit(self.X_train_scaled, self.y_train)
        self.lasso_features = [
            feature for feature, coef in zip(self.X_train.columns, lasso_selector.coef_) if coef != 0
        ]
        print(f"Lasso selected features: {self.lasso_features}")

        # Elastic Net Feature Selection
        elastic_net_selector.fit(self.X_train_scaled, self.y_train)
        self.elastic_net_features = [
            feature for feature, coef in zip(self.X_train.columns, elastic_net_selector.coef_) if coef != 0
        ]
        print(f"Elastic Net selected features: {self.elastic_net_features}")

    def evaluate_models(self):
        """
        Evaluates Linear Regression and Random Forest models using different
        feature selection methods and stores the results.
        """
        # Define models for evaluation
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(random_state=self.random_state)

        # Initialize results dictionary
        self.results = {'Method': [], 'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

        # Create feature sets based on selection methods
        feature_sets = {
            'Lasso': self.lasso_features,
            'ElasticNet': self.elastic_net_features
        }

        # Evaluate models with each feature selection method
        for method, features in feature_sets.items():
            X_train_fs = self.X_train[features]
            X_test_fs = self.X_test[features]

            # Linear Regression
            mae_linear, mse_linear, r2_linear = self._evaluate_model(
                linear_model, X_train_fs, X_test_fs, self.y_train, self.y_test
            )
            self.results['Method'].append(method)
            self.results['Model'].append('Linear Regression')
            self.results['MAE'].append(mae_linear)
            self.results['MSE'].append(mse_linear)
            self.results['R2'].append(r2_linear)

            # Random Forest
            mae_rf, mse_rf, r2_rf = self._evaluate_model(
                rf_model, X_train_fs, X_test_fs, self.y_train, self.y_test
            )
            self.results['Method'].append(method)
            self.results['Model'].append('Random Forest')
            self.results['MAE'].append(mae_rf)
            self.results['MSE'].append(mse_rf)
            self.results['R2'].append(r2_rf)

        print("Model evaluation completed.")

    def train_evaluate_rf_all_features(self):
        """
        Trains and evaluates a Random Forest model using all features.
        Stores the result in the 'results' dictionary.

        Returns:
        - model: the machine learning model to evaluate.
        """
        # Define the Random Forest model
        rf_model = RandomForestRegressor(random_state=self.random_state)

        # Train and evaluate the model using all features
        mae_rf, mse_rf, r2_rf = self._evaluate_model(
            rf_model, self.X_train, self.X_test, self.y_train, self.y_test
        )

        # Initialize results dictionary if not already done
        if self.results is None:
            self.results = {'Method': [], 'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

        # Store the results
        self.results['Method'].append('All Features')
        self.results['Model'].append('Random Forest')
        self.results['MAE'].append(mae_rf)
        self.results['MSE'].append(mse_rf)
        self.results['R2'].append(r2_rf)

        print(f"Random Forest with all features evaluated. MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

        # # Optionally, plot actual vs. predicted values
        # y_pred_rf = rf_model.predict(self.X_test)
        # self._plot_predictions(self.y_test, y_pred_rf, 'Random Forest with All Features')

        return rf_model
    
    def train_random_forest(self, n_estimators=100):
        """
        Trains a Random Forest model using the training data.

        Parameters:
        - n_estimators: int, number of trees in the forest.
        """
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state)
        self.rf_model.fit(self.X_train, self.y_train)
        print("Random Forest training completed.")

    def evaluate_random_forest(self):
        """
        Evaluates the trained Random Forest model on the test data.
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model has not been trained yet.")
        rf_predictions = self.rf_model.predict(self.X_test)
        rf_mae = mean_absolute_error(self.y_test, rf_predictions)
        rf_mse = mean_squared_error(self.y_test, rf_predictions)
        rf_r2 = r2_score(self.y_test, rf_predictions)
        self.results['Random Forest'] = {'MAE': rf_mae, 'MSE': rf_mse, 'R2': rf_r2}
        print(f"Random Forest Test MAE: {rf_mae:.2f} fantasy points")

    def tune_random_forest(self):
        """
        Performs hyperparameter tuning for the Random Forest model using GridSearchCV.
        """
        # Define the parameter grid without 'auto' for max_features
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2', 0.2, 0.5],  # Removed 'auto'
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize the Random Forest model
        rf = RandomForestRegressor(random_state=self.random_state)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_absolute_error',
            verbose=2
        )

        # Fit GridSearchCV
        grid_search.fit(self.X_train, self.y_train)

        # Retrieve the best parameters and set the model
        self.rf_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best MAE score: {-grid_search.best_score_:.4f}")

        # Evaluate the tuned model
        self.evaluate_random_forest()
        
    def build_lstm_model(self, units=64, dropout_rate=0.3):
        """
        Builds and compiles the LSTM model.

        Parameters:
        - units: int, number of units in the LSTM layers.
        - dropout_rate: float, dropout rate for regularization.
        """
        self.lstm_model = Sequential([
            LSTM(units, input_shape=(self.X_train_lstm.shape[1], self.X_train_lstm.shape[2]), activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        print("LSTM model built and compiled.")

    def train_lstm(self, epochs=100, batch_size=32, patience=10):
        """
        Trains the LSTM model using the training data.

        Parameters:
        - epochs: int, number of epochs to train the model.
        - batch_size: int, number of samples per gradient update.
        - patience: int, number of epochs with no improvement after which training will be stopped.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been built yet.")
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.lstm_model.fit(
            self.X_train_lstm, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        print("LSTM training completed.")

    def evaluate_lstm(self):
        """
        Evaluates the trained LSTM model on the test data.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been trained yet.")
        lstm_test_loss, lstm_test_mae = self.lstm_model.evaluate(self.X_test_lstm, self.y_test, verbose=0)
        self.results['LSTM'] = {'MAE': lstm_test_mae, 'Loss': lstm_test_loss}
        print(f"LSTM Test MAE: {lstm_test_mae:.2f} fantasy points")

    def evaluate_ensemble(self):
        """
        Evaluates an ensemble model that averages predictions from both
        the trained LSTM and Random Forest models.
        """
        if self.rf_model is None or self.lstm_model is None:
            raise ValueError("Both Random Forest and LSTM models must be trained before evaluating the ensemble.")
        rf_predictions = self.rf_model.predict(self.X_test)
        lstm_predictions = self.lstm_model.predict(self.X_test_lstm).flatten()
        ensemble_predictions = (lstm_predictions + rf_predictions) / 2
        ensemble_mae = mean_absolute_error(self.y_test, ensemble_predictions)
        ensemble_mse = mean_squared_error(self.y_test, ensemble_predictions)
        ensemble_r2 = r2_score(self.y_test, ensemble_predictions)
        self.results['Ensemble'] = {'MAE': ensemble_mae, 'MSE': ensemble_mse, 'R2': ensemble_r2}
        print(f"Ensemble Test MAE: {ensemble_mae:.2f} fantasy points")

    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Trains the model and evaluates it on the test set.

        Parameters:
        - model: the machine learning model to evaluate.
        - X_train: training features.
        - X_test: testing features.
        - y_train: training target.
        - y_test: testing target.

        Returns:
        - MAE: Mean Absolute Error of the model on the test set.
        - MSE: Mean Squared Error of the model on the test set.
        - R2: R-squared value of the model on the test set.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, mse, r2

    def _plot_predictions(self, y_true, y_pred, title):
        """
        Plots actual vs. predicted values.

        Parameters:
        - y_true: actual target values.
        - y_pred: predicted target values.
        - title: title for the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        sns.lineplot(x=y_true, y=y_true, color='red')  # Line showing perfect prediction
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.show()

    def get_results(self):
        """
        Returns the evaluation results as a pandas DataFrame.
        """
        return pd.DataFrame(self.results)
    