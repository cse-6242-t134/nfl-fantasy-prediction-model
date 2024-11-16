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
        self.results = None

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
        rf_model = RandomForestRegressor(random_state=self.random_state, max_depth=5)

        # Initialize results dictionary
        self.results = {'Method': [], 'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

        # Create feature sets based on selection methods
        feature_sets = {
            'Lasso': self.lasso_features,
            'ElasticNet': self.elastic_net_features,
            'All Features': self.X_train.columns
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

    def evaluate_models_train(self):
        """
        Evaluates Linear Regression and Random Forest models using different
        feature selection methods and stores the results.
        """
        # Define models for evaluation
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(random_state=self.random_state, max_depth=5)

        # Initialize results dictionary
        self.results = {'Method': [], 'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

        # Create feature sets based on selection methods
        feature_sets = {
            'Lasso': self.lasso_features,
            'ElasticNet': self.elastic_net_features,
            'All Features': self.X_train.columns
        }

        # Evaluate models with each feature selection method
        for method, features in feature_sets.items():
            X_train_fs = self.X_train[features]
            # X_test_fs = self.X_test[features]

            # Linear Regression
            mae_linear, mse_linear, r2_linear = self._evaluate_model(
                linear_model, X_train_fs, X_train_fs, self.y_train, self.y_train
            )
            self.results['Method'].append(method)
            self.results['Model'].append('Linear Regression')
            self.results['MAE'].append(mae_linear)
            self.results['MSE'].append(mse_linear)
            self.results['R2'].append(r2_linear)

            # Random Forest
            mae_rf, mse_rf, r2_rf = self._evaluate_model(
                rf_model, X_train_fs, X_train_fs, self.y_train, self.y_train
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
        rf_model = RandomForestRegressor(random_state=self.random_state, max_depth=5)

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