import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

        # rf_model = (RandomForestRegressor(random_state=self.random_state, max_depth=5)
        #             .fit(self.X_train_scaled, self.y_train))

        rf_model = RandomForestRegressor(random_state=self.random_state)
        rf_hyperparams = self._best_hyperparams(rf_model, {'max_depth': [3, 5]}) # 7 is better than 5 and 9, if all features
        print(f'Random Forest hyperparams: {rf_hyperparams}')
        rf_model = (RandomForestRegressor(random_state=self.random_state, **rf_hyperparams)
                    .fit(self.X_train_scaled, self.y_train))
        self._evaluate_model('Random Forest', rf_model)
        self.print_feature_importances(rf_model)

        # best_params = ['fantasy_points_mean_last5', 'fantasy_points_mean_career', 'n_games_career', 'fantasy_points_mean_prior_season', 'fantasy_points_mean_season', 'total_fg_missed_mean_career',
        #                'fantasy_points_mean_season_def', 'fantasy_points_mean_last5_def', 'fantasy_points_mean_prior_season_def',
        #                'temp', 'wind', 'roof', 'report_primary_injury', 'report_status']
        # rf_model_reduced = RandomForestRegressor(random_state=self.random_state)
        # rf_hyperparams = self._best_hyperparams(rf_model_reduced, {'max_depth': [5, 7, 9]}) # 7 is better than 5 and 9
        # print(f'Random Forest hyperparams: {rf_hyperparams}')
        # rf_model_reduced = (RandomForestRegressor(random_state=self.random_state, **rf_hyperparams)
        #             .fit(self.X_train_scaled[:, best_params], self.y_train))
        # self._evaluate_model('Random Forest Reduced', rf_model_reduced, params=best_params)
        # self.print_feature_importances(rf_model_reduced)

        # elnet_model = ElasticNet(random_state=0)
        # elnet_hyperparams = self._best_hyperparams(elnet_model, {'alpha': [0.01, 0.05, 0.1], 'l1_ratio': [0, 0.5, 1]})
        # print(f'ElasticNet hyperparams: {elnet_hyperparams}')
        # elnet_model = (ElasticNet(random_state=0, **elnet_hyperparams)
        #             .fit(self.X_train_scaled, self.y_train))
        # self._evaluate_model('ElasticNet', elnet_model)

        # xgb_hyperparams = {'n_estimators': 2, 'max_depth': 5, 'learning_rate': 1}
        # print(f'XGBoost hyperparams: {xgb_hyperparams}')
        # xgb_model = (XGBRegressor(objective='reg:squarederror', **xgb_hyperparams)
        #              .fit(self.X_train_scaled, self.y_train))
        # self._evaluate_model('XGBoost', xgb_model)

    def _best_hyperparams(self, model, hyperparams):
        gscv = GridSearchCV(estimator = model, param_grid = hyperparams)
        gscv.fit(self.X_train_scaled, self.y_train)
        return gscv.best_params_


    def _evaluate_model(self, name, model):
        # model = Model(**hyperparams)
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

    def print_feature_importances(self, model):
        feature_importances = dict(zip(self.X_train.columns, model.feature_importances_))
        feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        print(feature_importances)


    def get_results(self):
        """
        Returns the evaluation results as a pandas DataFrame.
        """
        return pd.DataFrame(self.results)
    

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