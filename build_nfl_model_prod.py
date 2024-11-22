import warnings 
import os
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import datetime as dt
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib  # For saving/loading models

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


rb_wr_features = [
 'spread_line',
 'div_game',
 'wind',
 'n_games_career',
 'n_games_season',
 'rushing_yards_mean_career',
 'rushing_yards_mean_season',
 'rushing_yards_mean_last5',
 'rushing_yards_last',
 'rushing_tds_mean_career',
 'rushing_tds_mean_season',
 'rushing_tds_mean_last5',
 'rushing_tds_last',
 'rushing_fumbles_mean_career',
 'rushing_fumbles_mean_season',
 'rushing_fumbles_mean_last5',
 'rushing_fumbles_last',
 'rushing_fumbles_lost_mean_career',
 'rushing_fumbles_lost_mean_season',
 'rushing_fumbles_lost_mean_last5',
 'rushing_fumbles_lost_last',
 'rushing_first_downs_mean_career',
 'rushing_first_downs_mean_season',
 'rushing_first_downs_mean_last5',
 'rushing_first_downs_last',
 'rushing_epa_mean_career',
 'rushing_epa_mean_season',
 'rushing_epa_mean_last5',
 'rushing_epa_last',
 'rushing_2pt_conversions_mean_career',
 'rushing_2pt_conversions_mean_season',
 'rushing_2pt_conversions_mean_last5',
 'rushing_2pt_conversions_last',
 'receptions_mean_career',
 'receptions_mean_season',
 'receptions_mean_last5',
 'receptions_last',
 'targets_mean_career',
 'targets_mean_season',
 'targets_mean_last5',
 'targets_last',
 'receiving_yards_mean_career',
 'receiving_yards_mean_season',
 'receiving_yards_mean_last5',
 'receiving_yards_last',
 'receiving_tds_mean_career',
 'receiving_tds_mean_season',
 'receiving_tds_mean_last5',
 'receiving_tds_last',
 'receiving_fumbles_mean_career',
 'receiving_fumbles_mean_season',
 'receiving_fumbles_mean_last5',
 'receiving_fumbles_last',
 'receiving_fumbles_lost_mean_career',
 'receiving_fumbles_lost_mean_season',
 'receiving_fumbles_lost_mean_last5',
 'receiving_fumbles_lost_last',
 'receiving_air_yards_mean_career',
 'receiving_air_yards_mean_season',
 'receiving_air_yards_mean_last5',
 'receiving_air_yards_last',
 'receiving_yards_after_catch_mean_career',
 'receiving_yards_after_catch_mean_season',
 'receiving_yards_after_catch_mean_last5',
 'receiving_yards_after_catch_last',
 'receiving_first_downs_mean_career',
 'receiving_first_downs_mean_season',
 'receiving_first_downs_mean_last5',
 'receiving_first_downs_last',
 'receiving_epa_mean_career',
 'receiving_epa_mean_season',
 'receiving_epa_mean_last5',
 'receiving_epa_last',
 'receiving_2pt_conversions_mean_career',
 'receiving_2pt_conversions_mean_season',
 'receiving_2pt_conversions_mean_last5',
 'receiving_2pt_conversions_last',
 'racr_mean_career',
 'racr_mean_season',
 'racr_mean_last5',
 'racr_last',
 'target_share_mean_career',
 'target_share_mean_season',
 'target_share_mean_last5',
 'target_share_last',
 'air_yards_share_mean_career',
 'air_yards_share_mean_season',
 'air_yards_share_mean_last5',
 'air_yards_share_last',
 'special_teams_tds_mean_career',
 'special_teams_tds_mean_season',
 'special_teams_tds_mean_last5',
 'special_teams_tds_last',
 'fantasy_points_ppr_mean_career',
 'fantasy_points_ppr_mean_season',
 'fantasy_points_ppr_mean_last5',
 'fantasy_points_ppr_last']

qb_features = ['player_last_fp',
 'spread_line',
 'div_game',
 'wind',
 'n_games_career',
 'n_games_season',
 'completions_mean_career',
 'completions_mean_season',
 'completions_mean_last5',
 'completions_last',
 'attempts_mean_career',
 'attempts_mean_season',
 'attempts_mean_last5',
 'attempts_last',
 'passing_yards_mean_career',
 'passing_yards_mean_season',
 'passing_yards_mean_last5',
 'passing_yards_last',
 'passing_tds_mean_career',
 'passing_tds_mean_season',
 'passing_tds_mean_last5',
 'passing_tds_last',
 'interceptions_mean_career',
 'interceptions_mean_season',
 'interceptions_mean_last5',
 'interceptions_last',
 'sacks_mean_career',
 'sacks_mean_season',
 'sacks_mean_last5',
 'sacks_last',
 'sack_yards_mean_career',
 'sack_yards_mean_season',
 'sack_yards_mean_last5',
 'sack_yards_last',
 'sack_fumbles_mean_career',
 'sack_fumbles_mean_season',
 'sack_fumbles_mean_last5',
 'sack_fumbles_last',
 'sack_fumbles_lost_mean_career',
 'sack_fumbles_lost_mean_season',
 'sack_fumbles_lost_mean_last5',
 'sack_fumbles_lost_last',
 'passing_air_yards_mean_career',
 'passing_air_yards_mean_season',
 'passing_air_yards_mean_last5',
 'passing_air_yards_last',
 'passing_yards_after_catch_mean_career',
 'passing_yards_after_catch_mean_season',
 'passing_yards_after_catch_mean_last5',
 'passing_yards_after_catch_last',
 'passing_first_downs_mean_career',
 'passing_first_downs_mean_season',
 'passing_first_downs_mean_last5',
 'passing_first_downs_last',
 'passing_epa_mean_career',
 'passing_epa_mean_season',
 'passing_epa_mean_last5',
 'passing_epa_last',
 'passing_2pt_conversions_mean_career',
 'passing_2pt_conversions_mean_season',
 'passing_2pt_conversions_mean_last5',
 'passing_2pt_conversions_last',
 'pacr_mean_career',
 'pacr_mean_season',
 'pacr_mean_last5',
 'pacr_last',
 'dakota_mean_career',
 'dakota_mean_season',
 'dakota_mean_last5',
 'dakota_last',
 'carries_mean_career',
 'carries_mean_season',
 'carries_mean_last5',
 'carries_last',
 'rushing_yards_mean_career',
 'rushing_yards_mean_season',
 'rushing_yards_mean_last5',
 'rushing_yards_last',
 'rushing_tds_mean_career',
 'rushing_tds_mean_season',
 'rushing_tds_mean_last5',
 'rushing_tds_last',
 'rushing_fumbles_mean_career',
 'rushing_fumbles_mean_season',
 'rushing_fumbles_mean_last5',
 'rushing_fumbles_last',
 'rushing_fumbles_lost_mean_career',
 'rushing_fumbles_lost_mean_season',
 'rushing_fumbles_lost_mean_last5',
 'rushing_fumbles_lost_last',
 'rushing_first_downs_mean_career',
 'rushing_first_downs_mean_season',
 'rushing_first_downs_mean_last5',
 'rushing_first_downs_last',
 'rushing_epa_mean_career',
 'rushing_epa_mean_season',
 'rushing_epa_mean_last5',
 'rushing_epa_last',
 'rushing_2pt_conversions_mean_career',
 'rushing_2pt_conversions_mean_season',
 'rushing_2pt_conversions_mean_last5',
 'rushing_2pt_conversions_last',
 'fantasy_points_ppr_mean_career',
 'fantasy_points_ppr_mean_season',
 'fantasy_points_ppr_mean_last5',
 'fantasy_points_ppr_last']


kicker_defense_features = ['n_games_career',
 'fantasy_points_ppr_mean_career',
 'fantasy_points_ppr_mean_season_k',
 'fantasy_points_ppr_mean_last5_k',
 'total_fg_made_mean_career',
 'total_fg_missed_mean_career',
 '40-49_fg_made_mean_career',
 '0-39_fg_made_mean_career',
 'missed_fg_50+_mean_career',
 'missed_fg_40-49_mean_career',
 'missed_fg_40-49_mean_season_k',
 'xp_attempt_19y_mean_career',
 'xp_attempt_19y_mean_season_k',
 'xp_attempt_19y_mean_last5_k',
 'xp_made_19y_mean_career',
 'xp_made_19y_mean_season_k',
 'xp_made_19y_mean_last5_k',
 'xp_attempt_33y_mean_career',
 'xp_attempt_33y_mean_season_k',
 'xp_attempt_33y_mean_last5_k',
 'xp_made_33y_mean_career',
 'xp_made_33y_mean_season_k',
 'xp_made_33y_mean_last5_k',
 'n_games_season_def',
 'fantasy_points_ppr_mean_season_def',
 'fantasy_points_ppr_mean_last5_def',
 '50+_fg_made_mean_last5_def',
 '40-49_fg_made_mean_last5_def',
 'xp_attempt_19y_mean_season_def',
 'xp_made_19y_mean_season_def',
 'xp_made_33y_mean_season_def',
 'wind']


class NFLModel:
    def __init__(self, position='QB', test_size=0.2, random_state=42, roster_data=None, pbp_df=None, schedules_df=None, weekly_df = None):
        """
        Initializes the NFLModel class.

        Parameters:
        - position: str, the position group for the model ('QB', 'Kicker', 'RW').
        - test_size: float, proportion of the dataset to include in the test split.
        - random_state: int, random seed for reproducibility.
        - roster_data: DataFrame, preloaded roster data.
        - pbp_df: DataFrame, preloaded play-by-play data.
        - schedules_df: DataFrame, preloaded schedules data.
        """
        self.position = position.upper()
        self.test_size = test_size
        self.random_state = random_state

        # Assign preloaded data or raise an error if not provided
        if roster_data is None or pbp_df is None or schedules_df is None:
            raise ValueError("Please provide preloaded data for roster_data, pbp_df, and schedules_df.")
        self.roster_data = roster_data
        self.pbp_df = pbp_df
        self.schedules_df = schedules_df
        self.weekly_df = weekly_df

        # Map positions to features and dataframes dynamically
        self.position_features_map = {
            'QB': {
                'features': qb_features,
                'generate_features_method': self.generate_qb_features,
                'features_df': 'qb_features_df'
            },
            'KICKER': {
                'features': kicker_defense_features,
                'generate_features_method': self.generate_features_kicker_defense,
                'features_df': 'kicker_defense_features_df'
            },
            'RW': {
                'features': rb_wr_features,
                'generate_features_method': self.generate_features_rb_wr,
                'features_df': 'rusher_receiver_features_df'
            }
        }

        if self.position not in self.position_features_map:
            raise ValueError("Invalid position. Please choose 'QB', 'Kicker', or 'RW'.")

        # Set features and dataframes dynamically
        self.features = self.position_features_map[self.position]['features']
        self.generate_features_method = self.position_features_map[self.position]['generate_features_method']
        self.features_df_name = self.position_features_map[self.position]['features_df']

        # Initialize placeholders for position-specific data
        self.features_df = self.generate_features()
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.lr_model = None
        self.rf_model = None
        self.lstm_model = None
        self.scaler = None
        self.results = {}


    def generate_features(self):
        """
        Calls the appropriate feature generation method based on the position.
        """
        self.generate_features_method()
        return getattr(self, self.features_df_name)


    def preprocess_data(self, target_variable='fantasy_points_ppr'):
        """
        Preprocesses the data for the specified position.

        Parameters:
        - target_variable: str, the target variable for prediction.

        Returns:
        - x_train, x_test, y_train, y_test: Scaled and split datasets.
        """
        # if self.features_df is None:
        #     raise ValueError("Features not generated. Call generate_features() first.")

        # Separate features and target
        df = self.features_df[self.features + [target_variable]].copy()
        df = self.get_dummy_variables(df)

        self.x = df.drop(columns=[target_variable])
        self.y = df[target_variable]

        # Split data into training and testing sets
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Align the training and testing data
        x_train, x_test = x_train_raw.align(x_test_raw, join='left', axis=1, fill_value=0)

        # Standardize data
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        self.x_train, self.x_test, self.y_train, self.y_test = (
            x_train_scaled,
            x_test_scaled,
            y_train.values.ravel(),
            y_test.values.ravel(),
        )


    def train_evaluate_model(self, model_type='LinearRegression'):
        """
        Trains the specified model for the current position and evaluates its performance.

        Parameters:
        - model_type: str, the type of model to train ('LinearRegression', 'RandomForest').

        Raises:
        - ValueError: If the model type is unsupported.
        """
        # Initialize the model based on the type
        if model_type == 'LinearRegression':
            self.lr_model = LinearRegression()
            model = self.lr_model
        elif model_type == 'RandomForest':
            self.rf_model = RandomForestRegressor(random_state=self.random_state)
            model = self.rf_model
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'LinearRegression' or 'RandomForest'.")

        # Train the model
        print(f"Training {model_type} model...")
        model.fit(self.x_train, self.y_train)
        print(f"{model_type} model trained successfully.")

        # Evaluate the model
        self.evaluate_model(model=model, model_name=model_type)


    def evaluate_model(self, model=None, model_name=None):
        """
        Evaluates the trained model using the test data.

        Parameters:
        - model: Trained machine learning model to evaluate.
        - model_name: str, name of the model (e.g., 'LinearRegression', 'RandomForest').

        Returns:
        - metrics: dict containing evaluation metrics (MAE, MSE, R2).

        Raises:
        - ValueError: If no model is provided.
        """
        if model is None:
            raise ValueError("Model has not been trained yet. Provide a valid model to evaluate.")

        # Make predictions
        print(f"Evaluating {model_name} model...")
        y_pred = model.predict(self.x_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Log the results
        print(f"{model_name} Evaluation Results - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

        # Store metrics
        metrics = {'MAE': mae, 'MSE': mse, 'R2': r2}
        self.results[model_name] = metrics


    def calc_agg_stats_adapted(self,group, fields, career=True):
        # Adjusting the original calc_agg function to fit this set of data
        df = pd.DataFrame(index=group.index)
        
        group_sorted = group.sort_values(['player_id', 'season', 'game_date'])

        df['n_games_career'] = group_sorted.groupby('player_id').cumcount() + 1
        df['n_games_season'] = group_sorted.groupby(['player_id', 'season']).cumcount() + 1

        for field in fields:
            if career:
                # Career stats
                df[f'{field}_mean_career'] = group_sorted.groupby('player_id')[field].expanding().mean().reset_index(level=0, drop=True).shift()

            # Season stats
            df[f'{field}_mean_season'] = group_sorted.groupby(['player_id', 'season'])[field].expanding().mean().reset_index(level=[0, 1], drop=True).shift()

            # Last 5 games
            df[f'{field}_mean_last5'] = group_sorted.groupby('player_id')[field].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift()
            )

            # Last game 
            df[f'{field}_last'] = group_sorted.groupby('player_id')[field].shift()

        return df


    def calc_agg_stats_kicker_d(self,group, fields, career=True):
        """
        Calculate aggregate statistics for each player over their career and season,
        including prior season means, rolling averages, and cumulative counts.

        Parameters:
        - group: DataFrame grouped by player or other identifier.
        - fields: List of fields to calculate statistics on.
        - career: Boolean indicating whether to calculate career-level stats.

        Returns:
        - DataFrame with calculated aggregate statistics.
        """
        # Ensure 'game_date' is datetime
        group['game_date'] = pd.to_datetime(group['game_date'], errors='coerce')
        
        # Sort the group chronologically
        group_sorted = group.sort_values('game_date')
        
        # Initialize the result DataFrame
        result = pd.DataFrame(index=group_sorted.index)
        
        # Calculate cumulative game counts
        if career:
            # Career game count (number of games up to current point, excluding current game)
            result['n_games_career'] = np.arange(len(group_sorted))
        
        # Season game count
        result['n_games_season'] = group_sorted.groupby('season').cumcount()

        # Loop over each field to calculate aggregate stats
        for field in fields:
            if career:
                # Career mean up to the previous game (excluding current game)
                result[f'{field}_mean_career'] = (
                    group_sorted[field]
                    .expanding()
                    .mean()
                    .shift()
                )
            
            # Season mean up to the previous game (excluding current game)
            result[f'{field}_mean_season'] = (
                group_sorted.groupby('season')[field]
                .expanding()
                .mean()
                .shift()
                .reset_index(level=0, drop=True)
            )
            
            # # Prior season mean (mean of the entire previous season)
            # result[f'{field}_mean_prior_season'] = (
            #     group_sorted.groupby('season')[field]
            #     .transform('mean')
            #     .shift()
            # )
            
            # Rolling mean for the last 5 games up to the previous game (excluding current game)
            result[f'{field}_mean_last5'] = (
                group_sorted[field]
                .rolling(window=5, min_periods=1)
                .mean()
                .shift()
            )
        
        # Combine the result with the original group_sorted DataFrame
        combined = pd.concat([group_sorted, result], axis=1)
        
        return combined


    def generate_features_rb_wr(self):
        '''
        Method that returns the dataframe of a certain position group with aggregated features. 

        Parameters: 
        roster_data,pbp_df,weekly_df,schedules_df = Dataframes returned from load_data()

        postition = str of position group to generate features for.


        Returns:
        df_combined = Dataframe with calculated features for a given position group.
        
        '''
        rb_wr_data = self.weekly_df[self.weekly_df['position'].isin(['WR','RB','TE','FB'])][['player_id','player_name','season', 'week','position','recent_team',
       'season_type', 'opponent_team','sack_fumbles', 
        'carries', 'rushing_yards',
       'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
       'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
       'receptions', 'targets', 'receiving_yards', 'receiving_tds',
       'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
       'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
       'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share',
        'special_teams_tds',  'fantasy_points_ppr']]

        rb_wr_data.rename(columns = {'recent_team':'team'}, inplace = True)


        rb_wr_data_prior = rb_wr_data[['week','season','team','fantasy_points_ppr','player_name','position']].copy()

        rb_wr_data_prior['week'] = rb_wr_data_prior['week'] + 1

        rb_wr_data_prior.rename(columns = {'fantasy_points_ppr':'player_last_fp'},inplace = True )


        rb_wr_data = rb_wr_data.merge(rb_wr_data_prior, on = ['week','player_name','season','team','position'], how = 'left')


        #Aggregating defensive performance for home and away teams
        home_points = self.schedules_df[['game_id', 'season', 'week', 'home_team', 'away_score']].rename(
            columns={'home_team': 'team', 'away_score': 'points_allowed'}
        )
        home_points['adj_week'] = home_points['week']+1
        home_points.head()

        away_points = self.schedules_df[['game_id', 'season', 'week', 'away_team', 'home_score']].rename(
            columns={'away_team': 'team', 'home_score': 'points_allowed'}
        )
        away_points['adj_week'] = away_points['week']+1
        away_points.head()


        # Combine both home and dataframes
        points_allowed_df = pd.concat([home_points, away_points], ignore_index=True)

        # Sort by season, team, and week to calculate cumulative points allowed up to each week
        points_allowed_df = points_allowed_df.sort_values(by=['season', 'team', 'week', 'adj_week'])

        # Calculate cumulative points allowed for each team
        points_allowed_df['cumulative_points_allowed'] = points_allowed_df.groupby(['season', 'team'])['points_allowed'].cumsum()

        points_allowed_df.rename(columns={
            'team': 'defense_name'
        }, inplace=True)

        #Creating defensive ranking based on cumulative points
        points_allowed_df['defensive_rank'] = points_allowed_df.groupby(['season', 'week'])['cumulative_points_allowed'].rank(method='min', ascending=True)


        points_allowed_df.rename(columns = {'defense_name':'opponent_team'},inplace = True)


        points_allowed_df = points_allowed_df.merge(rb_wr_data[['position','fantasy_points_ppr','week','season','opponent_team']] , on = ['week','season','opponent_team'], how = 'left')

        points_allowed_df.rename(columns = {'week':'week_of_interest','adj_week':'week', 'fantasy_points_ppr':'prior_allowed_fp'}, inplace = True)

        pbp_copy = self.pbp_df.copy()
        
        pbp_copy['team'] = pbp_copy['posteam']

        rb_wr_data = rb_wr_data.merge(pbp_copy.groupby(['game_id','week','season','spread_line','game_date','div_game','team']).agg({'wind':'first','roof':'first'}).reset_index(), on = ['team','week','season'], how = 'inner')


        cols = ['rushing_yards', 'rushing_tds', 'rushing_fumbles',
            'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
            'rushing_2pt_conversions', 'receptions', 'targets', 'receiving_yards',
            'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
            'receiving_air_yards', 'receiving_yards_after_catch',
            'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions',
            'racr', 'target_share', 'air_yards_share', 'special_teams_tds',
            'fantasy_points_ppr'
            ]

        rb_wr_data_lag_cols = rb_wr_data.groupby(
                    ['player_id'], 
                    group_keys=False
                ).apply(
                    self.calc_agg_stats_adapted, 
                    fields=cols
                ).reset_index(drop=True)


        rb_wr_data = rb_wr_data.join(rb_wr_data_lag_cols)


        self.rusher_receiver_features_df = rb_wr_data.fillna(0)


    def generate_qb_features(self):
        
        qb_data = self.weekly_df[self.weekly_df['position'] == 'QB'][['player_id','player_name','season', 'week','position','recent_team',
       'season_type', 'opponent_team', 'completions', 'attempts',
       'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards',
       'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
       'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
       'passing_2pt_conversions', 'pacr', 'dakota', 'carries', 'rushing_yards',
       'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
       'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions','fantasy_points_ppr']]

        qb_data.rename(columns = {'recent_team':'team'}, inplace = True)


        qb_data_prior = qb_data[['week','season','team','fantasy_points_ppr','player_name']].copy()

        qb_data_prior['week'] = qb_data_prior['week'] + 1

        qb_data_prior.rename(columns = {'fantasy_points_ppr':'player_last_fp'},inplace = True )


        qb_data = qb_data.merge(qb_data_prior, on = ['week','player_name','season','team'], how = 'left')


        #Aggregating defensive performance for home and away teams
        home_points = self.schedules_df[['game_id', 'season', 'week', 'home_team', 'away_score']].rename(
            columns={'home_team': 'team', 'away_score': 'points_allowed'}
        )
        home_points['adj_week'] = home_points['week']+1
        home_points.head()

        away_points = self.schedules_df[['game_id', 'season', 'week', 'away_team', 'home_score']].rename(
            columns={'away_team': 'team', 'home_score': 'points_allowed'}
        )
        away_points['adj_week'] = away_points['week']+1
        away_points.head()


        # Combine both home and dataframes
        points_allowed_df = pd.concat([home_points, away_points], ignore_index=True)

        # Sort by season, team, and week to calculate cumulative points allowed up to each week
        points_allowed_df = points_allowed_df.sort_values(by=['season', 'team', 'week', 'adj_week'])

        # Calculate cumulative points allowed for each team
        points_allowed_df['cumulative_points_allowed'] = points_allowed_df.groupby(['season', 'team'])['points_allowed'].cumsum()

        points_allowed_df.rename(columns={
            'team': 'defense_name'
        }, inplace=True)

        #Creating defensive ranking based on cumulative points
        points_allowed_df['defensive_rank'] = points_allowed_df.groupby(['season', 'week'])['cumulative_points_allowed'].rank(method='min', ascending=True)


        points_allowed_df.rename(columns = {'defense_name':'opponent_team'},inplace = True)


        points_allowed_df = points_allowed_df.merge(qb_data[['position','fantasy_points_ppr','week','season','opponent_team']] , on = ['week','season','opponent_team'], how = 'left')

        points_allowed_df.rename(columns = {'week':'week_of_interest','adj_week':'week', 'fantasy_points_ppr':'prior_allowed_fp'}, inplace = True)



        qb_data = qb_data.merge(self.pbp_df.groupby(['passer_player_id','game_id','week','season','passer_player_name','spread_line','game_date','div_game']).agg({'wind':'first','roof':'first'}).reset_index().rename(columns = {'passer_player_name':'player_name'}), on = ['player_name','week','season'], how = 'inner')


        cols = ['completions',
        'attempts',
        'passing_yards',
        'passing_tds',
        'interceptions',
        'sacks',
        'sack_yards',
        'sack_fumbles',
        'sack_fumbles_lost',
        'passing_air_yards',
        'passing_yards_after_catch',
        'passing_first_downs',
        'passing_epa',
        'passing_2pt_conversions',
        'pacr',
        'dakota',
        'carries',
        'rushing_yards',
        'rushing_tds',
        'rushing_fumbles',
        'rushing_fumbles_lost',
        'rushing_first_downs',
        'rushing_epa',
        'rushing_2pt_conversions',
        'fantasy_points_ppr']

        qb_data_lag_cols = qb_data.groupby(
                    ['player_id'], 
                    group_keys=False
                ).apply(
                    self.calc_agg_stats_adapted, 
                    fields=cols
                ).reset_index(drop=True)


        qb_data = qb_data.join(qb_data_lag_cols)


        self.qb_features_df = qb_data.fillna(0)


    def generate_features_kicker_defense(self):
        # Filter rows where 'kicker_player_name' is not null and the play type is relevant
        df_kicker_pbp = self.pbp_df.loc[
            self.pbp_df['kicker_player_name'].notnull() & 
            self.pbp_df['play_type'].isin(['field_goal', 'extra_point', 'kickoff'])
        ].copy() 

        # Ensure 'posteam' and 'defteam' columns exist
        if 'posteam' in df_kicker_pbp.columns and 'defteam' in df_kicker_pbp.columns:
            # Create a mask for kickoff plays
            kickoff_mask = df_kicker_pbp['play_type'] == 'kickoff'

            # Log the number of kickoff plays being processed
            print(f"Swapping 'posteam' and 'defteam' for {kickoff_mask.sum()} kickoff plays...")

            # Swap values using the mask
            df_kicker_pbp.loc[kickoff_mask, ['posteam', 'defteam']] = (
                df_kicker_pbp.loc[kickoff_mask, ['defteam', 'posteam']].values
            )

            print("Swap complete.")
        else:
            print("Error: Required columns 'posteam' and 'defteam' are missing from the DataFrame.")

        # Convert 'game_date' column to datetime format, with error handling
        try:
            df_kicker_pbp['game_date'] = pd.to_datetime(df_kicker_pbp['game_date'], errors='coerce')
            if df_kicker_pbp['game_date'].isnull().any():
                print("Warning: Some 'game_date' entries could not be converted and have been set to NaT.")
        except Exception as e:
            print(f"An error occurred while converting 'game_date' to datetime: {e}")

        # Final log for confirmation
        print("Data processing for 'df_kicker_pbp' completed.")

        # Set extra point distance based on year and create flags for XP attempts and success
        df_kicker_pbp['xp_distance'] = np.where(df_kicker_pbp['game_date'].dt.year < 2015, 19, 33)
        df_kicker_pbp["xp_attempt"] = df_kicker_pbp["extra_point_result"].notnull()
        df_kicker_pbp["xp_made"] = df_kicker_pbp["extra_point_result"] == "good"

        # Create flags for successful and attempted XPs by distance
        df_kicker_pbp["xp_made_33y"] = df_kicker_pbp["xp_made"] & (df_kicker_pbp["xp_distance"] == 33)
        df_kicker_pbp["xp_made_19y"] = df_kicker_pbp["xp_made"] & (df_kicker_pbp["xp_distance"] == 19)
        df_kicker_pbp["xp_attempt_33y"] = df_kicker_pbp["xp_attempt"] & (df_kicker_pbp["xp_distance"] == 33)
        df_kicker_pbp["xp_attempt_19y"] = df_kicker_pbp["xp_attempt"] & (df_kicker_pbp["xp_distance"] == 19)

        # Field goal (FG) results and distance-based flags
        df_kicker_pbp["50+_fg_made"] = (df_kicker_pbp["field_goal_result"] == "made") & (df_kicker_pbp["kick_distance"] >= 50)
        df_kicker_pbp["40-49_fg_made"] = (df_kicker_pbp["field_goal_result"] == "made") & (df_kicker_pbp["kick_distance"].between(40, 49))
        df_kicker_pbp["0-39_fg_made"] = (df_kicker_pbp["field_goal_result"] == "made") & (df_kicker_pbp["kick_distance"] < 40)

        # Missed FG flags by distance
        df_kicker_pbp["missed_fg_0-39"] = (df_kicker_pbp["field_goal_result"] == "missed") & (df_kicker_pbp["kick_distance"] < 40)
        df_kicker_pbp["missed_fg_40-49"] = (df_kicker_pbp["field_goal_result"] == "missed") & (df_kicker_pbp["kick_distance"].between(40, 49))
        df_kicker_pbp["missed_fg_50+"] = (df_kicker_pbp["field_goal_result"] == "missed") & (df_kicker_pbp["kick_distance"] >= 50)

        # Total FGs made and missed
        df_kicker_pbp["total_fg_made"] = df_kicker_pbp[["50+_fg_made", "40-49_fg_made", "0-39_fg_made"]].sum(axis=1)
        df_kicker_pbp["total_fg_missed"] = df_kicker_pbp[["missed_fg_0-39", "missed_fg_40-49", "missed_fg_50+"]].sum(axis=1)

        # Calculate fantasy points based on custom scoring system


        # Optional: Drop any rows with NaN values in the calculated columns
        # df_kicker_pbp.dropna(subset=["fantasy_points"], inplace=True)

        # Log completion message
        print("Kicker play-by-play data processing completed successfully.")
        df_kicker_game_level_stadium = df_kicker_pbp.groupby(['game_id', 'game_date', 'week', 'season', 'stadium']).agg({
            # Game level
            'home_team': 'first',
            'roof': 'first',
            'temp': 'first',
            'wind': 'first',
        }).sort_values(by=['game_date'], ascending=False)

        df_kicker_game_level = df_kicker_pbp.groupby(['game_id', 'game_date', 'week', 'season', 'posteam', 'defteam', 'kicker_player_name', 'kicker_player_id'], as_index=False).agg({
            # Game level
            'home_team': 'first',
            'away_team': 'first',

            # Play level
            'total_fg_made': 'sum',
            'total_fg_missed': 'sum',
            '50+_fg_made': 'sum',
            '40-49_fg_made': 'sum',
            '0-39_fg_made': 'sum',
            'missed_fg_0-39': 'sum',
            'missed_fg_40-49': 'sum',
            'missed_fg_50+': 'sum',
            'xp_attempt_19y': 'sum',
            'xp_made_19y': 'sum',
            'xp_attempt_33y': 'sum',
            'xp_made_33y': 'sum',
        })
        df_kicker_game_level.rename(columns = {'defteam':'opponent_team','kicker_player_id':'player_id'} , inplace = True )
        df_kicker_game_level = df_kicker_game_level.merge(self.weekly_df[['player_id','week','season','fantasy_points_ppr']], how = 'inner',on = ['player_id','week','season'])

        df_kicker_game_level["home"] = df_kicker_game_level["home_team"] == df_kicker_game_level["posteam"]
        # df_kicker_game_level.drop(columns=['home_team', 'away_team'], inplace=True)
        # Define the fields for which you want to calculate aggregate statistics
        kicker_fields = [
            'fantasy_points_ppr', 
            'total_fg_made', 
            'total_fg_missed', 
            '50+_fg_made', 
            '40-49_fg_made', 
            '0-39_fg_made', 
            'missed_fg_50+', 
            'missed_fg_40-49', 
            'missed_fg_0-39', 
            'xp_attempt_19y', 
            'xp_made_19y', 
            'xp_attempt_33y', 
            'xp_made_33y'
        ]

        # Apply the 'calc_agg_stats' function to each kicker's data
        df_kicker_game_level_agg = df_kicker_game_level.groupby(
            ['kicker_player_name', 'player_id'], 
            group_keys=False
        ).apply(
            self.calc_agg_stats_kicker_d, 
            fields=kicker_fields
        ).reset_index(drop=True).round(2)
        df_kicker_game_level_agg = df_kicker_game_level_agg.drop(columns=df_kicker_game_level_agg.loc[:, "fantasy_points_ppr":"home"].columns)

        df_kicker_game_level_agg_by_game = df_kicker_game_level.groupby(['game_id', 'game_date', 'week', 'season', 'posteam', 'opponent_team','home_team','away_team'], as_index=False).agg({
            # Play level
            'fantasy_points_ppr': 'sum',
            'total_fg_made': 'sum',
            'total_fg_missed': 'sum',
            '50+_fg_made': 'sum',
            '40-49_fg_made': 'sum',
            '0-39_fg_made': 'sum',
            'missed_fg_0-39': 'sum',
            'missed_fg_40-49': 'sum',
            'missed_fg_50+': 'sum',
            'xp_attempt_19y': 'sum',
            'xp_made_19y': 'sum',
            'xp_attempt_33y': 'sum',
            'xp_made_33y': 'sum',
        })

        # Group by 'defteam' and apply the 'calc_agg_stats' function
        df_kicker_game_level_agg_by_def = df_kicker_game_level_agg_by_game.groupby(
            ['opponent_team'], 
            group_keys=False
        ).apply(
            self.calc_agg_stats_kicker_d, 
            fields=kicker_fields, 
            career=False 
        ).reset_index(drop=True).round(2)
        df_kicker_game_level_agg_by_def = df_kicker_game_level_agg_by_def.drop(columns=df_kicker_game_level_agg_by_def.loc[:, "fantasy_points_ppr":"xp_made_33y"].columns)

        # Merge kicker aggregate stats with defensive team stats
        df_combined = pd.merge(
            df_kicker_game_level_agg,
            df_kicker_game_level_agg_by_def,
            on=['game_id', 'game_date', 'week', 'season', 'posteam', 'opponent_team'],
            how='left',
            suffixes=('_k', '_def')
        )

        # Merge with stadium data
        df_combined = pd.merge(
            df_combined,
            df_kicker_game_level_stadium,
            on=['game_id', 'game_date', 'week', 'season'],
            how='left'
        )

        # Merge with original kicker game level data to include 'fantasy_points'
        df_combined = pd.merge(
            df_combined,
            df_kicker_game_level[['game_id', 'fantasy_points_ppr', 'player_id']],
            on=['game_id', 'player_id'],
            how='left'
        )

        # Drop redundant columns if necessary
        # columns_to_drop = ['home_team']
        # df_combined.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Reset index
        df_combined.reset_index(drop=True, inplace=True)

        # Log completion message
        print("DataFrames merged successfully into 'df_combined'.")


        self.kicker_defense_features_df = df_combined.fillna(0)


    def get_dummy_variables(self,df, drop_first=True, dummy_na=False):
        """
        Converts non-numerical columns in a DataFrame to dummy variables.

        Parameters:
        - df: pandas DataFrame
            The input DataFrame containing the data.
        - drop_first: bool, default=False
            Whether to drop the first level of categorical variables to avoid the dummy variable trap.
        - dummy_na: bool, default=False
            Add a column to indicate NaNs, if False NaNs are ignored.

        Returns:
        - df_dummies: pandas DataFrame
            The DataFrame with non-numeric columns converted to dummy variables.
        """
        # Identify non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()

        # If there are no non-numeric columns, return the original DataFrame
        if not non_numeric_cols:
            print("No non-numerical columns to convert.")
            return df.copy()

        # Convert categorical variables to dummy variables
        df_dummies = pd.get_dummies(df, columns=non_numeric_cols, drop_first=drop_first, dummy_na=dummy_na)

        return df_dummies
    

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
    

    def tune_random_forest(self):
        """
        Performs hyperparameter tuning for the Random Forest model using GridSearchCV.
        """
        # Define the parameter grid without 'auto' for max_features
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2', 0.2, 0.5],
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
            verbose=0
        )

        # Fit GridSearchCV
        grid_search.fit(self.x_train, self.y_train)

        # Retrieve the best parameters and set the model
        self.rf_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best MAE score: {-grid_search.best_score_:.4f}")

        # Evaluate the tuned model
        self.evaluate_model(self, model=self.rf_model, model_name='RandomForest_Tuned')


    def build_and_train_lstm(self, units=64, dropout_rate=0.3, epochs=100, batch_size=32, patience=10):
        """
        Builds, compiles, and trains the LSTM model using the training data.

        Parameters:
        - units: int, number of units in the LSTM layers.
        - dropout_rate: float, dropout rate for regularization.
        - epochs: int, number of epochs to train the model.
        - batch_size: int, number of samples per gradient update.
        - patience: int, number of epochs with no improvement after which training will be stopped.
        """

        # Reshape data for LSTM input (samples, time steps, features)
        self.x_train_lstm = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1]))
        self.x_test_lstm = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1]))

        # Build and compile the model
        self.lstm_model = Sequential([
            LSTM(units, input_shape=(self.x_train_lstm.shape[1], self.x_train_lstm.shape[2]), activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        print("LSTM model built and compiled.")

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.lstm_model.fit(
            self.x_train_lstm, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0  # Set verbose to 1 for detailed training logs
        )
        print("LSTM training completed.")


    def evaluate_lstm(self):
        """
        Evaluates the trained LSTM model on the test data.
        Adds MAE, MSE, and R² metrics to the results dictionary.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been trained yet.")
        
        # Evaluate LSTM model
        lstm_test_loss, lstm_test_mae = self.lstm_model.evaluate(self.x_test_lstm, self.y_test, verbose=0)
        lstm_predictions = self.lstm_model.predict(self.x_test_lstm).flatten()

        # Calculate additional metrics
        lstm_test_mse = mean_squared_error(self.y_test, lstm_predictions)
        lstm_test_r2 = r2_score(self.y_test, lstm_predictions)

        # Store metrics in results
        self.results['LSTM'] = {'MAE': lstm_test_mae, 'MSE': lstm_test_mse, 'R2': lstm_test_r2, 'Loss': lstm_test_loss}
        print(f"LSTM Test MAE: {lstm_test_mae:.2f}, MSE: {lstm_test_mse:.2f}, R²: {lstm_test_r2:.2f}")
    


    def evaluate_ensemble(self):
        """
        Evaluates an ensemble model that averages predictions from both
        the trained LSTM and Random Forest models.
        """
        if self.rf_model is None or self.lstm_model is None:
            raise ValueError("Both Random Forest and LSTM models must be trained before evaluating the ensemble.")
        rf_predictions = self.rf_model.predict(self.x_test)
        lstm_predictions = self.lstm_model.predict(self.x_test_lstm).flatten()
        ensemble_predictions = (lstm_predictions + rf_predictions) / 2
        ensemble_mae = mean_absolute_error(self.y_test, ensemble_predictions)
        ensemble_mse = mean_squared_error(self.y_test, ensemble_predictions)
        ensemble_r2 = r2_score(self.y_test, ensemble_predictions)
        self.results['Ensemble'] = {'MAE': ensemble_mae, 'MSE': ensemble_mse, 'R2': ensemble_r2}
        print(f"Ensemble Test MAE: {ensemble_mae:.2f}")


    def get_results(self):
        """
        Returns the evaluation results as a pandas DataFrame.
        """
        return pd.DataFrame(self.results)
    

    def save_model(self, model, filename):
        """
        Saves the trained model to the 'nfl_models' folder. Creates the folder if it doesn't exist.

        Parameters:
        - model: The trained model object.
        - filename: str, the file name to save the model.
        """
        # Ensure the folder 'nfl_models' exists
        folder = 'nfl_models'
        os.makedirs(folder, exist_ok=True)

        # Construct the full file path
        filepath = os.path.join(folder, filename)

        # Save the model
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")


    def load_model(self, filename):
        """
        Loads a trained model from the 'nfl_models' folder. Ensures the file exists.

        Parameters:
        - filename: str, the file name of the saved model.

        Returns:
        - model: The loaded model object.

        Raises:
        - FileNotFoundError: If the model file does not exist.
        """
        # Construct the full file path
        folder = 'nfl_models'
        filepath = os.path.join(folder, filename)

        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} does not exist.")

        # Load the model
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def process_predictions(self, ensemble=False, save_to_file=None):
        """
        Generates predictions using trained models and optionally saves them to a CSV file.
        Supports ensemble predictions by combining Random Forest and LSTM outputs.

        Parameters:
        - ensemble: bool, whether to use an ensemble of Random Forest and LSTM models.
        - save_to_file: str or None, path to save the predictions as a CSV file. If None, does not save to file.

        Returns:
        - predictions_df: DataFrame with predictions for the entire dataset.
        """
        print("Generating predictions...")

        # Check if models are trained
        if not hasattr(self, 'rf_model') or not hasattr(self, 'lstm_model'):
            raise ValueError("Models are not trained. Please train the models before generating predictions.")

        # Scale the data for predictions
        x_scaled = self.scaler.transform(self.x)

        # Prepare the DataFrame for predictions
        predictions_df = self.features_df.copy()

        # Generate predictions
        if ensemble:
            print("Generating ensemble predictions...")

            # Reshape for LSTM
            x_lstm = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))

            # Random Forest predictions
            rf_preds = self.rf_model.predict(x_scaled)

            # LSTM predictions
            lstm_preds = self.lstm_model.predict(x_lstm).flatten()

            # Combine predictions (average)
            predictions_df['predicted_fantasy'] = (rf_preds + lstm_preds) / 2
        else:
            print("Using Random Forest predictions...")
            predictions_df['predicted_fantasy'] = self.rf_model.predict(x_scaled)

        # Save predictions to a CSV file if requested
        if save_to_file:
            print(f"Saving predictions to file: {save_to_file}...")
            predictions_df.to_csv(save_to_file, index=False)
            print(f"Predictions successfully saved to {save_to_file}.")

        return predictions_df


                        


