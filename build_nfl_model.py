import warnings 
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import datetime as dt
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


rb_wr_features = [
    'prior_ssn_avg_fp',
    'n_games_career',
    'n_games_season',
    'fantasy_points_mean_career', 
    'fantasy_points_mean_season',
    'fantasy_points_total_season',
    'fantasy_points_mean_last5',
    'fantasy_points_total_last5',
    'reception_mean_career',
    'reception_mean_season',
    'reception_total_season',
    'reception_mean_last5',
    'reception_total_last5',
    'reception_last',
    'rushing_yards_mean_season',
    'rushing_yards_last',
    'touchdown_total_season',
    'touchdown_total_last5',
    'receiving_yards_mean_career',
    'receiving_yards_total_career',
    'receiving_yards_mean_season',
    'receiving_yards_total_season',
    'receiving_yards_mean_last5',
    'receiving_yards_total_last5',
    'fumble_mean_career',
    'passing_yards_mean_career',
    'passing_yards_total_career',
    'passing_yards_mean_season',
    'passing_yards_total_season',
    'passing_yards_mean_last5',
    'passing_yards_total_last5',
    'pass_touchdown_mean_career',
    'pass_touchdown_total_career',
    'two_points_total_career',
    'points_allowed_mean_season',
    'points_allowed_mean_last5']

qb_features = ['div_game',
 'wind',
 'prior_ssn_avg_fp',
 'home_flag',
 'fantasy_points_mean_season',
 'interception_total_season',
 'interception_total_last5',
 'interception_last',
 'rush_attempt_total_season',
 'complete_pass_mean_career',
 'complete_pass_mean_season',
 'complete_pass_last',
 'rushing_yards_total_career',
 'rushing_yards_mean_last5',
 'receiving_yards_mean_career',
 'receiving_yards_mean_season',
 'receiving_yards_mean_last5',
 'fumble_total_season',
 'fumble_total_last5',
 'pass_touchdown_mean_career',
 'pass_touchdown_total_career',
 'pass_touchdown_total_last5',
 'pass_touchdown_last',
 'two_points_mean_career',
 'two_points_total_season',
 'two_points_total_last5',
 'points_allowed_mean_season',
 'points_allowed_mean_last5']


kicker_defense_features = ['n_games_career',
 'fantasy_points_mean_career',
 'fantasy_points_mean_season_k',
 'fantasy_points_mean_last5_k',
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
 'fantasy_points_mean_season_def',
 'fantasy_points_mean_last5_def',
 '50+_fg_made_mean_last5_def',
 '40-49_fg_made_mean_last5_def',
 'xp_attempt_19y_mean_season_def',
 'xp_made_19y_mean_season_def',
 'xp_made_33y_mean_season_def',
 'wind']

class NFLModel:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the NFLModel class.

        Parameters:
        - test_size: float, proportion of the dataset to include in the test split.
        - random_state: int, random seed for reproducibility.
        """
        self.target_variable = 'fantasy_points'
        self.test_size = test_size
        self.random_state = random_state
        self.roster_data,self.pbp_df,self.schedules_df = self.load_data()
        self.rb_wr_features = rb_wr_features
        self.kicker_defense_features = kicker_defense_features
        self.qb_features = qb_features
        self.kicker_defense_features_df = None
        self.qb_features_df = None
        self.rusher_receiver_features_df = None
        self.qb_x_train = None
        self.qb_x_test = None
        self.qb_y_train = None
        self.qb_y_test = None
        self.qb_model = None
        self.rb_wr_x_train = None
        self.rb_wr_x_test = None
        self.rb_wr_y_train = None
        self.rb_wr_y_test = None
        self.rb_wr_model = None
        self.kicker_defense_x_train = None
        self.kicker_defense_x_test = None
        self.kicker_defense_y_train = None
        self.kicker_defense_y_test = None
        self.kicker_defense_model = None


    def calc_agg_stats(self,group, fields, career=True):
        '''Helper function used to generate lagged stats within various windows of time
        '''

        # Create a copy to avoid modifying the original
        # df = pd.DataFrame({'game_date': group['game_date']}, index=group.index)
        df = pd.DataFrame(index=group.index)
        
        # Sort chronologically
        group_sorted = group.sort_values('game_date')

        # Calculate the number of unique games for career, season, and prior season
        if career:
            df['n_games_career'] = range(len(group_sorted))

        df['n_games_season'] = group_sorted.groupby(
            group_sorted.index.get_level_values('season')
        ).cumcount()

        # df['n_games_prior_season'] = group_sorted.groupby(
        #     group_sorted.index.get_level_values('season')
        # ).transform('size').shift()



        # Calculate aggregate stats
        for field in fields:
            if career:
                # Career stats
                df[f'{field}_mean_career'] = group_sorted[field].transform(lambda x: x.expanding().mean().shift())
                df[f'{field}_total_career'] = group_sorted[field].transform(lambda x: x.expanding().sum().shift())
            
            # Season stats
            df[f'{field}_mean_season'] = group_sorted.groupby([group_sorted.index.get_level_values('season')])[field].transform(lambda x: x.expanding().mean().shift())
            df[f'{field}_total_season'] = group_sorted.groupby([group_sorted.index.get_level_values('season')])[field].transform(lambda x: x.expanding().sum().shift())

            # # Prior season stats
            # df[f'{field}_mean_prior_season'] = group_sorted.groupby([group_sorted.index.get_level_values('season') - 1])[field].transform('mean')
            
            # Last 5 games
            df[f'{field}_mean_last5'] = group_sorted[field].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift())
            df[f'{field}_total_last5'] = group_sorted[field].transform(lambda x: x.rolling(window=5, min_periods=1).sum().shift())
            # Last Game
            df[f'{field}_last'] = group_sorted[field].shift()
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


    def load_data(self):
        '''
        Method returns four dataframes that will be used to generate features.

        Paramaters: None

        Returns = roster_data, pbp_df, weekly_df,schedules_df
        
        '''
        self.roster_data = nfl.import_seasonal_rosters([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999])
        self.pbp_df = pd.DataFrame(nfl.import_pbp_data([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]))
        self.schedules_df = pd.DataFrame(nfl.import_schedules([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]))

        return self.roster_data,self.pbp_df,self.schedules_df

    def generate_features_rb_wr(self):
        '''
        Method that returns the dataframe of a certain position group with aggregated features. 

        Parameters: 
        roster_data,pbp_df,weekly_df,schedules_df = Dataframes returned from load_data()

        postition = str of position group to generate features for.


        Returns:
        df_combined = Dataframe with calculated features for a given position group.
        
        '''

       
        team = self.roster_data[['season','player_id','team','depth_chart_position']]

        receiver_rusher_stats =  self.pbp_df[(self.pbp_df['receiver_player_id'].notnull()) | (self.pbp_df['rusher_player_id'].notnull())]
                                

        receiver_rusher_stats['two_points'] = np.where(receiver_rusher_stats['two_point_conv_result'] == 'success',1,0)
                                
        receiver_rusher_stats.rename(columns = {'complete_pass':'reception'},inplace = True)



        receiver_stats= receiver_rusher_stats.groupby(['game_id', 'game_date', 'week','div_game','posteam','defteam', 'home_team', 'away_team', 'weather', 'stadium',  'spread_line', 'total_line', 'roof', 'surface', 'temp', 'wind', 'home_coach', 'away_coach', 'receiver_player_id', 'receiver_player_name','season']).agg({
            'passing_yards': 'sum',
            'air_yards': 'sum',
            'pass_touchdown': 'sum', 
            'pass_attempt': 'sum',
            'reception': 'sum',
            'interception': 'sum', #the passing stats are duplicated for receivers
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',# Sum passing yards
            'rush_touchdown': 'sum',
            'lateral_rush': 'sum',
            'receiving_yards': 'sum',
            'yards_after_catch':'sum',
            'touchdown':'sum',
            'fumble': 'sum',
            'two_points': 'sum'
        }).reset_index()


        rushing_stats = receiver_rusher_stats.groupby(['game_id', 'game_date', 'week', 'div_game','posteam','defteam', 'home_team', 'away_team', 'weather', 'stadium',  'spread_line', 'total_line', 'roof', 'surface', 'temp', 'wind', 'home_coach', 'away_coach', 'rusher_player_id', 'rusher_player_name','season']).agg({
            'passing_yards': 'sum',
            'air_yards': 'sum',
            'pass_touchdown': 'sum', 
            'pass_attempt': 'sum',
            'reception': 'sum',
            'interception': 'sum',
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',# Sum passing yards
            'rush_touchdown': 'sum',
            'lateral_rush': 'sum',
            'receiving_yards': 'sum',
            'yards_after_catch': 'sum',
            'touchdown':'sum',
            'fumble': 'sum',
            'two_points': 'sum'
        }).reset_index()

        ## Grabbing seasonal info


        team['team'] = team['team'].replace({'OAK':'LV', 'STL':'LA', 'SD':'LAC','HST':'HOU', 'BLT':'BAL', 'CLV':'CLE','SL':'LA','ARZ':'ARI'})


        # team.rename(columns = {'player_id':'passer_player_id'},inplace = True)

        ## Standardizing Columns
        rushing_stats.rename(columns = {'rusher_player_id':'player_id'}, inplace = True)

        receiver_stats.rename(columns = {'receiver_player_id':'player_id'}, inplace = True)

        rushing_stats.rename(columns = {'rusher_player_name':'player_name'}, inplace = True)

        receiver_stats.rename(columns = {'receiver_player_name':'player_name'}, inplace = True)


        rusher_receiver_df = pd.concat([receiver_stats,rushing_stats])




        game_score_info = self.schedules_df[['season','home_score','away_score','game_id']].copy()


        rusher_receiver_df = rusher_receiver_df.merge(game_score_info, on = ['game_id','season'], how = 'inner')

        rusher_receiver_df = rusher_receiver_df.groupby(['game_id', 'game_date', 'week', 'div_game', 'posteam','defteam','home_team', 'away_team', 'weather', 'stadium',  'spread_line', 'total_line', 'roof', 'surface', 'temp', 'wind', 'home_coach', 'away_coach', 'player_id', 'player_name','season']).agg({
        'passing_yards': 'sum',
        'air_yards': 'sum',
        'pass_touchdown': 'sum', 
        'pass_attempt': 'sum',
        'reception': 'sum',
        'interception': 'sum', #the passing stats are duplicated for receivers
        'rush_attempt': 'sum',
        'rushing_yards': 'sum',# Sum passing yards
        'rush_touchdown': 'sum',
        'lateral_rush': 'sum',
        'receiving_yards': 'sum',
        'yards_after_catch':'sum',
        'touchdown':'sum',
        'fumble': 'sum',
        'two_points': 'sum'
        }).reset_index()


        rusher_receiver_df.rename(columns = {'defteam':'opponent_team'} , inplace = True )


        # #Checking the passing stats dataframe
        # rusher_receiver_df.head(2)


        rusher_receiver_df = rusher_receiver_df.drop_duplicates()

        rusher_receiver_df['fantasy_points'] = ((rusher_receiver_df['passing_yards']/25 )
                                                + (rusher_receiver_df['pass_touchdown'] * 4) + 
                                                (rusher_receiver_df['interception'] * -2) +
                                                (rusher_receiver_df['reception'] * 1) +
                                                (rusher_receiver_df['touchdown'] * 6) +
                                                (rusher_receiver_df['receiving_yards'] * .1) +
                                                (rusher_receiver_df['fumble'] * -2) +
                                                (rusher_receiver_df['two_points'] * 2))
        


        calculating_prior_points = rusher_receiver_df[['player_name','season','fantasy_points']].copy()

        calculating_prior_points['prior_ssn_avg_fp'] = calculating_prior_points.groupby(['player_name','season'])['fantasy_points'].transform('mean')

        calculating_prior_points = calculating_prior_points.drop_duplicates()


        calculating_prior_points.rename(columns = {'season':'actual_season'}, inplace = True)

        calculating_prior_points['season'] = calculating_prior_points['actual_season'] + 1

        rusher_receiver_df = rusher_receiver_df.merge(calculating_prior_points[['player_name','season','prior_ssn_avg_fp']], how = 'left', on = ['player_name','season'])

        rusher_receiver_df = rusher_receiver_df.drop_duplicates()


        
        df_rusher_receiver_game_level = rusher_receiver_df.groupby(['game_id', 'game_date', 'week', 'season', 'posteam', 'opponent_team', 'player_name', 'player_id']).agg({
            # Game level
            'home_team': 'first',
            'away_team': 'first',

            # Play level
            'fantasy_points': 'sum',
            'passing_yards': 'sum',
            'air_yards': 'sum',
            'pass_touchdown': 'sum', 
            'pass_attempt': 'sum',
            'reception': 'sum',
            'interception': 'sum',
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',# Sum passing yards
            'rush_touchdown': 'sum',
            'lateral_rush': 'sum',
            'receiving_yards': 'sum',
            'yards_after_catch': 'sum',
            'touchdown':'sum',
            'fumble': 'sum',
            'two_points': 'sum'

        })

        df_rusher_receiver_game_level["home"] = df_rusher_receiver_game_level["home_team"] == df_rusher_receiver_game_level.index.get_level_values("posteam")
        df_rusher_receiver_game_level.drop(columns=['home_team', 'away_team'], inplace=True)
        fields = ['fantasy_points','reception','rushing_yards','touchdown','receiving_yards','fumble','passing_yards','pass_touchdown','two_points']


        # Apply the function
        df_rusher_receiver_game_level = df_rusher_receiver_game_level.groupby(['player_name', 'player_id']).apply(self.calc_agg_stats, fields=fields)



        df_rusher_receiver_game_level = df_rusher_receiver_game_level.reset_index(0).reset_index(0).drop(columns = ['player_name','player_id']).reset_index()


        schedules_df_copy = self.schedules_df[self.schedules_df['game_id'].isin(self.schedules_df['game_id'].unique()) & (self.schedules_df['gameday'] >= '2001-09-09')]
        schedules_df_copy.rename(columns = {'gameday':'game_date'}, inplace = True)

        home_teams = schedules_df_copy[['game_id', 'game_date','season','home_team','away_score','week']].copy()

        away_teams = schedules_df_copy[['game_id', 'game_date','season','away_team','home_score','week']].copy()

        home_teams.rename(columns = {'home_team':'team','away_score':'points_allowed'}, inplace = True)
        away_teams.rename(columns = {'away_team':'team','home_score':'points_allowed'}, inplace = True)

        points_allowed_df = pd.concat([home_teams,away_teams])

        points_allowed_df = points_allowed_df.groupby(['game_id', 'game_date','season','week','team']).agg({'points_allowed':'sum'})

        group_sorted = points_allowed_df.sort_values('week')

        pa_df = group_sorted.groupby(['team']).apply(self.calc_agg_stats, fields=['points_allowed']).reset_index(0).drop(columns = 'team').reset_index()[['game_id','game_date','season','week','team','points_allowed_mean_season','points_allowed_mean_last5']]


        pa_df.rename(columns = {'team':'opponent_team'}, inplace = True)


        
        rusher_receiver_features = rusher_receiver_df.merge(df_rusher_receiver_game_level, how = 'inner' ,on = ['game_id','game_date','week','season','posteam','opponent_team','player_name','player_id'])
        # rusher_receiver_features['opponent_team'] = np.where(rusher_receiver_features['team'] == rusher_receiver_features['home_team'],rusher_receiver_features['away_team'],rusher_receiver_features['home_team'])
        rusher_receiver_features = rusher_receiver_features.merge(pa_df , how = 'inner',on = ['game_date','season','week','opponent_team','game_id'])


        rusher_receiver_features = rusher_receiver_features.fillna(0)


        self.rusher_receiver_features_df = rusher_receiver_features.copy()
                            
    

    def generate_qb_features(self):
        
        passing_stats = self.pbp_df[~self.pbp_df['passer_player_id'].isna()].copy()


        passing_stats['two_points'] = np.where(passing_stats['two_point_conv_result'] == 'success',1,0)


        passing_stats = passing_stats.groupby(['game_id', 'game_date','season', 'week', 'div_game', 'home_team', 'away_team','posteam','defteam', 'weather', 'location', 'stadium',  'spread_line', 'total_line', 'roof', 'surface', 'temp', 'wind', 'home_coach', 'away_coach', 'passer_player_id', 'passer_player_name']).agg({
                    'passing_yards': 'sum',
                    'air_yards': 'sum',
                    'pass_touchdown': 'sum', 
                    'pass_attempt': 'sum',
                    'complete_pass': 'sum',
                    'interception': 'sum', #the passing stats are duplicated for receivers
                    'rush_attempt': 'sum',
                    'rushing_yards': 'sum',# Sum passing yards
                    'rush_touchdown': 'sum',
                    'lateral_rush': 'sum',
                    'receiving_yards': 'sum',
                    'yards_after_catch':'sum',
                    'touchdown':'sum',
                    'fumble': 'sum',
                    'two_points': 'sum'
        }).reset_index()

        game_score_info = self.schedules_df[['season','home_score','away_score','game_id']].copy()

        passing_stats = passing_stats.merge(game_score_info, on = ['game_id','season'], how = 'inner')

        passing_stats.rename(columns = {'defteam':'opponent_team', 'passer_player_name':'player_name', 'passer_player_id':'player_id'} , inplace = True )

        ## Aggregate average score to opposition 

        # passing_stats['opponent_team'] = passing_stats.apply(get_opposing_team,axis = 1)
        passing_stats['fantasy_points'] = ((passing_stats['passing_yards']/25 )
                                                + (passing_stats['pass_touchdown'] * 4) + 
                                                (passing_stats['interception'] * -2) +
                                                (passing_stats['touchdown'] * 6) +
                                                (passing_stats['receiving_yards'] * .1) +
                                                (passing_stats['fumble'] * -2) +
                                                (passing_stats['two_points'] * 2))
        calculating_prior_points = passing_stats[['player_name','season','fantasy_points']].copy()

        calculating_prior_points['prior_ssn_avg_fp'] = calculating_prior_points.groupby(['player_name','season'])['fantasy_points'].transform('mean')

        calculating_prior_points = calculating_prior_points.drop_duplicates()


        calculating_prior_points.rename(columns = {'season':'actual_season'}, inplace = True)

        calculating_prior_points['season'] = calculating_prior_points['actual_season'] + 1

        passing_stats = passing_stats.merge(calculating_prior_points[['player_name','season','prior_ssn_avg_fp']], how = 'left', on = ['player_name','season'])

        passing_stats = passing_stats.drop_duplicates()


        passing_stats['home_flag'] = np.where(passing_stats['opponent_team'] == passing_stats['home_team'], 0,1)


        df_game_level = passing_stats.groupby(['game_id', 'game_date', 'week', 'season', 'posteam', 'opponent_team', 'player_name', 'player_id']).agg({
        # Game level
        'home_team': 'first',
        'away_team': 'first',
        # Play level
        'fantasy_points': 'sum',
        'passing_yards': 'sum',
        'air_yards': 'sum',
        'pass_touchdown': 'sum', 
        'pass_attempt': 'sum',
        'complete_pass': 'sum',
        'interception': 'sum',
        'rush_attempt': 'sum',
        'rushing_yards': 'sum',# Sum passing yards
        'rush_touchdown': 'sum',
        'lateral_rush': 'sum',
        'receiving_yards': 'sum',
        'yards_after_catch': 'sum',
        'touchdown':'sum',
        'fumble': 'sum',
        'two_points': 'sum'
    })
        
        fields = ['fantasy_points','pass_attempt','interception','rush_attempt','rush_touchdown','complete_pass','rushing_yards','touchdown','receiving_yards','fumble','passing_yards','pass_touchdown','two_points']


        # Apply the function
        df_game_level = df_game_level.groupby(['player_name', 'player_id']).apply(self.calc_agg_stats, fields=fields)

        df_game_level = df_game_level.reset_index(0).reset_index(0).drop(columns = ['player_name','player_id']).reset_index()


        schedules_df_copy = self.schedules_df[self.schedules_df['game_id'].isin(self.schedules_df['game_id'].unique()) & (self.schedules_df['gameday'] >= '2001-09-09')]
        schedules_df_copy.rename(columns = {'gameday':'game_date'}, inplace = True)

        home_teams = schedules_df_copy[['game_id', 'game_date','season','home_team','away_score','week']].copy()

        away_teams = schedules_df_copy[['game_id', 'game_date','season','away_team','home_score','week']].copy()

        home_teams.rename(columns = {'home_team':'team','away_score':'points_allowed'}, inplace = True)
        away_teams.rename(columns = {'away_team':'team','home_score':'points_allowed'}, inplace = True)

        points_allowed_df = pd.concat([home_teams,away_teams])

        points_allowed_df = points_allowed_df.groupby(['game_id', 'game_date','season','week','team']).agg({'points_allowed':'sum'})

        group_sorted = points_allowed_df.sort_values('week')

        pa_df = group_sorted.groupby(['team']).apply(self.calc_agg_stats, fields=['points_allowed']).reset_index(0).drop(columns = 'team').reset_index()[['game_id','game_date','season','week','team','points_allowed_mean_season','points_allowed_mean_last5']]


        pa_df.rename(columns = {'team':'opponent_team'}, inplace = True)


                    
        passing_stats = passing_stats.merge(df_game_level, how = 'inner' ,on = ['game_id','game_date','week','season','posteam','opponent_team','player_name','player_id'])
        # rusher_receiver_features['opponent_team'] = np.where(rusher_receiver_features['team'] == rusher_receiver_features['home_team'],rusher_receiver_features['away_team'],rusher_receiver_features['home_team'])
        passing_stats = passing_stats.merge(pa_df , how = 'inner',on = ['game_date','season','week','opponent_team','game_id'])


        passing_stats = passing_stats.fillna(0)


        self.qb_features_df = passing_stats.copy()




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
        df_kicker_pbp["fantasy_points"] = (
            df_kicker_pbp["50+_fg_made"] * 5 +
            df_kicker_pbp["40-49_fg_made"] * 4 +
            df_kicker_pbp["0-39_fg_made"] * 3 +
            df_kicker_pbp["xp_made"] * 1 +
            df_kicker_pbp["missed_fg_0-39"] * -2 +
            df_kicker_pbp["missed_fg_40-49"] * -1
        )

        # Optional: Drop any rows with NaN values in the calculated columns
        # df_kicker_pbp.dropna(subset=["fantasy_points"], inplace=True)

        # Log completion message
        print("Kicker play-by-play data processing completed successfully.")
        df_kicker_game_level_stadium = df_kicker_pbp.groupby(['game_id', 'game_date', 'week', 'season', 'stadium'], as_index=False).agg({
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
            'fantasy_points': 'sum',
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

        df_kicker_game_level["home"] = df_kicker_game_level["home_team"] == df_kicker_game_level["posteam"]
        df_kicker_game_level.drop(columns=['home_team', 'away_team'], inplace=True)
        # Define the fields for which you want to calculate aggregate statistics
        kicker_fields = [
            'fantasy_points', 
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
            ['kicker_player_name', 'kicker_player_id'], 
            group_keys=False
        ).apply(
            self.calc_agg_stats_kicker_d, 
            fields=kicker_fields
        ).reset_index(drop=True).round(2)
        df_kicker_game_level_agg = df_kicker_game_level_agg.drop(columns=df_kicker_game_level_agg.loc[:, "fantasy_points":"home"].columns)


        df_kicker_game_level_agg_by_game = df_kicker_game_level.groupby(['game_id', 'game_date', 'week', 'season', 'posteam', 'defteam'], as_index=False).agg({
            # Play level
            'fantasy_points': 'sum',
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
            ['defteam'], 
            group_keys=False
        ).apply(
            self.calc_agg_stats_kicker_d, 
            fields=kicker_fields, 
            career=False 
        ).reset_index(drop=True).round(2)
        df_kicker_game_level_agg_by_def = df_kicker_game_level_agg_by_def.drop(columns=df_kicker_game_level_agg_by_def.loc[:, "fantasy_points":"xp_made_33y"].columns)

        # Merge kicker aggregate stats with defensive team stats
        df_combined = pd.merge(
            df_kicker_game_level_agg,
            df_kicker_game_level_agg_by_def,
            on=['game_id', 'game_date', 'week', 'season', 'posteam', 'defteam'],
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
            df_kicker_game_level[['game_id', 'fantasy_points', 'kicker_player_id']],
            on=['game_id', 'kicker_player_id'],
            how='left'
        )

        # Drop redundant columns if necessary
        columns_to_drop = ['home_team']
        df_combined.drop(columns=columns_to_drop, inplace=True, errors='ignore')


        # Reset index
        df_combined.reset_index(drop=True, inplace=True)

        # Log completion message
        print("DataFrames merged successfully into 'df_combined'.")


        self.kicker_defense_features_df = df_combined.fillna(0)



    def preprocess_data(self,data,target_variable):
            """
            Preprocesses the data by splitting into training and testing sets,
            converting categorical variables to dummy variables, and scaling the features.
            """
            # Separate features and target
            X = data.drop(columns=[target_variable])
            y = data[target_variable]

            # Split data into training and testing sets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state= 42
            )

            # Align the training and testing data
            X_train, X_test = X_train_raw.align(X_test_raw, join='left', axis=1, fill_value=0)

            # Convert y_train and y_test to 1D arrays if necessary
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()

            # Standardize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)


            return X_train_scaled,X_test_scaled,y_train,y_test




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
    

    def train_qb_model(self):
        df = self.qb_features_df[self.qb_features + ['fantasy_points']].copy()
        df = self.get_dummy_variables(df)

        self.qb_x_train,self.qb_x_test,self.qb_y_train,self.qb_y_test = self.preprocess_data(df, 'fantasy_points')


        self.qb_model = LinearRegression()

        self.qb_model.fit(self.qb_x_train,self.qb_y_train)


    def train_rb_wr_model(self):

        df = self.rusher_receiver_features_df[self.rb_wr_features + ['fantasy_points']].copy()
        df = self.get_dummy_variables(df)

        self.rb_wr_x_train,self.rb_wr_x_test,self.rb_wr_y_train,self.rb_wr_y_test = self.preprocess_data(df, 'fantasy_points')


        self.rb_wr_model = LinearRegression()

        self.rb_wr_model.fit(self.rb_wr_x_train,self.rb_wr_y_train)


    def train_kicker_defense_model(self):

        df = self.kicker_defense_features_df[self.kicker_defense_features + ['fantasy_points']].copy()

        df = self.get_dummy_variables(df)

        self.kicker_defense_x_train,self.kicker_defense_x_test,self.kicker_defense_y_train,self.kicker_defense_y_test = self.preprocess_data(df, 'fantasy_points')


        self.kicker_defense_model = LinearRegression()

        self.kicker_defense_model.fit(self.kicker_defense_x_train,self.kicker_defense_y_train)





            


