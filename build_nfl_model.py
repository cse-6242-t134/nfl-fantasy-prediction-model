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


rb_wr_features = ['prior_ssn_avg_fp', 'n_games_career', 'n_games_season', 'fantasy_points_mean_career', 'fantasy_points_mean_season', 'fantasy_points_total_season', 'fantasy_points_mean_last5', 'fantasy_points_total_last5', 'reception_mean_career', 'reception_mean_season', 'reception_total_season', 'reception_mean_last5', 'reception_total_last5', 'reception_last', 'rushing_yards_mean_season', 'rushing_yards_last', 'touchdown_total_season', 'touchdown_total_last5', 'receiving_yards_mean_career', 'receiving_yards_total_career', 'receiving_yards_mean_season', 'receiving_yards_total_season', 'receiving_yards_mean_last5', 'receiving_yards_total_last5', 'fumble_mean_career', 'passing_yards_mean_career', 'passing_yards_total_career', 'passing_yards_mean_season', 'passing_yards_total_season', 'passing_yards_mean_last5', 'passing_yards_total_last5', 'pass_touchdown_mean_career', 'pass_touchdown_total_career', 'two_points_total_career', 'points_allowed_mean_season', 'points_allowed_mean_last5', 'home_team_BUF', 'home_team_GB', 'home_team_TEN', 'away_team_CHI', 'away_team_DET', 'away_team_GB', 'away_team_IND', 'away_team_NE', 'away_team_NYJ', 'away_team_PIT', 'away_team_SF', 'away_team_WAS', 'opponent_team_BAL', 'opponent_team_BUF', 'opponent_team_CAR', 'opponent_team_CIN', 'opponent_team_DEN', 'opponent_team_DET', 'opponent_team_KC', 'opponent_team_LV', 'opponent_team_PIT', 'opponent_team_TEN']



def calc_agg_stats(group, fields, career=True):
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




def load_data():
    '''
    Method returns four dataframes that will be used to generate features.

    Paramaters: None

    Returns = roster_data, pbp_df, weekly_df,schedules_df
    
    '''
    roster_data = nfl.import_seasonal_rosters([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999])
    pbp_df = pd.DataFrame(nfl.import_pbp_data([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]))
    weekly_df = pd.DataFrame(nfl.import_weekly_data([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]))
    schedules_df = pd.DataFrame(nfl.import_schedules([2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]))

    return roster_data,pbp_df,weekly_df,schedules_df

def generate_features(roster_data,pbp_df,weekly_df,schedules_df, position = 'rb_wr'):
    '''
    Method that returns the dataframe of a certain position group with aggregated features. 

    Parameters: 
    roster_data,pbp_df,weekly_df,schedules_df = Dataframes returned from load_data()

    postition = str of position group to generate features for.


    Returns:
    df_combined = Dataframe with calculated features for a given position group.
    
    '''

    if position in ['kicker','rb_wr']:
        if position == 'rb_wr':
            team = roster_data[['season','player_id','team','depth_chart_position']]

            receiver_rusher_stats =  pbp_df[(pbp_df['receiver_player_id'].notnull()) | (pbp_df['rusher_player_id'].notnull())]
                                    

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

            game_score_info = schedules_df[['season','home_score','away_score','game_id']].copy()


            rusher_receiver_df = rusher_receiver_df.merge(game_score_info, on = ['game_id','season'], how = 'inner')



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
            df_rusher_receiver_game_level = df_rusher_receiver_game_level.groupby(['player_name', 'player_id']).apply(calc_agg_stats, fields=fields)



            df_rusher_receiver_game_level = df_rusher_receiver_game_level.reset_index(0).reset_index(0).drop(columns = ['player_name','player_id']).reset_index()


            schedules_df_copy = schedules_df[schedules_df['game_id'].isin(schedules_df['game_id'].unique()) & (schedules_df['gameday'] >= '2001-09-09')]
            schedules_df_copy.rename(columns = {'gameday':'game_date'}, inplace = True)

            home_teams = schedules_df_copy[['game_id', 'game_date','season','home_team','away_score','week']].copy()

            away_teams = schedules_df_copy[['game_id', 'game_date','season','away_team','home_score','week']].copy()

            home_teams.rename(columns = {'home_team':'team','away_score':'points_allowed'}, inplace = True)
            away_teams.rename(columns = {'away_team':'team','home_score':'points_allowed'}, inplace = True)

            points_allowed_df = pd.concat([home_teams,away_teams])

            points_allowed_df = points_allowed_df.groupby(['game_id', 'game_date','season','week','team']).agg({'points_allowed':'sum'})

            group_sorted = points_allowed_df.sort_values('week')

            pa_df = group_sorted.groupby(['team']).apply(calc_agg_stats, fields=['points_allowed']).reset_index(0).drop(columns = 'team').reset_index()[['game_id','game_date','season','week','team','points_allowed_mean_season','points_allowed_mean_last5']]


            pa_df.rename(columns = {'team':'opponent_team'}, inplace = True)


            
            rusher_receiver_features = rusher_receiver_df.merge(df_rusher_receiver_game_level, how = 'inner' ,on = ['game_id','game_date','week','season','posteam','opponent_team','player_name','player_id'])
            # rusher_receiver_features['opponent_team'] = np.where(rusher_receiver_features['team'] == rusher_receiver_features['home_team'],rusher_receiver_features['away_team'],rusher_receiver_features['home_team'])
            rusher_receiver_features = rusher_receiver_features.merge(pa_df , how = 'inner',on = ['game_date','season','week','opponent_team','game_id'])


            rusher_receiver_features = rusher_receiver_features.fillna(0)


            df_combined = rusher_receiver_features.copy()
                        
        elif position == 'kicker':

            # Filter rows where 'kicker_player_name' is not null and the play type is relevant
            df_kicker_pbp = pbp_df.loc[
                pbp_df['kicker_player_name'].notnull() & 
                pbp_df['play_type'].isin(['field_goal', 'extra_point', 'kickoff'])
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
                calc_agg_stats, 
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


            df_kicker_game_level_agg_by_def = df_kicker_game_level_agg_by_game.groupby(
                ['defteam'], 
                group_keys=False
            ).apply(
                calc_agg_stats, 
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

            # Calculate the percentage of null values in each column
            null_percentages = df_combined.isnull().mean() * 100

            # Sort the percentages in descending order for better readability
            null_percentages = null_percentages.sort_values(ascending=False)

            # Format the output to display percentages with two decimal places
            null_percentages_formatted = null_percentages.apply(lambda x: f"{x:.2f}%")

            # Print the results
            print("Percentage of Null Values in Each Column:")
            print(null_percentages_formatted)


            # Ensure 'temp' and 'wind' are numeric
            df_combined['temp'] = pd.to_numeric(df_combined['temp'], errors='coerce')
            df_combined['wind'] = pd.to_numeric(df_combined['wind'], errors='coerce')

            # Calculate mean 'temp' and 'wind' by stadium
            temp_wind_means = (
                df_combined.groupby('stadium')[['temp', 'wind']]
                .mean()
                .reset_index()
            )

            # Merge the mean values back to the original DataFrame
            df_combined = pd.merge(
                df_combined,
                temp_wind_means,
                on='stadium',
                how='left',
                suffixes=('', '_mean')
            )

            # Impute missing 'temp' and 'wind' with the group mean values
            df_combined['temp'].fillna(df_combined['temp_mean'], inplace=True)
            df_combined['wind'].fillna(df_combined['wind_mean'], inplace=True)

            # If any missing 'temp' or 'wind' values remain, fill them with the overall mean
            df_combined['temp'].fillna(df_combined['temp'].mean(), inplace=True)
            df_combined['wind'].fillna(df_combined['wind'].mean(), inplace=True)

            # Drop the temporary mean columns
            df_combined.drop(columns=['temp_mean', 'wind_mean'], inplace=True)

            # For the rest of the columns, fill missing values with 0
            # Exclude 'temp' and 'wind' as they've already been imputed
            columns_to_fill = df_combined.columns.difference(['temp', 'wind'])
            df_combined[columns_to_fill] = df_combined[columns_to_fill].fillna(0)

            # Check if any missing values remain
            remaining_nulls = df_combined.isnull().sum()
            if remaining_nulls.sum() > 0:
                print("Remaining null values after imputation:")
                print(remaining_nulls[remaining_nulls > 0])
            else:
                print("All missing values have been imputed.")

        else:
            print("Provide a relevant position group")

    return df_combined




def preprocess_data(data,target_variable):
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




def get_dummy_variables(df, drop_first=True, dummy_na=False):
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


def train_model(df_combined, position = 'rb_wr'):
    '''
    Method that trains and returns the model object for a given position.

    Parameters: 
    df_combined: dataframe object returned from generate_features()
    position: str of position to build the model for

    Returns:

    model - model object of a given position

    df - final model development dataset 
    x_train,x_test,y_train,y_test - dataset split up after train/test split is applied
    
    '''

    if position in ['kicker','rb_wr']:
        if position == 'rb_wr':
            df = df_combined[rb_wr_features + ['fantasy_points']].copy()
            df = get_dummy_variables(df)

            x_train,x_test,y_train,y_test = preprocess_data(df, 'fantasy_points')


            model = LinearRegression()

            model.fit(x_train,y_train)


    else:
        print("Provide relevant features")

        model = None

    return model, df,x_train,x_test,y_train,y_test



def run_entire_model_process(position = "rb_wr"):
    '''
    Method runs entire process from data aggregation to model training. 
    
    Parameters:
    position: str of position to build model for
    
    Returns:
    
    model - model object from position of interest
     
    df - final model development dataset used for building the model
    
    x_train,x_test,y_train,y_test - dataset split up after train/test split is applied

    '''
    if position == "rb_wr":
        roster_data,pbp_df,weekly_df,schedules_df = load_data()
        df_combined = generate_features()


        model,df = train_model(df_combined)

    else:
        print("provide relevant position")

    return model,df,x_train,x_test,y_train,y_test





          


