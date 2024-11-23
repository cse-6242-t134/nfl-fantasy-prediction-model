import build_nfl_model_prod as nfl_mod
from utils.nfl_data_loader import load_data
import pandas as pd



## Load in necessary data for the model development dataset 

roster_data, pbp_df, schedules_df,weekly_df = load_data(start_year=1999, end_year=2024)



## Training Model and Generating Predictions for Kickers 

kicker_obj = nfl_mod.NFLModel(position='Kicker', roster_data=roster_data, pbp_df=pbp_df, schedules_df=schedules_df, weekly_df = weekly_df)

# Preprocess data
kicker_obj.preprocess_data()

# Train and evaluate models
kicker_obj.train_evaluate_model(model_type='LinearRegression')
kicker_obj.train_evaluate_model(model_type='RandomForest')
kicker_obj.build_and_train_lstm()

predictions_kicker = kicker_obj.process_predictions(long_term = True)


predictions_kicker.drop(columns = ['home_team'], inplace = True)
predictions_kicker.rename(columns = {'home_team_k':'home_team', 'away_team_k':'away_team'} , inplace = True)

predictions_kicker['position'] = 'K'

## Training Model and Generating Predictions for Kickers 


rw_obj = nfl_mod.NFLModel(position='RW', roster_data=roster_data, pbp_df=pbp_df, schedules_df=schedules_df, weekly_df = weekly_df)

# Preprocess data
rw_obj.preprocess_data()

# Train and evaluate models
rw_obj.train_evaluate_model(model_type='LinearRegression')

x_scaled = rw_obj.scaler.transform(rw_obj.x)


predictions_rw = rw_obj.features_df.copy()


predictions_rw['predicted_fantasy'] = rw_obj.lr_model.predict(x_scaled)

predictions_rw.drop(columns = ['player_name'], inplace = True)

predictions_rw = predictions_rw.merge(roster_data[['player_id','player_name']], how = 'inner',on = ['player_id'])




## Training Model and Generating PRedictions for QB

qb_obj = nfl_mod.NFLModel(position='QB', roster_data=roster_data, pbp_df=pbp_df, schedules_df=schedules_df, weekly_df = weekly_df)

# Preprocess data
qb_obj.preprocess_data()

# Train and evaluate models
qb_obj.train_evaluate_model(model_type='LinearRegression')
qb_obj.train_evaluate_model(model_type='RandomForest')
qb_obj.build_and_train_lstm()


predictions_qb = qb_obj.process_predictions(ensemble=True)



fantasy_prediction_data = pd.concat([predictions_kicker,predictions_rw, predictions_qb])


fantasy_prediction_data = fantasy_prediction_data.drop_duplicates()


fantasy_prediction_data.to_csv('./fantasy_prediction_data.csv')