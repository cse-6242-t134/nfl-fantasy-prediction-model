import pandas as pd
import nfl_data_py as nfl

def load_data(start_year=1999, end_year=2024):
    """
    Loads the required NFL data and handles 'object' type columns by converting them to appropriate types.

    Parameters:
    - start_year: int, the starting year of the data to load.
    - end_year: int, the ending year of the data to load.

    Returns:
    - roster_data: DataFrame with player roster information.
    - pbp_df: DataFrame with play-by-play data.
    - schedules_df: DataFrame with game schedules.
    """
    print("Fetching data from source...")
    roster_data = nfl.import_seasonal_rosters(list(range(start_year, end_year + 1)))
    pbp_df = pd.DataFrame(nfl.import_pbp_data(list(range(start_year, end_year + 1))))
    schedules_df = pd.DataFrame(nfl.import_schedules(list(range(start_year, end_year + 1))))
    weekly_df = pd.DataFrame(nfl.import_weekly_data(list(range(start_year, end_year + 1))))

    return roster_data, pbp_df, schedules_df, weekly_df
