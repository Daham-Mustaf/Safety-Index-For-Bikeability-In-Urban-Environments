import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the given dataframe by replacing lists with their first element, converting selected columns to numeric,
    and casting selected columns to specific data types.

    Parameters:
        df (pandas.DataFrame): the dataframe to be cleaned

    Returns:
        df (pandas.DataFrame): the cleaned dataframe
    """
    # rename coulumnes name
    """
    Renaming the columns using underscores makes it easier to access and manipulate the column names in the code.
    For example, the original column name "cycleway:left" would need to be accessed using quotes and brackets, 
    like so: row['cycleway:left']. But after renaming the column to "cycleway_left", it can be accessed using dot notation,
    like so: row.cycleway_left. 
    """
    # col_name = list(df.columns)
    # cleaned_cols = [col_name.replace(':', '_') if ':' in col else col for col in col_name]


    # List of columns to process
    cols_list = ['lanes', 'maxspeed', 'highway', 'surface', 'parking:lane:both', 'cycleway', 'bicycle',
                 'cycleway:both', 'cycleway:left', 'cycleway:right', 'oneway:bicycle', 'cycleway:right:lane',
                 'cycleway:left:lane', 'width', 'access', 'lit']
    cols_numeric = ['maxspeed', 'lanes']
    cols_dict = {'maxspeed': int, 'lanes': int}
    process_value = lambda x: x[0] if isinstance(x, list) else 0 if x is None else x
    
    try:
        # Replace each value in the columns with the first value in the list, if it is a list
        for col in cols_list:
            if col in df.columns:
                # df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0 if x is None or (isinstance(x, list) and len(x) == 0) else x)
                df[col] = df[col].apply(process_value)
        
        # Convert selected columns to numeric
        df[cols_numeric] = df[cols_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Cast selected columns to specific data types
        df = df.astype(cols_dict)

        print("\nData cleaning completed successfully.")
        print("*****************\n")
    except Exception as e:
        print("An error occurred during data cleaning: ", e)
    
    return df
   

def create_score_columns(df):
    """
    Creates columns for lane_score, maxspeed_score, parking, parking_score, facility_type,
    separated_bike_lane, shared_bike_lane, no_access_bike, cycle_track, dirt_cycle_track,
    no_bike_infra, has_cycle, and score in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to add the columns to.

    Returns:
        pandas.DataFrame: The DataFrame with the added score columns.
    """
    df['lane_score'] = np.nan
    df['maxspeed_score'] = np.nan
    df['parking'] = np.nan
    df['parking_score'] = np.nan
    df['facility_type'] = ''
    df['separated_bike_lane'] = np.nan
    df['shared_bike_lane'] = np.nan
    df['no_access_bike'] = np.nan
    df['cycle_track'] = np.nan
    df['dirt_cycle_track'] = np.nan
    df['no_bike_infra'] = np.nan
    df['has_cycle'] = np.nan
    df['score'] = np.nan

    return df

        
def compute_default_and_lane_score(df):
    """
        Computes and assigns default lane scores to a Pandas DataFrame based on the highway type, and assigns a lane score to each road segment.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to compute the lane scores for.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added 'lanes' and 'lane_score' columns.

    """
    lanes_by_highway_type = {
        'unclassified': 2,
        'secondary_link': 1,
        'service': 1,
        'living_street': 1,
        'track': 1,
        'footway': 1,
        'path': 1,
        'primary': 2,
        'primary_link': 2,
        'residential': 2,
        'secondary': 2,
        'tertiary': 2,
        'tertiary_link': 2
    }

    df.loc[df['lanes'] == 0, 'lanes'] = df.apply(
        lambda row: lanes_by_highway_type.get(row['highway'], 0) if row['highway'] in lanes_by_highway_type else 0,
        axis=1
    )

    df['lane_score'] = df['lanes'].apply(lambda x: min(x, 4))
    print("\nDefault and lane scores computation completed successfully.")
    print("*****************\n")
    return df

def set_def_maxspeed_and_speed_score(df):
    """
    This function takes a pandas DataFrame object as an input and modifies it in place. Specifically, it sets a default value for the 'maxspeed' column 
    based on the 'highway' column, and calculates a 'maxspeed_score' column based on the 'maxspeed' values.

    Parameters
    ---------- 
    df: a pandas DataFrame object
    Return:
    ----------
     None (the function modifies the input DataFrame in place)
    """

    # Define a dictionary to map highway types to default speed limits
    speed_limits = {
        'residential': 30,
        'service': 30,
        'primary': 50,
        'primary_link': 50,
        'secondary_link': 50,
        'secondary': 50,
        'tertiary': 50,
        'tertiary_link': 50,
        'unclassified': 30,
        'cycleway': 20,
        'footway': 20,
        'path': 20,
        'living_street': 20,
        'track': 20,
        'pedestrian': 20,
        'disused': 100,
        'razed': 150,
        'road': 50,
        'bridleway': 45
    }

    # Apply default speed limits to any rows where maxspeed is 0
    df.loc[df['maxspeed'] == 0, 'maxspeed'] = df.apply(
        lambda row: speed_limits.get(row['highway'], 0) if row['highway'] in speed_limits else 0,
        axis=1
    )

    # Calculate a maxspeed score based on the maxspeed values
    df['maxspeed_score'] = df['maxspeed'].apply(
    lambda x: 0 if x == 0
    else 2 if (0 < x <= 20)
    else 3 if (20 < x <= 30)
    else 4 if (30 < x <= 40)
    else 5 if (40 < x <= 50)
    else 6 if (50 < x <= 60)
    else 7 if (60 < x <= 70)
    else 8 if (70 < x <= 80)
    else 9 if (80 < x <= 90)
    else 10 if x > 90
    else x)

    print("\nMaxspeed score computed successfully!")
    print("*****************\n")
    return df


def compute_parking_and_parking_score(df):
    """
    This function computes a 'parking' variable based on the presence of specific parking-related tags in the input DataFrame.
    It also calculates a 'parking_score' column based on the 'parking' values.

    Parameters
    ----------
    df: a pandas DataFrame object

    Returns
    -------
    None (the function modifies the input DataFrame in place)
    """
    parking_tags = ['parking:lane:both', 'parking:lane:left', 'parking:lane:left:parallel', 'parking:lane:right',
                    'parking:lane:both:parallel']

    # Define a dictionary to map parking tag values to parking scores
    parking_scores = {
        'marked': 1,
        'separate': 1,
        'parallel': 1,
        'on_street': 1,
        'diagonal': 1,
        'half_on_kerb': 1,
        'on_kerb': 1,
        'street_side': 1,
        'perpendicular': 1,
        'painted_area_only': 1
    }

    # Compute the parking variable
    df['parking'] = 0
    for tag in parking_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(parking_scores.keys()), 'parking'] = 1

    # Compute the parking score
    # Define parking_scores dictionary
    parking_scores = {
    0: 0,   # No parking
    1: 1,   # Some parking
    }
    df['parking_score'] = df['parking'].map(parking_scores)

    if 'parking' not in df.columns or 'parking_score' not in df.columns:
        print('Error: the computation of parking and/or parking_score failed.')
    else:
        print('\nThe parking and parking_score computed successfully!')
        print("*****************\n")
    return df



def set_separated_bike_lane(df):
    """
    Set the separated bike lane variable based on the presence and value of cycleway:both tags.

    Args:
        df (pandas.DataFrame): The dataframe to modify.

    Returns:
        pandas.DataFrame: The modified dataframe with the 'separated_bike_la' and 'facility_type' columns added.

    """
    # List of cycleway tags to check for
    separated_tags = ['cycleway:both', 'cycleway:left', 'cycleway:right', 'cycleway']

    # Define a dictionary to map cycleway tag values to separated bike lane scores
    separated_score = {
        'separate': 1
    }

    # Compute the separated bike lane variable
    df['facility_type'] = ''
    df['separated_bike_la'] = 0
    for tag in separated_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(separated_score.keys()), 'separated_bike_la'] = 1

    # Fill any NaN values with 0
    df['separated_bike_la'] = df['separated_bike_la'].fillna(0)

    # Set facility_type to 'separated_bike_la' for roads with separated bike lanes
    df.loc[df['separated_bike_la'] == 1, 'facility_type'] = 'separated_bike_la'
    print('\nSetting of separated_bike_la and facility_type computed successfully!')
    print("*****************\n")
    return df



def compute_shared_bike_lane(df):
    """
    Computes the shared bike lane variable based on the values of the cycleway tags in the input dataframe.

    If the 'facility_type' column has any value other than 'shared_bike_lane', this function will compute the shared bike lane
    variable and update the 'facility_type' column if necessary.

    Args:
        df (pandas.DataFrame): The input dataframe containing the cycleway tags and facility_type column.

    Returns:
        pandas.DataFrame: The input dataframe with the shared_bike_lane variable computed and the 'facility_type' column
        updated if necessary.
    """

    # Define lists of cycleway tags to check and their corresponding scores
    shared_tags = ['cycleway:right', 'cycleway:left', 'cycleway:both', 'cycleway:both:lane']
    way_tag = ['cycleway', 'ped']

    way_shared_bike = {'track': 1}
    shared_scores = {
        'share_busway': 1,
        'track': 1, 
        'shared_lane': 1, 
        'lane': 1, 
        'advisory': 1, 
        'exclusive': 1, 
        # 'no': 1, 
        'opposite_lane': 1,
        'opposite_share_busway': 1,
        'opposite_track': 1,
        'pictogram': 1}

    # Check if 'facility_type' column has any rows with 'separated_bike_la' value
    computed_rows = df['facility_type'].isin(['separated_bike_la', 'a'])

    # Compute the shared bike lane variable for rows that are not already separated bike lane
    df.loc[~computed_rows, 'shared_bike_lane'] = 0
    for tag in shared_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(shared_scores.keys()) & ~computed_rows, 'shared_bike_lane'] = 1
    for tag in way_tag:
        if tag in df.columns:
            df.loc[df[tag].isin(way_shared_bike.keys()) & ~computed_rows, 'shared_bike_lane'] = 1

    # Update facility_type column if necessary
    df.loc[df['shared_bike_lane'] == 1 & ~computed_rows, 'facility_type'] = 'shared_bike_lane'
    # Fill any NaN values with 0
    df['shared_bike_lane'] = df['shared_bike_lane'].fillna(0)
    print('\nSetting of shared_bike_la was successfully')
    print("*****************\n")
        
    return df

def set_cycle_track(df):
    """
    Set the 'facility_type' variable to 'cycle_track' for roads with 'highway' = 'cycleway' and 'bicycle' = ['designated', 'yes'] and 'facility_type' is null. 
    Set the 'facility_type' variable to 'no_bike_infra' for roads with 'highway' = 'cycleway' and 'bicycle' = ['designated', 'yes'] and 'facility_type' is null. 

    Args:
        df (pandas.DataFrame): The dataframe to modify.

    Returns:
        pandas.DataFrame: The modified dataframe with the 'facility_type' column added.
    """
    #  solve conflicted tages 
    # Check if 'facility_type' column has any rows with 'separated_bike_la', 'no_access_bike' or 'shared_bike_lane' value
    computed_rows = df['facility_type'].isin(['separated_bike_la', 'shared_bike_lane'])
    # Compute the cycle_track variable for rows that are not already computed
    df.loc[~computed_rows, 'cycle_track'] = 0

   
    # List of tags to check for no-access bike parameters
    cycle_track_tags = ['highway', 'bicycle', 'cycleway']
    # Set facility_type to 'cycle_track' for roads with 'highway' = 'cycleway' and 'bicycle' = ['designated', 'yes'] and 'facility_type' is null

    cycle_track_param = {
        'cycleway': 1,
        'official': 1,
        'designated': 1,
        'destination': 1,
        'permissive': 1,
        'undefined': 1,
        'use_sidepath': 1,
        'yes': 1,
        'crossing': 1,
        'link': 1,
        'opposite': 1,
        'opposite_lane': 1}
    for tag in cycle_track_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(cycle_track_param.keys()) & ~computed_rows, 'cycle_track'] = 1
    
    # Update facility_type column if necessary
    df.loc[df['cycle_track'] == 1 & ~computed_rows, 'facility_type'] = 'cycle_track'
    # Fill any NaN values with 0
    df['cycle_track'] = df['cycle_track'].fillna(0)
    print('\nSetting of set_cycle_track was successfully!')
    print("*****************\n")
    return df

def set_dir_surface(df):

    dirt_tags = ['surface', 'A', 'B']
    dirt_param = {
        'sett': 1,
        'dirt/sand': 1,
        'earth': 1,
        'grass': 1,
        'grass': 1,
        'ground': 1,
        'mud': 1,
        'overgrown': 1,
        'sand': 1,
        'soil': 1,
        'wood': 1
    }
    computed_rows = df['facility_type'].isin(['separated_bike_la', 'shared_bike_lane', 'cycle_track'])
    for tag in dirt_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(dirt_param.keys()) & ~computed_rows, 'dirt_cycle_track'] = 1
    
    # Update facility_type column if necessary
    df.loc[df['dirt_cycle_track'] == 1 & ~computed_rows, 'facility_type'] = 'dirt_cycle_track'
    # Fill any NaN values with 0
    df['dirt_cycle_track'] = df['dirt_cycle_track'].fillna(0)
    print('\nSetting of dirt_cycle_track was successfully!')
    print("*****************\n")



def set_bik_no_infra(df):
    """
    Set the 'facility_type' variable to 'no_bike_infra' for roads with no dedicated bike infrastructure.

    Args:
        df (pandas.DataFrame): The dataframe to modify.

    Returns:
        pandas.DataFrame: The modified dataframe with the 'facility_type' column added.
    """
    
    # Check if 'facility_type' column has any rows with 'separated_bike_la' or 'shared_bike_lane' value
    computed_rows = df['facility_type'].isin(['separated_bike_la', 'shared_bike_lane','dirt_cycle_track', 'cycle_track'])
    # Compute the shared bike lane variable for rows that are not already separated bike lane
    df.loc[~computed_rows, 'no_bike_infra'] = 0

   
    # List of tags to check for no-access bike parameters
    no_bike_infra_tags = ['highway', 'surface']

    # Define a dictionary to map tag values to no-access bike scores
    no_bike_infra_param = {
        'asphalt':1, 
        'secondary': 1,
        'secondary_link': 1,
        'living_street': 1,
        'unclassified': 1,
        'primary': 1,
        'primary_link': 1,
        'tertiary': 1,
        'track': 1,
        'trunk': 1,
        'bridleway': 1, 
        'residential': 1,
        'tertiary_link': 1,
        'unclassified': 1,
        'trunk_link':1
    }
    for tag in no_bike_infra_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(no_bike_infra_param.keys()) & ~computed_rows, 'no_bike_infra'] = 1
    
    # Update facility_type column if necessary
    df.loc[df['no_bike_infra'] == 1 & ~computed_rows, 'facility_type'] = 'no_bike_infra'
    # Fill any NaN values with 0
    df['no_bike_infra'] = df['no_bike_infra'].fillna(0) 
    print('Setting of no_bike_infra parameter was successful.')
    return df


def set_no_access(df):
    """
    Set the 'no_access_bike' variable based on the presence and value of 'cycleway', 'access', and 'bicycle' tags.

    Args:
        df (pandas.DataFrame): The dataframe to modify.

    Returns:
        pandas.DataFrame: The modified dataframe with the 'no_access_bike' column added.

    """
 
    #  solve conflicted tages 
    # Check if 'facility_type' column has any rows with 'separated_bike_la' or 'shared_bike_lane' value
    computed_rows = df['facility_type'].isin(['separated_bike_la', 'shared_bike_lane', 'cycle_track', 'dirt_cycle_track', 'no_bike_infra'])
    # Compute the shared bike lane variable for rows that are not already separated bike lane
    df.loc[~computed_rows, 'no_access_bike'] = 0

   
    # List of tags to check for no-access bike parameters
    no_access_tags = ['access', 'highway']

    # Define a dictionary to map tag values to no-access bike scores
    no_access_param = {
        'no': 1,
        'military': 1,
        'none': 1,
        'disused': 1,
        'razed': 1
    }

    for tag in no_access_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(no_access_param.keys()) & ~computed_rows, 'no_access_bike'] = 1
    
    # Update facility_type column if necessary
    df.loc[df['no_access_bike'] == 1 & ~computed_rows, 'facility_type'] = 'no_access_bike'
    # Fill any NaN values with 0
    df['no_access_bike'] = df['no_access_bike'].fillna(0) 
    print('\nSetting of no-access bike parameter was successfully!')
    print("*****************\n")
    return df


def set_has_cycle(df):
  
    #  solve conflicted tages 
    # Check if 'facility_type' column has any rows with 'separated_bike_la', 'no_access_bike' or 'shared_bike_lane' value
    computed_rows = df['facility_type'].isin(['no_bike_infra', 'shared_bike_lane', 'cycle_track',
       'no_access_bike', 'separated_bike_la', 'dirt_cycle_track'])
    # Compute the cycle_track variable for rows that are not already computed
    df.loc[~computed_rows, 'path_cycle'] = 0

   
    # List of tags to check for no-access bike parameters
    cycle_track_tags = ['highway', 'a', 'b']
    # Set facility_type to 'cycle_track' for roads with 'highway' = 'cycleway' and 'bicycle' = ['designated', 'yes'] and 'facility_type' is null

    has_cycle_param = {
        'path': 1,
        'pedestrian': 1,
        'road': 1,
        'service': 1,
        'non': 1
        }
    for tag in cycle_track_tags:
        if tag in df.columns:
            df.loc[df[tag].isin(has_cycle_param.keys()) & ~computed_rows, 'path_cycle'] = 1
    
    # Update facility_type column if necessary
    df.loc[df['path_cycle'] == 1 & ~computed_rows, 'facility_type'] = 'path_cycle'
    # Fill any NaN values with 0
    df['path_cycle'] = df['path_cycle'].fillna(0)
    print('\nSetting of path_cycle was successfully!')
    print("*****************\n")
    return df


def score_computation(df):
    # ['separated_bike_la', 'shared_bike_lane','dirt_cycle_track', 'cycle_track']
    df['score'] = 0
    A = .7
    B = 1.03662691652468
    C = 1.30834752981261
    D = 1.52123793299262
    # for cycleway=no #  no infrastructure is given 
    E = 1.72123793299262
    F = 1.82123793299262
    s = 0.371379897785347
    l = 0.597955706984669
    p = 0.5809199318569

    df['score'] = np.where(df['facility_type'] == 'separated_bike_la',
    # A + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           A + df['lane_score'] * l + df['parking_score'] * p,

                           df['score'])
    df['score'] = np.where(df['facility_type'] == 'shared_bike_lane',
                           B + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           df['score'])
    df['score'] = np.where(df['facility_type'] == 'cycle_track',
                           C + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           df['score'])
    # # # no_bike_infra
    df['score'] = np.where(df['facility_type'] == 'no_bike_infra',
                           D + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           df['score'])
    df['score'] = np.where(df['facility_type'] == 'path_cycle',
                           E + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           df['score'])
    df['score'] = np.where(df['facility_type'] == 'dirt_cycle_track',
                           F + s * df['maxspeed_score'] + df['lane_score'] * l + df['parking_score'] * p,
                           df['score'])

    df['score'] = np.where(df['facility_type'] == 'no_access_bike',
                           5,
                           df['score'])
    print('\nscore computation has benn successful done!')
    print("*****************\n")
    return df


def set_scores(df):
    """
    Performs all score-related computations and modifications on the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
        pandas.DataFrame: The processed DataFrame with scores and modifications applied.
    """
    # Clean the data
    df = clean_data(df)
    
    df = create_score_columns(df)

    # Compute default and lane scores
    df = compute_default_and_lane_score(df)

    # Set default max speed and max speed scores
    df = set_def_maxspeed_and_speed_score(df)

    # Compute parking and parking scores
    df = compute_parking_and_parking_score(df)

    # Set separated bike lane
    df = set_separated_bike_lane(df)

    # Compute shared bike lane
    df = compute_shared_bike_lane(df)

    # Set cycle track
    df = set_cycle_track(df)

    # Set dirt surface
    df = set_dir_surface(df)

    # Set no bike infrastructure
    df = set_bik_no_infra(df)

    # Set has cycle
    # df = set_has_cycle(df)

    # Compute score computation
    # df = score_computation(df)

    # return df