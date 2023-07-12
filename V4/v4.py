import networkx as nx
import osmnx as ox
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import missingno as no
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import matplotlib.pyplot as plt

useful_tags_way = ['cycleway:left:buffer', 'oneway', 'lanes', 'source:maxspeed','surface', "amenity",
                   'highway', 'maxspeed','parking:lane:both:parallel','class:bicycle',
                   'parking:lane:both', 'segregated','access','lit', 'sidewalk',
                   'parking:lane:right:parallel', 'parking:lane:left:parallel',
                   'cycleway:left', 'cycleway', 'bicycle', 'cycleway:left:buffer',
                   'cycleway:right:buffer', "cycleway:right", "cycleway:left", "cycleway:both",
                   "width", 'parking:lane:left', 'parking:lane:right', 'oneway:bicycle',
                   'oneway:bus', 'cycleway:right:lane', 'cycleway:both:lane', 'sidewalk:right:bicycle',
                   "lanes:width", 'cycleway:left:lane', 'cycleway:both:bicycle',
                   "cycleway:right", 'unclassified', 'from','to']

def get_place_network(place, simplify=False, useful_tags_way=useful_tags_way):
    """
    Returns a networkx graph for a given place, with only the useful tags specified in useful_tags_way.

    Parameters
    ----------
    place : str
        A query string for the place name or geography (e.g., "New York City, New York, USA").
    simplify : bool, optional
        If True, simplify the graph's topology. Default is False.
    useful_tags_way : list of str, optional
        A list of the useful tags to keep. Default is the list defined in this function.

    Returns
    -------
    networkx.MultiDiGraph
        A networkx graph representing the place's bike network.
    """
    try:
        ox.config(use_cache=True, log_console=True, useful_tags_way=useful_tags_way)
        G_nx = ox.graph_from_place(place, network_type='bike', simplify=simplify, retain_all=True)
    except (KeyError, ox.errors.OverpassBadRequestError) as e:
        raise ValueError(f"Error getting graph for place {place}: {e}")
    
    return G_nx


def save_graph_geopackage(G, filename):
    """
    Saves a networkx graph as a geopackage in the 'garaph_geopackage' folder of the current working directory.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The graph to save as a geopackage.
    filename : str
        The filename to use for the geopackage (without the .gpkg extension).

    Returns
    -------
    None
    """
    try:
        # Create the 'garaph_geopackage' folder if it doesn't exist
        if not os.path.exists('garaph_geopackage'):
            os.makedirs('garaph_geopackage')

        # Save the graph as a geopackage in the 'garaph_geopackage' folder
        filepath = f"garaph_geopackage/{filename}.gpkg"
        ox.save_graph_geopackage(G, filepath=filepath, directed=False)
    except Exception as e:
        print(f"Error saving graph as geopackage: {e}")


def save_graph_shapefile(network, fp):
    """
    Save the input graph to a shapefile.

    Parameters
    ----------
    network : networkx.MultiDiGraph
        Input graph to be saved.
    fp : str
        Filepath where the shapefile will be saved.

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If `osmnx` module does not have the `io` sub-module.
    TypeError
        If `fp` is not a string.

    Notes
    -----
    This function is deprecated in favor of `save_graph_geopackage()`, which
    saves the graph to the superior GeoPackage file format.

    The `network` should be a `networkx.MultiDiGraph` instance representing a
    street network. The function will save this graph to a set of shapefiles in
    the folder specified by `fp`. If the `fp` folder does not exist, it will be
    created.

    """
    # Check if `osmnx` module has the `io` sub-module
    if not hasattr(ox, 'io'):
        raise AttributeError("Module 'osmnx' has no attribute 'io'.")

    # Check if `fp` is a string
    if not isinstance(fp, str):
        raise TypeError("'fp' should be a string.")

    # Create the folder if it does not exist
    folder = os.path.dirname(fp)
    if folder != '':
        os.makedirs(folder, exist_ok=True)

    # Save the graph to the shapefile
    ox.io.save_graph_shapefile(network, filepath=fp, directed=False)

    # Print success message
    print(f"Graph saved to '{fp}' as a shapefile.")

def save_graph_csv(nodes, edges, filename):
    """
    Saves a graph as a CSV file with the given filename.

    Args:
        nodes (list): List of tuples representing nodes with (id, lon, lat) format.
        edges (list): List of tuples representing edges with (u, v, weight) format.
        filename (str): Name of the CSV file to be saved.

    Returns:
        None
    """
    # Create the 'graph_csv' folder if it doesn't exist
    if not os.path.exists('graph_csv'):
        os.makedirs('graph_csv')
    nodes.to_csv(f"{filename}_nodes.csv", index=False)
    edges.to_csv(f"{filename}_edges.csv", index=False)
    print(f"Saved nodes to {filename}_nodes.csv")
    print(f"Saved edges to {filename}_edges.csv")

def create_graph(edges, nodes):
    """
    Create a networkx graph object from two GeoDataFrames: an `edges` GeoDataFrame and a `nodes` GeoDataFrame.
    
    Parameters:
    edges (GeoDataFrame): A GeoDataFrame containing edge data for the network.
    nodes (GeoDataFrame): A GeoDataFrame containing node data for the network.
    
    Returns:
    G (Graph): A networkx graph object representing the network.
    """
    
    # Drop the 'geometry' column from the edges DataFrame
    col = edges.drop(columns=["geometry"]).columns.tolist()
    
    # Create a new GeoDataFrame with the same columns as the original edges DataFrame
    gdf = gpd.GeoDataFrame(edges, columns=col, geometry=edges['geometry'])
    
    # Convert the GeoDataFrames to a networkx graph object
    G = ox.graph_from_gdfs(nodes, gdf)
    
    return G

    
def read_geopackage_to_gdf(geopackage_path, layers):
    """
    Reads specified layers from a GeoPackage file and returns them as separate GeoDataFrames.

    Parameters
    ----------
    geopackage_path : str
        The file path of the GeoPackage file.
    layers : list of str
        The names of the layers to read from the GeoPackage file.

    Returns
    -------
    dict
        A dictionary of GeoDataFrames, with keys as the layer names and values as the GeoDataFrames.
    """
    gdfs = {}
    for layer in layers:
        gdf = gpd.read_file(geopackage_path, layer=layer)
        gdfs[layer] = gdf
    return gdfs


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
    cols_list = [ 'lanes', 'maxspeed', 'highway', 'surface', 'parking:lane:both', 'cycleway', 'bicycle'
    , 'cycleway:both', 'cycleway:left', 'cycleway:right', 'oneway:bicycle', 'cycleway:right:lane'
    , 'cycleway:left:lane', 'width', 'access', 'lit']
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

        print("Data cleaning completed successfully.")
    except Exception as e:
        print("An error occurred during data cleaning: ", e)

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

    # return df


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

    print("Maxspeed score set successfully!")

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
        print('The computation of parking and parking_score was successful.')



def set_separated_bike_la(df):
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
    print('Setting of separated_bike_la and facility_type in was successful.')

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
    print('Setting of shared_bike_la was successful.')
        
    # return df

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
    print('Setting of set_cycle_track was successful.')

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
    print('Setting of dirt_cycle_track was successful.')



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
    # return df
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
    print('Setting of no-access bike parameter was successful.')
    # return df


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
    print('Setting of path_cycle was successful.')


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
    print('score computation has benn successful done!')

def time_functions(df, functions):
    times = []
    for func in functions:
        start_time = time.time()
        func(df)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(functions)), times)
    ax.set_xticks(range(len(functions)))
    # ax.set_xticklabels([f.__name__ for f in functions])
    ax.set_ylabel('Time (seconds)')

    # Add labels to the bars
    top_funcs = sorted(range(len(times)), key=lambda i: times[i])[-3:]
    for i, bar in enumerate(bars):
        if i in top_funcs:
            bar.set_label(functions[i].__name__)

    ax.legend()
    plt.savefig("runtime.png")
    plt.show()

    num_rows = len(df)
    num_cols = len(df.columns)
    print(f"{num_rows} rows and {num_cols} columns took {elapsed_time} seconds.")





def set_edges_dgree(df, G):
    # Create new columns with values from the MultiIndex
    df = df.assign(new_u = df.index.get_level_values('u'), new_v=df.index.get_level_values('v'))
    spn = ox.utils_graph.count_streets_per_node(network)
    df['degree'] = df['new_v'].map(spn)
    return df

def count_frequency(data, column_name):
    """
    Counts the frequency of each value in the specified column of a pandas DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame to count the frequency of values in.
        column_name (str): The name of the column to count the frequency of values in.

    Returns:
        pandas.DataFrame: A DataFrame with two columns, "value" and "count", where "value" is a unique value from the specified column and "count" is the number of times that value appears in the column.
    """
    # consider the []'s value  it is slow
    # frequency_df = pd.DataFrame(edges["highway"].apply(pd.Series)[0].value_counts().reset_index())
    # fast but after cleaning
    frequency_df = pd.DataFrame(data[column_name].value_counts().reset_index())
    frequency_df.columns = ["value", "count"]
    return frequency_df


def graph_info(graph):
    """
    Takes a NetworkX graph object and returns information about its nodes and edges.
    """
    # node_count = graph.number_of_nodes()
    # edge_count = graph.number_of_edges()
    info = nx.info(graph)
    crs= network.graph["crs"]
    return f"edges.\n{info} crs= \n{crs} "


# def remove_columns(df, cols_to_remove):
#     """
#     Removes specified columns from a pandas DataFrame.

#     Args:
#         df (pandas DataFrame): The DataFrame to remove columns from.
#         cols_to_remove (list): A list of column names to remove from the DataFrame.

#     Returns:
#         pandas DataFrame: The modified DataFrame with the specified columns removed.
#     """
#     df.drop(df[df['type'].isin(cols_to_drop)].index, inplace=True)




place_name = 'Bonn, Germany'
hamburg = 'Hamburg , Germany'
koln = 'köln, germany'
pp = 'Poppelsdorf, Bonn, Germany'
hlw= 'Hoheluft-West , Hamburg, Germany'
network = get_place_network(hlw, simplify=True,
                                      useful_tags_way=useful_tags_way)

graph_info(network)


nodes, edges = ox.graph_to_gdfs(network, nodes=True, edges=True)

edges.crs
# edges.to_crs({'init': 'epsg:32632'})
edges.crs = {'init': 'epsg:32632'}

edges.head()
edges.info()

# edges['cycleway:both:bicycle'].unique()

# # edges.describe()
# type(edges['length'])

# compute run time
# functions = [clean_data, compute_default_and_lane_score, set_def_maxspeed_and_speed_score, compute_parking_and_parking_score, set_separated_bike_la, compute_shared_bike_lane, set_no_access, set_cycle_track, set_dir_surface, set_bik_no_infra, set_has_cycle, score_computation]
# time_functions(edges, functions)



start_time = time.time()
# edges.to_csv('köln_edge.csv') 
clean_data(edges)
compute_default_and_lane_score(edges)
set_def_maxspeed_and_speed_score(edges)
compute_parking_and_parking_score(edges)

set_separated_bike_la(edges)
compute_shared_bike_lane(edges)
set_no_access(edges)
set_cycle_track(edges)
set_dir_surface(edges)
set_bik_no_infra(edges)
set_has_cycle(edges)
score_computation(edges)



end_time = time.time()

elapsed_time = end_time - start_time
length = pd.Series(edges["length"])
total_length = edges["length"].sum() / 1000



G2 = create_graph(edges, nodes)
nodes, edges = ox.graph_to_gdfs(G2, nodes=True, edges=True)
edges.info()
fn= 'Bonn'
hum ='Hamburg'
pd = 'Poppelsdorf'
hw= 'Hoheluft-West'

save_graph_geopackage(G2, hw)
# df.to_csv('Bonn4_p_edge.csv') 


# the above codes are necessry for computing the scores:





# Statictisac analysis and visualisation is in bellow computed
# 
# 




edges['facility_type'].unique()
grouped = edges.groupby("facility_type")["length"].sum()/1000
ax = grouped.plot(kind="bar")
ax.set_xlabel("Facility Type")
ax.set_ylabel("Length (km)")
plt.show()


print("Elapsed time:", elapsed_time, "seconds")
# Assuming you already have the grouped data
grouped = edges.groupby("facility_type")["length"].sum()

# Convert length to percentage
total_length = grouped.sum()
grouped_pct = grouped / total_length * 100

# Plot the data
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(grouped_pct.index, grouped_pct.values)
ax.set_xlabel("Percentage of Total Length")
ax.set_ylabel("Facility Type")
ax.set_title("Length of Each Facility Type in Percentage")
plt.yticks(rotation=0)  # Rotate the tick labels to be vertical
plt.show()








osmids = list(network.nodes)
ox.plot.get_colors(n=5, cmap="plasma", return_hex=True)
nc = ox.plot.get_node_colors_by_attr(network, attr="y", cmap="plasma")
fig, ax = ox.plot_graph(network,node_size=1, edge_linewidth=0.3)

# plot the graph with colored edges
ec = ox.plot.get_edge_colors_by_attr(network, attr="length")
fig, ax = ox.plot_graph(network, node_size=0.5, edge_color=ec, bgcolor="k")

fig = no.matrix(edges)
fig.get_figure().savefig("matrix_data.png")




surface_types = pd.DataFrame(edges["surface"].apply(pd.Series)[0].value_counts().reset_index())
# dose not consider array
surface_types = edges["surface"].value_counts().reset_index()
total_count = surface_types["count"].sum()
surface_types.columns = ["type", "count"]
surface_types["percentage"] = (surface_types["count"] / total_count) * 100

street_types = pd.DataFrame(edges["highway"].apply(pd.Series)[0].value_counts().reset_index())
street_types.columns = ["type", "count"]
total_count = street_types["count"].sum()
rows_to_drop = ['secondary_link', 'tertiary_link', 'primary_link', 'road', 'disused', 'trunk', 'trunk_link', 'razed']
street_types = street_types[~street_types['type'].isin(rows_to_drop)]

street_types["percentage"] = (street_types["count"] / total_count) * 100
# remove_columns(street_types, cols_to_drop)


infra_types = pd.DataFrame(edges["facility_type"].apply(pd.Series)[0].value_counts().reset_index())
infra_types.columns = ["type", "count"]
total_count = infra_types["count"].sum()
infra_types["percentage"] = (infra_types["count"] / total_count) * 100

list(street_types['type'])



sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12,10))
# barplot, factorplot
# sns.lmplot(y="type", x="count", data=street_types)
sns.factorplot(y="type", x="percentage", data=infra_types, ax=ax)
ax.set_ylabel("Highway functional class", fontsize=16)  # modify y-label
ax.set_xlabel("Frequency", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.title("Distribution of Infrastructure Types", fontsize=20)
plt.tight_layout()
plt.savefig("Infrastructure_types.png")



street_types["percentage"] = (street_types["count"] / total_count) * 100
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12,10))
sns.factorplot(y="type", x="percentage", data=street_types, ax=ax)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel("Highway Functional Class", fontsize=16)
plt.title("Distribution of Street Types", fontsize=16)
plt.tight_layout()
plt.savefig("Highway_type.png")


fig, ax = plt.subplots(figsize=(12,10))
sns.barplot(x="percentage", y="type", data=street_types, ax=ax)
ax.set_ylabel("Highway Type", fontsize=16)
ax.set_xlabel("% of Highway types dataset", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.title("Distribution of Highway Type", fontsize=16)
plt.tight_layout()
plt.savefig("Highway_type.png")


edges.info()

has_cycle = pd.DataFrame(edges["has_cycle"].apply(pd.Series)[0].value_counts().reset_index())
surface_types = count_frequency(edges, "surface")


edges.to_csv('Bonn4_p_edge.csv')
G = create_graph(edges, nodes)
fn= 'Bonn'
save_graph_geopackage(G, fn)



# edges['surface'].unique()

# edges.to_csv('köln_p_edge.csv') 
# edges.to_csv('Bonn4_p_edge.csv') 



tags = {'amenity': True, 'amenity':['parking', 'fuel', 'bicycle_parking', 'parking', 'parking_space', 'taxi', 'parking_entrance', 'charging_station']}
gdf = ox.geometries_from_place(place_name, tags)



gdf.to_csv('Bonn4_amenities.csv')
# gdf.to_file("output.gpkg", driver="GPKG")
columns_to_keep = ['amenity', 'parking', 'geometry']
index = gdf.index
gdf = gdf[columns_to_keep].set_index(index)



gdf['geometry']
# gdf2 = pd.concat([edges, gdf], ignore_index=True)

gdf = gdf.reset_index(level='element_type')
gdf = gdf.reset_index(level='osmid')



gdf = gdf.reset_index(level=['element_type', 'osmid'])









# # load GeoPackage as node/edge GeoDataFrames indexed as described in OSMnx docs
# gdf_nodes = gpd.read_file('/Users/m-store/Desktop/Data_thesis/OMS/version_4/garaph_geopackage/graph.gpkg', layer='nodes').set_index('osmid')
# gdf_edges = gpd.read_file('/Users/m-store/Desktop/Data_thesis/OMS/version_4/garaph_geopackage/graph.gpkg', layer='edges').set_index(['u', 'v', 'key'])
# gdf_edges.info()
# len(useful_tags_way


address = 'Bonn, Germany'


fn = '/Users/m-store/Desktop/Data_thesis/OMS/version_4/garaph_geopackage/Bonn.gpkg'

layers = ["edges", "nodes"]
gdfs = read_geopackage_to_gdf(fn, layers)
gdf_edges = gdfs["edges"]
gdf_nodes = gdfs["nodes"]

clean_data(gdf_edges)


# bellow function about the historical data of OSM. 

def get_network_info(place_name, date):
    ox.config(use_cache=True, log_console=True, useful_tags_way=useful_tags_way, overpass_settings=f'[out:json][timeout:90][date:"{date}T00:00:00Z"]')
    G_nx = ox.graph_from_place(place_name, network_type='bike', simplify=True, retain_all=True)
    nodes, edges = ox.graph_to_gdfs(G_nx, nodes=True, edges=True)
    return {'place': place_name,
            'date': date,
            'num_nodes': len(nodes),
            'num_edges': len(edges)}

# def write_network_info_to_csv(place_name, years, filename):
#     # write header row to CSV file
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Year', 'Place', 'Num Nodes', 'Num Edges'])

#     # loop through years, get network info, and append to CSV file
#     for year in years:
#         date = f'{year}-01-01'
#         network_info = get_network_info(place_name, date)
#         with open(filename, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([year, network_info['place'], network_info['num_nodes'], network_info['num_edges'], network_info['avg_degree']])


years = ['2020', '2021', '2022', '2023']
place_name = 'Bonn, Germany'

network_info_list = []
for year in years:
    date = f'{year}-01-01'
    network_info = get_network_info(place_name, date)
    network_info_list.append(network_info)

df = pd.DataFrame(network_info_list)
print(df)



