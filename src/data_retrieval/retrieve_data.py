import os
import geopandas as gpd



def save_row_data_as_csv(df, filename):
    """
    Saves the DataFrame as a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        filename (str): The name of the CSV file.

    Returns:
        None
    """
    try:
        # Get the current working directory
        base_dir = os.getcwd()

        # Specify the directory to save the file
        save_directory = os.path.join(base_dir, "data", "raw")

        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save the DataFrame as a CSV file
        file_path = os.path.join(save_directory, filename)
        df.to_csv(file_path, index=False)
        print(f"\n Row data {filename} saved at: {file_path}")
        print("*****************\n")
    except Exception as e:
        print("An error occurred while saving the data:", e)


def save_processed_data_as_csv(df, filename):
    """
    Saves the DataFrame as a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        filename (str): The name of the CSV file.

    Returns:
        None
    """
    try:
        # Get the current working directory
        base_dir = os.getcwd()

        # Specify the directory to save the file
        save_directory = os.path.join(base_dir, "data", "processed")

        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save the DataFrame as a CSV file
        file_path = os.path.join(save_directory, filename)
        df.to_csv(file_path, index=False)
        print(f"\n Processed data {filename} saved at: {file_path}")
        print("*****************\n")
    except Exception as e:
        print("An error occurred while saving the data:", e)



def save_as_geojson(df, output_file):
    """
    Saves a DataFrame as a GeoJSON file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved as GeoJSON.
        output_file (str): The path of the output GeoJSON file.

    Returns:
        None
    """
    try:
        # Get the current working directory
        base_dir = os.getcwd()

        # Specify the directory to save the file
        save_directory = os.path.join(base_dir, "data", "processed")

        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Convert the Shapely geometries to WKT format
        df['geometry'] = df['geometry'].apply(lambda x: wkt.dumps(x))

        # Create a GeoDataFrame from the DataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Save the GeoDataFrame as a GeoJSON file
        file_path = os.path.join(save_directory, output_file)
        gdf.to_file(file_path, driver='GeoJSON')

        print(f"GeoJSON saved successfully at: {file_path}")
    except Exception as e:
        print("An error occurred while saving the data:", e)
