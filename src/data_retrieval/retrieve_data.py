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
        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Convert list values to strings in the GeoDataFrame
        for col in gdf.columns:
            if gdf[col].dtype == object and isinstance(gdf[col][0], list):
                gdf[col] = gdf[col].apply(lambda x: ','.join(x))

        # Save the GeoDataFrame as GeoJSON
        gdf.to_file(output_file, driver='GeoJSON')

        print("GeoJSON saved successfully!")
    except Exception as e:
        print("An error occurred while saving GeoJSON:", e)
