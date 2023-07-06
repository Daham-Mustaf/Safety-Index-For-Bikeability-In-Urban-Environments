import networkx as nx
import osmnx as ox
import geopandas as gpd
import os


useful_tags_way = ['cycleway:left:buffer', 'oneway', 'lanes', 'source:maxspeed', 'surface', 'amenity',
                   'highway', 'maxspeed', 'parking:lane:both:parallel', 'class:bicycle',
                   'parking:lane:both', 'segregated', 'access', 'lit', 'sidewalk',
                   'parking:lane:right:parallel', 'parking:lane:left:parallel',
                   'cycleway:left', 'cycleway', 'bicycle', 'cycleway:left:buffer',
                   'cycleway:right:buffer', 'cycleway:right', 'cycleway:left', 'cycleway:both',
                   'width', 'parking:lane:left', 'parking:lane:right', 'oneway:bicycle',
                   'oneway:bus', 'cycleway:right:lane', 'cycleway:both:lane', 'sidewalk:right:bicycle',
                   'lanes:width', 'cycleway:left:lane', 'cycleway:both:bicycle',
                   'cycleway:right', 'unclassified', 'from', 'to']


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


# def graph_info(graph):
#     """
#     Takes a NetworkX graph object and returns information about its nodes and edges.
#     """
#     # node_count = graph.number_of_nodes()
#     # edge_count = graph.number_of_edges()
#     info = nx.info(graph)
#     crs= network.graph["crs"]
#     return f"edges.\n{info} crs= \n{crs} "

def convert_graph_to_gdfs(graph):
    """
    Converts a networkx graph to GeoDataFrames (nodes and edges) using ox.graph_to_gdfs.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The networkx graph to convert.

    Returns
    -------
    tuple of geopandas.GeoDataFrame
        A tuple containing the nodes GeoDataFrame and edges GeoDataFrame.
    """
    try:
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        return nodes, edges
    except Exception as e:
        print(f"Error converting graph to GeoDataFrames: {e}")
        return None, None


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
        # Get the current working directory
        base_dir = os.getcwd()

        # Specify the directory to save the file
        save_directory = os.path.join(base_dir, "data", "processed", "garaph_geopackage")

        # Create the 'garaph_geopackage' folder if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the graph as a geopackage in the 'garaph_geopackage' folder
        filepath = os.path.join(save_directory, f"{filename}.gpkg")
        ox.save_graph_geopackage(G, filepath=filepath, directed=False)

        print("Graph saved as geopackage successfully!")
    except Exception as e:
        print(f"Error saving graph as geopackage: {e}")