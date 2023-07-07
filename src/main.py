# main.py
from data_processing.process_data import (
clean_data,compute_default_and_lane_score,
set_def_maxspeed_and_speed_score,
compute_parking_and_parking_score,set_separated_bike_lane,
compute_shared_bike_lane, set_no_access, set_cycle_track,
set_dir_surface,
set_bik_no_infra,
set_has_cycle,
score_computation,
)
from graph_creation.create_graph import get_place_network, convert_graph_to_gdfs, create_graph, save_graph_geopackage
from data_retrieval.retrieve_data import save_row_data_as_csv, save_processed_data_as_csv, save_as_geojson
from visualization.visualize_data import visualize_data
from geometry_processing.process_geometry import get_node_ids, compute_coordinates

def main():
    # set up tag settings
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

    bonn = 'Bonn, Germany'
    hamburg = 'Hamburg, Germany'
    koln = 'Köln, Germany'
    pp = 'Poppelsdorf, Bonn, Germany'
    hlw = 'Hoheluft-West, Hamburg, Germany'
    fn= 'Bonn' 
    # Step 1: Retrieve data from place and get a graph.
    network = get_place_network(bonn, simplify=True, useful_tags_way=useful_tags_way)
    
    if network is None:
        print("Error: Failed to retrieve network.")
        return

    # Step 2: Convert graph to GeoDataFrames
    nodes, edges = convert_graph_to_gdfs(network)

    if nodes is None or edges is None:
        print("Error: Failed to convert graph to GeoDataFrames.")
        return
        # Step 3: Clean data
        
     # Save row data as a CSV file
    save_row_data_as_csv(edges, "raw_edges.csv")
    
    cleaned_edges = clean_data(edges)
    clean_data(cleaned_edges)
    compute_default_and_lane_score(cleaned_edges)
    set_def_maxspeed_and_speed_score(cleaned_edges)
    compute_parking_and_parking_score(cleaned_edges)
    set_separated_bike_lane(cleaned_edges)
    compute_shared_bike_lane(cleaned_edges)
    set_no_access(cleaned_edges)
    set_cycle_track(cleaned_edges)
    set_dir_surface(cleaned_edges)
    set_bik_no_infra(cleaned_edges)
    set_has_cycle(cleaned_edges)
    score_computation(cleaned_edges)
    # visualize_data(cleaned_edges)
   

    # Get a list of all node IDs in the graph
    node_ids = get_node_ids(network)
    # Print the list of node IDs
    print(node_ids)
    
    # Call the function to compute coordinates
    result = compute_coordinates(network)

    if result is not None:
        # Print the computed coordinates
        for lon, lat in result:
            print(f"Lon: {lon}, Lat: {lat}")

    G2 = create_graph(cleaned_edges, nodes)
    save_graph_geopackage(G2, fn)
    
    # # save_as_geojson(cleaned_edges, 'output.geojson')
    save_processed_data_as_csv(cleaned_edges, "scored_edges.csv")
    
  
    


if __name__ == '__main__':
    main()
