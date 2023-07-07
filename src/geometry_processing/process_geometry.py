import osmnx as ox

class NodeIDRetrievalError(Exception):
    pass

def get_node_ids(graph):
    try:
        # Get a list of all node IDs in the graph
        node_ids = list(graph.nodes)
        return node_ids
    except Exception as e:
        print("An error occurred while retrieving node IDs:", e)
        return None
    
def compute_coordinates(graph):
    # Get the list of node IDs
    node_ids = get_node_ids(graph)

    if node_ids is not None:
        # Compute coordinates for each node ID using a list comprehension
        coordinates = [(graph.nodes[node_id]['x'], graph.nodes[node_id]['y']) for node_id in node_ids]
        return coordinates
    else:
        print("Error occurred. Node IDs could not be retrieved.")
        return None
    
def print_graph_project(graph):
    # Project the graph
    graph_proj = ox.project_graph(graph)

    # Print the projected graph
    print(graph_proj)