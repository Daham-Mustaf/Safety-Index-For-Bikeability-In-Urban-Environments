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