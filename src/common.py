import os
import logging
import time
import sys
import networkx as nx

SINE_MODEL_PATH_DIC = {
    'epinions': './embeddings/sine_epinions_models',
    'slashdot': './embeddings/sine_slashdot_models',
    'bitcoin_alpha': './embeddings/sine_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/sine_bitcoin_otc_models'
}

SIDE_MODEL_PATH_DIC = {
    'epinions': './embeddings/side_epinions_models',
    'slashdot': './embeddings/side_slashdot_models',
    'bitcoin_alpha': './embeddings/side_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/side_bitcoin_otc_models'
}

def verify_sequential_labels(G):
    """
    Verify if the nodes in graph G are renumbered sequentially.
    
    Parameters:
    - G: NetworkX graph where nodes are expected to be renumbered (e.g., 0, 1, 2, ...)
    
    This function prints:
    - The total number of unique nodes.
    - The maximum node label.
    - A message indicating whether the nodes are sequentially numbered.
    """
    unique_nodes = set(G.nodes())
    count = len(unique_nodes)
    
    try:
        max_node = max(unique_nodes)
    except TypeError:
        print("Node labels are not numeric. Cannot verify sequential numbering.")
        return

    print("Number of unique nodes:", count)
    print("Maximum node label:", max_node)
    
    if count > 0 and max_node == count - 1:
        print("Nodes are sequentially renumbered (0 to {}), with no gaps.".format(max_node))
        return False
    else:
        print("Nodes are not sequentially renumbered. There might be gaps or non-sequential labeling.")
        return True

def get_dataset_nodes(filename):
    G = nx.read_edgelist(filename, create_using=nx.DiGraph(), data=(("sign", int),))  
    num_nodes = G.number_of_nodes()
    if verify_sequential_labels(G):
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        sampled_G = nx.relabel_nodes(G, mapping)
        nx.write_edgelist(sampled_G, filename, data=['sign'])
    return num_nodes + 3

# DATASET_NUM_DIC = {
#     'epinions': 131828,
#     'slashdot': 82140,
#     'bitcoin_alpha': 3653,
#     'bitcoin_otc': 5881,
# }

EMBEDDING_SIZE = 20