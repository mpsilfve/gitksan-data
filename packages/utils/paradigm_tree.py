"""Module for generating a Paradigm dependency tree. 
"""
from ..eval.eval_sigm import distance
import networkx as nx

def gen_fully_connected_graph(form_list):
    """Generates a fully connected graph between wordforms in the list, with the (undirected)
    edge-weight being the levenshtein distance between the words.  

    Args:
        form_list ([str]): list of forms in the paradigm.
    """
    p_graph = nx.Graph()# paradigm graph
    for i in range(len(form_list)):
        for j in range(len(form_list)):
            if i != j: # No self connections
                src_word = form_list[i]
                tgt_word = form_list[j]
                p_graph.add_edge(src_word, tgt_word, weight=distance(src_word, tgt_word))
    return p_graph

def obtain_mst(graph):
    """[summary]

    Args:
        graph (networkx.Graph): Graph with wordforms and weighted edges.
    
    Returns: 
        (networkx.graph)
    """
    mst = nx.algorithms.tree.mst.minimum_spanning_tree
    return mst(graph)

def calculate_mst_weight(mst):
    """Calculate the total weight of the edges in the MST. 

    Args:
        mst (networkx.Graph): MST with wordforms and weighted edges. 
    """
    total_weight = 0
    for (u, v, wt) in mst.edges.data('weight'):
        total_weight += wt
    return total_weight