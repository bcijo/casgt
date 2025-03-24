import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
import networkx as nx
import torch
import threading
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import scipy.sparse as sp
import random
import argparse
from collections import defaultdict
from common import get_dataset_nodes
from cegt import GraphTransformer, GraphTransformerWithClassificationHead, FocalLoss
from fea_extra import FeaExtra, FeaMoreExtra  # Import FeaExtra and FeaMoreExtra for feature extraction
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from functools import wraps

def load_and_preprocess_test_data(filename, num_nodes, embed_dim):
    """
    Load and preprocess test data from a file.
    
    Args:
    - filename: Path to the test dataset.
    - num_nodes: Number of nodes in the graph.
    
    Returns:
    - test_edge_list: List of tuples representing edges (source, destination).
    - test_labels: Tensor of shape (num_edges,) with values 1 for positive edges and 0 for negative edges.
    - node_features: Node features tensor for test nodes.
    - centrality_features: Centrality features for test nodes.
    """
    # Load test edges and labels
    test_edge_list, test_labels = load_test_data(filename)

    # Create a graph from the test edges to compute centrality features
    graph = nx.Graph()
    graph.add_edges_from(test_edge_list)
    centrality_features = compute_centrality_features_for_test(test_edge_list, num_nodes)
    node_features = nn.Embedding(num_nodes, embed_dim).to('cuda')  # Adjust the device if needed
    node_features.weight.requires_grad = True

    return test_edge_list, test_labels, node_features, centrality_features

def evaluation_metrics_decorator(func):
    """
    Decorator to compute evaluation metrics for inference results.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Execute the original inference function
        preds, true_labels = func(*args, **kwargs)
        # Convert predictions to binary (0 or 1)
        binary_preds = (torch.sigmoid(preds) > 0.5).int()
        
        # Calculate evaluation metrics
        acc = accuracy_score(true_labels, binary_preds)
        prec = precision_score(true_labels, binary_preds)
        rec = recall_score(true_labels, binary_preds)
        f1 = f1_score(true_labels, binary_preds)
        auc = roc_auc_score(true_labels, torch.sigmoid(preds).detach().cpu().numpy())
        
        # Print the metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return preds, true_labels
    return wrapper


def load_data2(filename='', add_public_foe=True):

    adj_lists1   = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2   = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3   = defaultdict(set)


    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            person1 = int(info[0])
            person2 = int(info[1])
            v = int(info[2])
            adj_lists3[person2].add(person1)
            adj_lists3[person1].add(person2)

            if v == 1:
                adj_lists1[person1].add(person2)
                adj_lists1[person2].add(person1)

                adj_lists1_1[person1].add(person2)
                adj_lists1_2[person2].add(person1)
            else:
                adj_lists2[person1].add(person2)
                adj_lists2[person2].add(person1)

                adj_lists2_1[person1].add(person2)
                adj_lists2_2[person2].add(person1)


    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3

def load_test_data(filename):
    """
    Load test data from file.
    
    Args:
    - filename: Path to the test dataset.
    
    Returns:
    - edge_list: List of tuples representing edges (source, destination).
    - labels: 1D Tensor of shape (num_edges,) with values 1 for positive edges and 0 for negative edges.
    """
    edge_list = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # Skip malformed lines

            source = int(parts[0])
            destination = int(parts[1])
            label = int(parts[2])

            edge_list.append((source, destination))
            
            # Convert label: 1 if label is 1, 0 if label is -1
            labels.append(1 if label == 1 else 0)

    # Convert labels to a 1D tensor of shape (num_edges,)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Ensure that the shape is (num_edges,)
    assert labels.shape == (len(edge_list),), f"Expected labels shape ({len(edge_list)},), but got {labels.shape}"

    return edge_list, labels


def calculate_node_sign_influence_from_file(filename, num_nodes, alpha=1.0, beta=0.1):
    """
    filename: Path to the file containing edge data (source, destination, sign)
    num_nodes: Total number of nodes in the graph
    alpha: Scaling factor for net influence

    
    beta: Bias term for degree-based contribution
    
    Returns:
    node_sign_influence: Tensor of shape (num_nodes,) with values between -1 and 1
    """

    # Initialize counts for positive and negative edges for each node
    pos_count = torch.zeros(num_nodes, dtype=torch.float32)
    neg_count = torch.zeros(num_nodes, dtype=torch.float32)

    # Read the edge data from the file
    with open(filename, 'r') as file:
        for line in file:
            # Each line is in the format: "source destination sign"
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # Skip any malformed lines

            source = int(parts[0])
            destination = int(parts[1])
            sign = int(parts[2])

            # Update the counts based on the sign of the edge
            if sign == 1:
                # Add to positive count for both source and destination nodes
                pos_count[source] += 1
                pos_count[destination] += 1
            elif sign == -1:
                # Add to negative count for both source and destination nodes
                neg_count[source] += 1
                neg_count[destination] += 1

    # Calculate the total degree for each node
    total_count = pos_count + neg_count

    # Calculate net influence for each node, normalized by total degree
    net_influence = (pos_count - neg_count) / (total_count + 1e-6)  # Avoid division by zero

    # Apply tanh to net influence to get a smooth value between -1 and 1
    net_influence = torch.tanh(alpha * net_influence)

    # Apply degree-based scaling to account for node connectivity
    degree_scaled = beta * (total_count / (total_count.max() + 1e-6))  # Normalize by max degree

    # Calculate final sign influence
    # We add the degree-based term with the same sign as the net influence
    node_sign_influence = net_influence + torch.sign(net_influence) * degree_scaled
    node_sign_influence = torch.clamp(node_sign_influence, -1, 1)  # Ensure values are between -1 and 1

    return node_sign_influence


import multiprocessing as mp
import networkx as nx
import torch
from tqdm import tqdm
import threading
import time

def compute_betweenness(graph, progress_dict, task_name):
    result = nx.betweenness_centrality(graph)
    progress_dict[task_name] = 1  # Mark as complete
    return result

def compute_closeness(graph, progress_dict, task_name):
    result = nx.closeness_centrality(graph)
    progress_dict[task_name] = 1  # Mark as complete
    return result

def update_progress(progress_dict, total_tasks):
    with tqdm(total=total_tasks, desc="Computing Centrality Features", unit="task") as pbar:
        completed_tasks = 0
        while completed_tasks < total_tasks:
            completed_tasks = sum(progress_dict.values())
            pbar.n = completed_tasks
            pbar.refresh()
            time.sleep(0.1)  # Sleep briefly to avoid excessive CPU usage
        pbar.n = total_tasks
        pbar.refresh()

def compute_centrality_features(graph, num_nodes):
    # Use a Manager dictionary to track progress across processes
    manager = mp.Manager()
    progress_dict = manager.dict({
        "betweenness": 0,
        "closeness": 0
    })
    total_tasks = 2  # Number of centrality measures

    # Start the progress bar in a separate thread
    progress_thread = threading.Thread(target=update_progress, args=(progress_dict, total_tasks))
    progress_thread.start()

    # Use 2 processes to compute the 2 centrality measures in parallel
    with mp.Pool(processes=2) as pool:
        # Launch each computation asynchronously
        betweenness_future = pool.apply_async(compute_betweenness, (graph, progress_dict, "betweenness"))
        closeness_future = pool.apply_async(compute_closeness, (graph, progress_dict, "closeness"))

        # Retrieve the results
        betweenness = betweenness_future.get()
        closeness = closeness_future.get()

    # Wait for the progress thread to finish
    progress_thread.join()

    # Initialize the centrality features dictionary
    centrality_features = {
        "betweenness": torch.zeros(num_nodes, dtype=torch.float32),
        "closeness": torch.zeros(num_nodes, dtype=torch.float32)
    }

    # Assign computed values to the dictionary with a progress bar
    for node in tqdm(range(num_nodes), desc="Assigning Centrality Features"):
        centrality_features["betweenness"][node] = betweenness.get(node, 0)
        centrality_features["closeness"][node] = closeness.get(node, 0)

    return centrality_features


def compute_centrality_features_for_test(edge_list, num_nodes):
    """
    Compute centrality measures for nodes in the test graph, including isolated nodes.
    """
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    
    # Calculate centralities for all nodes, including isolated nodes
    centrality_features = {
        "betweenness": torch.zeros(num_nodes, dtype=torch.float32),
        "closeness": torch.zeros(num_nodes, dtype=torch.float32)
    }

    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    for node in tqdm(range(num_nodes), desc="Computing Test Centrality Features"):
        centrality_features["betweenness"][node] = betweenness.get(node, 0)
        centrality_features["closeness"][node] = closeness.get(node, 0)

    return centrality_features

def prepare_adjacency_matrix(adj_list, num_nodes):
    """
    Prepare adjacency matrix from adjacency list.
    """
    edges = []
    for a in tqdm(adj_list, desc="Preparing Adjacency Matrix"):
        for b in adj_list[a]:
            edges.append((a, b))
    
    edges = np.array(edges)
    adj = torch.sparse_coo_tensor(edges.T, torch.ones(len(edges)), (num_nodes, num_nodes))
    
    return adj


def extract_features(adj_lists1, adj_lists2, adj_lists3, added_info, dataset, k):
    """
    Extract additional features using FeaExtra or FeaMoreExtra.
    """
    if added_info:
        print(added_info)
        fea_model = FeaMoreExtra(dataset=dataset, k=k)
    else:
        fea_model = FeaExtra(dataset=dataset, k=k)
    
    adj_additions1 = [defaultdict(set) for _ in range(16)]
    adj_additions2 = [defaultdict(set) for _ in range(16)]
    adj_additions3 = [defaultdict(set) for _ in range(4)]  # Adjusted from 16 to 4
    adj_additions0 = [defaultdict(set) for _ in range(16)]
    a, b = 0, 0

    for i in tqdm(adj_lists3, desc="Extracting Features for adj_lists3"):
        for j in adj_lists3[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions0[index][i].add(j)

    print(f"Number of nodes with positive edges in adj_lists1: {len(adj_lists1)}")
    for i in tqdm(adj_lists1, desc="Extracting Features for adj_lists1"):
        for j in adj_lists1[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions1[index][i].add(j)
                    a += 1

    print(f"Number of nodes with negative edges in adj_lists2: {len(adj_lists2)}")
    for i in tqdm(adj_lists2, desc="Extracting Features for adj_lists2"):
        for j in adj_lists2[i]:
            v_list = fea_model.feature_part2(i, j)
            for index, v in enumerate(v_list):
                if v > 0:
                    adj_additions2[index][i].add(j)
                    b += 1

    if a == 0:
        print("Warning: No positive features extracted from positive edges in adj_lists1")
    if b == 0:
        print("Warning: No positive features extracted from negative edges in adj_lists2")
    else:
        assert b > 0, 'negative something wrong'

    if added_info:
        graphs = nx.DiGraph()
        graph = {key: list(value) for key, value in adj_lists3.items()}
        for node, neighbors in tqdm(graph.items(), desc="Adding Edges to Graph"):
            for neighbor in neighbors:
                graphs.add_edge(node, neighbor)
                
        betweenness_centrality, closeness_centrality = fea_model.compute_centralities_and_clustering(graphs)
        for node, neighbors in tqdm(graph.items(), desc="Computing Additional Features"):
            for neighbor in neighbors:
                additional_features = fea_model.features_part3(node, neighbor, betweenness_centrality, closeness_centrality)
                for index, feature in enumerate(additional_features):
                    adj_additions3[index][node].add(feature) 

        return adj_additions0 + adj_additions1 + adj_additions2 + adj_additions3
    
    else:
        return adj_additions1 + adj_additions2 + adj_additions0
    
class CE_GCT_Trainer:
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, adj_lists, aggs, device, model_type='pt', learning_rate=0.001, weight_decay=0.0001, dropout_rate=0.1):
        """
        Initialize the CE_GCT_Trainer with conditional model loading.
        
        Args:
        - model_type (str): 'pt' for pretrain or 'ft' for finetune.
        """
        self.device = device
        self.model_type = model_type

        if model_type == 'pt':
            # Initialize Graph Transformer model for pretraining
            self.model = GraphTransformer(
                node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, adj_lists, aggs, device, dropout_rate
            ).to(device)

            self.model.load_state_dict(torch.load("/home/cegt/embeddings/cegt/bitcoin_alpha/DCO/bitcoin_alpha_54/ce_gct_pretrained.pth"))
            # self.model.load_state_dict(torch.load("/path/to/pretrained_model.pth"))  # Adjust path as needed
        elif model_type == 'ft':
            self.model = GraphTransformerWithClassificationHead(
                node_feat_dim=node_feat_dim, 
                embed_dim=embed_dim, 
                centrality_dim=centrality_dim, 
                num_heads=num_heads, 
                num_layers=num_layers, 
                adj_lists=adj_lists, 
                aggs=aggs, 
                device=device, 
                dropout_rate=dropout_rate
            ).to(device)
            self.model.load_state_dict(torch.load("/home/cegt/embeddings/cegt/bitcoin_alpha/DCO/bitcoin_alpha_54/ce_gct_pretrained.pth"), strict=False)

            for param in self.model.transformer.parameters():
                param.requires_grad = False
        else:
            self.model = GraphTransformerWithClassificationHead(
                node_feat_dim=node_feat_dim, 
                embed_dim=embed_dim, 
                centrality_dim=centrality_dim, 
                num_heads=num_heads, 
                num_layers=num_layers, 
                adj_lists=adj_lists, 
                aggs=aggs, 
                device=device, 
                dropout_rate=dropout_rate
            ).to(device)
            self.model.load_state_dict(torch.load("/home/cegt/embeddings/cegt/bitcoin_alpha/DCO/bitcoin_alpha_54/ce_gct_finetuned.pth"))
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def pretrain(self, nodes, node_features, centrality_features, adj_matrix, node_sign_influence, epochs, save_path='ce_gct_pretrained.pth'):
        """
        Unsupervised pretraining for graph reconstruction.
        """
        if self.model_type != 'pt':
            raise RuntimeError("Model type is set to 'ft'. Use 'pt' for pretraining.")
        
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Pretraining Epochs"):
            self.optimizer.zero_grad()
            
            # Forward pass
            output_embeddings = self.model(nodes, node_features, centrality_features, adj_matrix, node_sign_influence)

            # Reconstruct adjacency matrix
            reconstructed_adj = torch.mm(output_embeddings, output_embeddings.t())
            loss = self.criterion(reconstructed_adj, adj_matrix.to_dense())
            loss.backward()
            self.optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Pretraining Loss: {loss.item():.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def finetune(self, nodes, node_features, centrality_features, adj_matrix, node_sign_influence, train_labels, edge_list, epochs=10, save_path='ce_gct_finetuned.pth'):
        """
        Supervised fine-tuning for signed link prediction using test data, with negative sampling.
        """
        if self.model_type != 'ft':
            raise RuntimeError("Model type is set to 'pt'. Use 'ft' for fine-tuning.")
        
        num_nodes = node_features.num_embeddings
        nodes = torch.tensor(nodes, dtype=torch.long).to(self.device)
        centrality_features = {k: v.to(self.device) for k, v in centrality_features.items()}
        adj_matrix = adj_matrix.to(self.device)
        self.model.train()
        focal_loss = FocalLoss(alpha=1, gamma=2)

        print("Model device:", next(self.model.parameters()).device)
        print("Nodes device:", nodes.device)
        print("Node features device:", node_features.weight.device)
        print("Centrality features device:", centrality_features["betweenness"].device)
        print("Adj matrix device:", adj_matrix.device)

        for epoch in tqdm(range(epochs), desc="Fine-tuning Epochs"):
            self.optimizer.zero_grad()
            node_sign_influence = node_sign_influence.to(self.device)
            print("number of nodes: {}".format(len(nodes)))
            logits = self.model(nodes, node_features, centrality_features, adj_matrix, node_sign_influence, edge_list)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(self.device)
            loss = focal_loss(logits, train_labels_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Fine-tuning Loss: {loss.item():.4f}")
            
            if save_path:
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

    @evaluation_metrics_decorator
    def inference(self, nodes, node_features, centrality_features, adj_matrix, node_sign_influence, edge_list, test_file, model_path='ce_gct_finetuned.pth'):
        # Load the model, mapping tensors to the correct device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)  # Ensure the model itself is on the correct device
        self.model.eval()
        
        # Move inputs to the correct device
        nodes = torch.tensor(nodes, dtype=torch.long).to(self.device)
        node_features = node_features.to(self.device)
        centrality_features = {k: v.to(self.device) for k, v in centrality_features.items()}
        adj_matrix = adj_matrix.to(self.device)
        node_sign_influence = node_sign_influence.to(self.device)
        edge_list = torch.tensor(edge_list, dtype=torch.long).to(self.device) if isinstance(edge_list, list) else edge_list.to(self.device)
        
        # Load and preprocess test data
        num_nodes = node_features.num_embeddings
        embed_dim = node_features.embedding_dim
        _, test_labels, test_node_features, test_centrality_features = load_and_preprocess_test_data(test_file, num_nodes, embed_dim)
        test_labels = test_labels.to(self.device)
        test_node_features = test_node_features.to(self.device)
        test_centrality_features = {k: v.to(self.device) for k, v in test_centrality_features.items()}
        
        # Inference on test data
        with torch.no_grad():
            preds = self.model(nodes, node_features, centrality_features, adj_matrix, node_sign_influence, edge_list)
        
        return preds, test_labels


# Argument Parser and Main Execution
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='cpu', help='Devices')
    parser.add_argument('--seed', type=int, default=13, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.012, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset')
    parser.add_argument('--dim', type=int, default=16, help='Embedding Dimension')
    parser.add_argument('--fea_dim', type=int, default=20, help='Feature Embedding Dimension')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--k', default=1, help='Folder k')
    parser.add_argument('--output_dir', default='output', type=str, help='Output directory')
    parser.add_argument('--added_info', default= None,type = str, help='changes made')
    parser.add_argument('--model_type', default= None,type = str, help='changes made')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    NEG_LOSS_RATIO = 1
    INTERVAL_PRINT = 20

    # NUM_NODE = DATASET_NUM_DIC[args.dataset]
    WEIGHT_DECAY = args.weight_decay
    NODE_FEAT_SIZE = args.fea_dim
    EMBEDDING_SIZE1 = args.dim
    DEVICES = torch.device(args.devices)
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DROUPOUT = args.dropout
    K = args.k
    MODEL_TYPE = args.model_type
    # num_nodes = DATASET_NUM_DIC[args.dataset] + 3

    # Load data and prepare adjacency lists
    filename = '/home/cegt/experiment-data/{}/{}-train-{}.edgelist'.format(args.dataset, args.dataset, args.k)
    num_nodes = get_dataset_nodes(filename)
    NUM_NODE = num_nodes
    
    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3 = load_data2(filename, add_public_foe=False)
    features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features.weight.requires_grad = True

    features.to(args.devices)

    adj_lists = [adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2]
    print(f"Data loaded for {args.dataset} with folder {args.k}.")

    adj_lists1 = extract_features(adj_lists1, adj_lists2, adj_lists3, args.added_info, args.dataset, args.k)
    adj_lists = adj_lists + adj_lists1 


    def func(adj_list):
        edges = []
        for a in tqdm(adj_list, desc="Creating Adjacency List Edges"):
            for b in adj_list[a]:
                edges.append((a, b))
        edges = np.array(edges)
        if len(edges) == 0: # fix missing motifs edges 
            edges = np.array([[0, 0]]) 
        adj = sp.csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(num_nodes, num_nodes))
        return adj

    # Compute centrality features
    graph = nx.from_dict_of_lists(adj_lists3)
    centrality_features = compute_centrality_features(graph, num_nodes)
    centrality_dim = 2

    adj_matrix = prepare_adjacency_matrix(adj_lists3, num_nodes)
    adj_lists = list(map(func, adj_lists))
    adj_lists = [matrix for matrix in adj_lists if matrix.nnz > 1]

    node_sign_influence = calculate_node_sign_influence_from_file(filename, num_nodes)
    # Initialize and train the model
    trainer = CE_GCT_Trainer(
    NODE_FEAT_SIZE, args.dim, centrality_dim, 12, 20, adj_lists, [], 
    torch.device(args.devices), learning_rate=args.lr, weight_decay=args.weight_decay, 
    dropout_rate=args.dropout, model_type=MODEL_TYPE
    )
    # output_dir = args.output_dir
    output_dir = "/home/cegt/output"
    os.makedirs(output_dir, exist_ok=True)

    if MODEL_TYPE == "pt":
        trainer.pretrain(list(range(num_nodes)), features, centrality_features, adj_matrix, node_sign_influence, epochs=args.epochs, save_path=os.path.join(output_dir, f'ce_gct_pretrained_{args.dataset}-{args.epochs}-epochs.pth'))
    elif MODEL_TYPE == 'ft':
        edge_list = load_test_data(filename)[0]
        trainer.finetune(list(range(num_nodes)), features, centrality_features, adj_matrix, node_sign_influence, load_test_data(filename)[1],edge_list, epochs=args.epochs, save_path=os.path.join(output_dir, f'ce_gct_finetuned_{args.dataset}-{args.epochs}-epochs.pth'))
    else:
        # Inference
        edge_list = load_test_data(filename.replace("train", "test"))[0]
        link_predictions, true_labels = trainer.inference(
        list(range(num_nodes)), features, centrality_features, adj_matrix, 
        node_sign_influence, edge_list, filename.replace("train", "test"), 
        model_path=os.path.join(output_dir, f'ce_gct_finetuned_{args.dataset}-{args.epochs}-epochs.pth')
        )
        print("Link predictions:", link_predictions)

if __name__ == "__main__":
    main()
