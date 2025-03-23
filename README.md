# CA-SGT: Centrality-Aware Signed Graph Transformer
CEGT (Centrality-Enhanced Graph Transformer) is a graph-based deep learning framework specifically designed for signed network analysis. It leverages node centrality measures and graph transformer architecture to effectively model complex relationships in signed networks.
## Overview
Signed networks, where edges can have positive or negative weights, are common in many real-world applications such as trust networks, social media platforms, and financial systems. CEGT enhances traditional graph neural networks by integrating:

1. Node centrality metrics (betweenness and closeness)
2. Sign-aware attention mechanisms
3. Edge feature extraction based on local network structures
4. Pre-training and fine-tuning pipeline for better performance

This implementation is particularly effective for tasks like signed link prediction, where the goal is to predict whether a new edge between two nodes would be positive or negative.
Features

- Signed Attention Mechanism: Custom attention mechanism that considers edge signs
- Centrality-Aware Encoder: Integrates node centrality metrics into the model
- Feature Extraction: Extensive feature engineering pipeline for signed networks
- Focal Loss: Implements focal loss to handle class imbalance
- Multi-stage Training: Supports pre-training, fine-tuning, and inference stages

## Installation
### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/bcijo/casgt.git
cd casgt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file should include:
```bash
torch>=1.9.0
networkx>=2.6
scipy>=1.7.0
numpy>=1.20.0
tqdm>=4.62.0
scikit-learn>=0.24.0
```
## Usage
### Data Preparation
Place your network data in the edgelist format in the experiment-data/{dataset}/ directory. Each file should contain edges in the format:
```
source_node destination_node sign
```
where sign is 1 for positive edge and -1 (or 0) for negative edge.
## Running the Model
### Pre-training
To pre-train the model on a dataset (e.g., bitcoin_alpha):
```bash
python main.py --dataset bitcoin_alpha --model_type pt --epochs 50 --lr 0.001 --devices cuda:0
```
### Fine-tuning
After pre-training, fine-tune the model for the signed link prediction task:
```bash
python main.py --dataset bitcoin_alpha --model_type ft --epochs 20 --lr 0.0005 --devices cuda:0
```
### Inference
Run inference on test data using the fine-tuned model:
```
python main.py --dataset bitcoin_alpha --model_type inf --devices cuda:0
```
## Command Line Arguments

--`devices`: Computing device (e.g., 'cpu', 'cuda:0')\
--`seed`: Random seed for reproducibility\
--`epochs`: Number of training epochs\
--`lr`: Learning rate\
--`weight_decay`: L2 regularization strength\
--`dataset`: Dataset name (must match folder name in experiment-data)\
--`dim`: Embedding dimension\
--`fea_dim`: Feature embedding dimension\
--`batch_size`: Training batch size\
--`dropout`: Dropout rate\
--`k`: Cross-validation fold\
--`output_dir`: Directory to save model checkpoints\
--`model_type`: Model operation mode ('pt' for pre-train, 'ft' for fine-tune, 'inf' for inference)

## Model Architecture
CASGT consists of several components:

1. **CentralityAwareEncoder**: Encodes node features while considering centrality metrics
2. **SignedAttention**: Multi-head attention mechanism that accounts for edge signs
3. **GraphTransformer**: Core transformer-based architecture for node embedding
4. **GraphTransformerWithClassificationHead**: Extends GraphTransformer with a classification layer for link prediction

## Example
Here's a minimal example to run the full pipeline on the Bitcoin Alpha dataset:
```bash
# Pre-train the model
python main.py --dataset bitcoin_alpha --model_type pt --epochs 50 --lr 0.001 --weight_decay 0.0001 --dim 16 --fea_dim 20 --devices cuda:0

# Fine-tune for link prediction
python main.py --dataset bitcoin_alpha --model_type ft --epochs 20 --lr 0.0005 --weight_decay 0.0001 --dim 16 --fea_dim 20 --devices cuda:0

# Run inference on test data
python main.py --dataset bitcoin_alpha --model_type inf --devices cuda:0
```

## Results
## Citation
##License        

The implementation builds upon ideas from several graph neural network architectures
Thanks to the authors of the network datasets used for evaluation
