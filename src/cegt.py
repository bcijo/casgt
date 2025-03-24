import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the Focal Loss.

        Args:
        - logits: Model predictions before sigmoid (logits).
        - targets: Ground truth labels (0 or 1).

        Returns:
        - Focal loss value.
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # Probabilities of correctly classified
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    
class SignedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(SignedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.sqrt_d = math.sqrt(embed_dim)
        self.query_layer = nn.Linear(embed_dim, embed_dim * num_heads)
        self.key_layer = nn.Linear(embed_dim, embed_dim * num_heads)
        self.value_layer = nn.Linear(embed_dim, embed_dim * num_heads)
        self.attn_weight_proj = nn.Linear(embed_dim * num_heads, embed_dim)
        self.sign_weight = nn.Parameter(torch.Tensor(num_heads, 1))
        nn.init.xavier_uniform_(self.sign_weight)

    def forward(self, node_embeddings, node_sign_influence, adj_matrix):
        num_nodes = node_embeddings.size(0)
        Q = self.query_layer(node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
        K = self.key_layer(node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
        V = self.value_layer(node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
        adj_matrix = adj_matrix.to_dense()
        edge_index = (adj_matrix > 0).nonzero(as_tuple=False)
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        Q_edges = Q[src]
        K_edges = K[tgt]
        scores = (Q_edges * K_edges).sum(dim=-1) / self.sqrt_d
        sign_factor = node_sign_influence[src].unsqueeze(1)
        scores = scores * sign_factor
        attention_weights = torch.empty_like(scores)
        for h in range(self.num_heads):
            scores_h = scores[:, h]
            unique_src, inverse_indices = torch.unique(src, return_inverse=True)
            max_per_src = torch.zeros(unique_src.size(0), device=scores_h.device).scatter_reduce(0, inverse_indices, scores_h, reduce="amax", include_self=False)
            exp_scores = torch.exp(scores_h - max_per_src[inverse_indices])
            sum_exp = torch.zeros(unique_src.size(0), device=scores_h.device).scatter_add_(0, inverse_indices, exp_scores)
            attention_weights[:, h] = exp_scores / (sum_exp[inverse_indices] + 1e-10)
        attention_weights = self.dropout(attention_weights)
        V_edges = V[tgt]
        weighted_V = V_edges * attention_weights.unsqueeze(-1)
        out_per_head = torch.zeros(self.num_heads, num_nodes, self.embed_dim, device=node_embeddings.device)
        for h in range(self.num_heads):
            out_per_head[h].index_add_(0, src, weighted_V[:, h, :])
        out = out_per_head.transpose(0, 1).reshape(num_nodes, -1)
        out = self.attn_weight_proj(out)
        return out

class CentralityAwareEncoder(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, adj_lists, aggs, device):
        super(CentralityAwareEncoder, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.centrality_dim = centrality_dim
        self.adj_lists = adj_lists
        self.aggs = aggs
        self.device = device
        self.feature_combiner = nn.Linear(self.node_feat_dim, embed_dim)
        self.centrality_encoder = nn.Linear(centrality_dim, embed_dim)
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(self.device)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.feature_combiner.apply(init_weights)
        self.centrality_encoder.apply(init_weights)
    
    def forward(self, nodes, node_features, centrality_features):
        # Instead of creating a new LongTensor and moving it, just ensure it's on the right device
        nodes_tensor = nodes if isinstance(nodes, torch.Tensor) else torch.LongTensor(nodes)
        
        # Make sure nodes_tensor is on the same device as the model
        if nodes_tensor.device != self.device:
            nodes_tensor = nodes_tensor.to(self.device)
        
        # Now use the tensor with node_features
        self_feats = node_features(nodes_tensor)
        
        # Rest of the function remains the same
        betweenness = centrality_features["betweenness"][nodes]
        closeness = centrality_features["closeness"][nodes]
        central_feats = torch.stack([betweenness, closeness], dim=1).to(self.device)
        central_feats_encoded = self.centrality_encoder(central_feats)
        combined_feats = self.feature_combiner(self_feats) + central_feats_encoded
        return combined_feats

class GraphTransformer(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, adj_lists, aggs, device, dropout_rate=0.1):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.encoder = CentralityAwareEncoder(node_feat_dim, embed_dim, centrality_dim, adj_lists, aggs, device)
        self.transformer_layers = nn.ModuleList([
            SignedAttention(embed_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, nodes, node_features, centrality_features, adj_matrix, node_sign_influence):
        x = self.encoder(nodes, node_features, centrality_features)
        for i in range(self.num_layers):
            attn_out = self.transformer_layers[i](x, node_sign_influence, adj_matrix)
            x = self.layer_norms[i](x + attn_out)
            x = self.dropout(x)
        return x

class GraphTransformerWithClassificationHead(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, adj_lists, aggs, device, dropout_rate=0.1):
        super(GraphTransformerWithClassificationHead, self).__init__()
        self.transformer = GraphTransformer(node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, adj_lists, aggs, device, dropout_rate)
        self.linear = nn.Linear(embed_dim * 2, 1)
        
    def forward(self, nodes, node_features, centrality_features, adj_matrix, node_sign_influence, edge_list):
        node_embeddings = self.transformer(nodes, node_features, centrality_features, adj_matrix, node_sign_influence)
        edge_embeddings = []
        for edge in edge_list:
            source_emb = node_embeddings[edge[0]]
            target_emb = node_embeddings[edge[1]]
            combined_emb = torch.cat([source_emb, target_emb], dim=0)
            edge_embeddings.append(combined_emb)
        edge_embeddings = torch.stack(edge_embeddings).to(self.transformer.device)
        link_logits = self.linear(edge_embeddings)
        return link_logits.squeeze()