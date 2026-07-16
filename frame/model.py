
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_max_pool as gmp
from torch_geometric.utils import to_undirected, dense_to_sparse
from pathlib import Path


# ==============================================================================
# 1. Standard
# ==============================================================================

class DNN(nn.Module):
    def __init__(self, layers, dropout=0.2):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList([
                                             nn.Linear(layer[0], layer[1]) for layer in
                                             list(zip(layers[:-1], layers[1:]))
                                         ] + [
                                             nn.BatchNorm1d(layer[1]) for layer in list(zip(layers[:-1], layers[1:]))
                                         ])
        self.dropout = nn.Dropout(p=dropout)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        step = int(len(self.dnn_network) / 2)
        for i in range(step):
            linear = self.dnn_network[i]
            batchnorm = self.dnn_network[i + step]
            x = self.dropout(x)
            x = linear(x)
            if x.size(0) > 1:
                x = batchnorm(x)
            x = self.activate(x)
        x = self.dropout(x)
        return x


class GlobalAttentionBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GlobalAttentionBranch, self).__init__()
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.attn_conv1 = TransformerConv(input_dim, hidden_dim, heads=2)
        self.attn_conv2 = TransformerConv(hidden_dim * 2, output_dim, heads=2)
        self.pool = gmp

    def forward(self, x, batch):
        ptr = torch.ops.torch_sparse.ind2ptr(batch, batch.max().item() + 1)
        fc_edges = []
        for i in range(len(ptr) - 1):
            start_node_idx, end_node_idx = ptr[i], ptr[i + 1]
            num_nodes_in_graph = end_node_idx - start_node_idx
            adj = torch.ones((num_nodes_in_graph, num_nodes_in_graph), device=x.device)
            adj.fill_diagonal_(0)
            edge, _ = dense_to_sparse(adj)
            fc_edges.append(edge + start_node_idx)
        full_edge_index = to_undirected(torch.cat(fc_edges, dim=1))
        h_attn = self.activate(self.attn_conv1(x, full_edge_index))
        h_attn = self.dropout(h_attn)
        h_attn = self.activate(self.attn_conv2(h_attn, full_edge_index))
        graph_rep = self.pool(h_attn, batch)
        return graph_rep


# ==============================================================================
# 2.  Expert Core
# ==============================================================================

class ParameterGeneratorUnbounded(nn.Module):
    def __init__(self, context_dim, output_dim):
        super(ParameterGeneratorUnbounded, self).__init__()
        self.network = nn.Sequential(nn.Linear(context_dim, output_dim))

    def forward(self, context):
        return self.network(context)


class DecoupledNormalizedSiameseExpert(nn.Module):
    def __init__(self, context_dim, score_dim, output_dim, dropout=0.2):
        super(DecoupledNormalizedSiameseExpert, self).__init__()
        self.pure_linear = nn.Linear(context_dim, output_dim)
        self.mod_linear = nn.Linear(context_dim, output_dim)
        self.mod_norm = nn.LayerNorm(output_dim)
        self.soul_network = nn.Sequential(
            nn.Linear(score_dim, output_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, context, scores):
        h_pure = self.pure_linear(context)
        h_mod = self.mod_norm(self.mod_linear(context))
        soul = self.soul_network(scores)
        k_native = h_pure + self.alpha * (h_mod * soul)
        return k_native


# ==============================================================================
# 3. Pure Interaction
# ==============================================================================

class PureContextAwareTripletInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super(PureContextAwareTripletInteraction, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Weights
        self.interaction_score_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(self.head_dim, self.head_dim)) for _ in range(num_heads)])
        self.independent_projection_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(3, self.head_dim, self.head_dim)) for _ in range(num_heads)])
        self.static_kernel_dd = nn.ParameterList(
            [nn.Parameter(torch.randn(self.head_dim)) for _ in range(num_heads)])
        self.static_kernel_dc = nn.ParameterList(
            [nn.Parameter(torch.randn(self.head_dim)) for _ in range(num_heads)])
        self.static_kernel_self = nn.ParameterList(
            [nn.Parameter(torch.randn(self.head_dim)) for _ in range(num_heads)])
        self.output_projection_weights = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))

        # Experts
        context_dim = embed_dim
        self_native_ctx_dim = embed_dim + context_dim
        dd_native_ctx_dim = embed_dim * 2 + context_dim
        dc_native_ctx_dim = embed_dim * 3 + context_dim
        inter_ctx_dim = num_heads * 9

        self.kernel_gen_dd_native = DecoupledNormalizedSiameseExpert(dd_native_ctx_dim, inter_ctx_dim, self.head_dim,
                                                                     dropout)
        self.kernel_gen_dc_native = DecoupledNormalizedSiameseExpert(dc_native_ctx_dim, inter_ctx_dim, self.head_dim,
                                                                     dropout)
        self.kernel_gen_self_native = DecoupledNormalizedSiameseExpert(self_native_ctx_dim, inter_ctx_dim,
                                                                       self.head_dim, dropout)
        self.kernel_gen_inter = ParameterGeneratorUnbounded(inter_ctx_dim, self.head_dim)

        # Gates
        self.gate_gen_dd_inter = nn.Sequential(nn.Linear(dd_native_ctx_dim, self.head_dim), nn.Sigmoid())
        self.gate_gen_dc_inter = nn.Sequential(nn.Linear(dc_native_ctx_dim, self.head_dim), nn.Sigmoid())
        self.gate_gen_self_inter = nn.Sequential(nn.Linear(self_native_ctx_dim, self.head_dim), nn.Sigmoid())

        self.gate_gen_dd_final = nn.Sequential(nn.Linear(dd_native_ctx_dim, self.head_dim), nn.Sigmoid())
        self.gate_gen_dc_final = nn.Sequential(nn.Linear(dc_native_ctx_dim, self.head_dim), nn.Sigmoid())
        self.gate_gen_self_final = nn.Sequential(nn.Linear(self_native_ctx_dim, self.head_dim), nn.Sigmoid())

        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim))


        self.layer_norm = nn.LayerNorm(embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'generator' not in name and 'gate_gen' not in name and 'alpha' not in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
        with torch.no_grad():
            self.kernel_gen_dd_native.mod_linear.weight.copy_(self.kernel_gen_dd_native.pure_linear.weight)
            self.kernel_gen_dd_native.mod_linear.bias.copy_(self.kernel_gen_dd_native.pure_linear.bias)
            self.kernel_gen_dc_native.mod_linear.weight.copy_(self.kernel_gen_dc_native.pure_linear.weight)
            self.kernel_gen_dc_native.mod_linear.bias.copy_(self.kernel_gen_dc_native.pure_linear.bias)
            self.kernel_gen_self_native.mod_linear.weight.copy_(self.kernel_gen_self_native.pure_linear.weight)
            self.kernel_gen_self_native.mod_linear.bias.copy_(self.kernel_gen_self_native.pure_linear.bias)

    def forward(self, triplet_features, context):
        batch_size, num_features, embed_dim = triplet_features.size()
        triplet_heads = triplet_features.view(batch_size, num_features, self.num_heads, self.head_dim).permute(2, 0, 1,
                                                                                                               3)

        d1, d2, c = triplet_features[:, 0], triplet_features[:, 1], triplet_features[:, 2]
        mean_features_with_context = (d1 + d2 + c) / 3.0
        inp_native_self = torch.cat([mean_features_with_context, context], dim=-1)
        inp_native_dd = torch.cat([d1, d2, context], dim=-1)
        inp_native_dc = torch.cat([d1, d2, c, context], dim=-1)

        head_outputs = []
        for i in range(self.num_heads):
            current_head_features = triplet_heads[i]
            raw_interaction_scores = torch.einsum('bid,dh,bjh->bij', current_head_features,
                                                  self.interaction_score_weights[i], current_head_features)
            flat_scores = raw_interaction_scores.reshape(batch_size, -1)

            B_global = self.kernel_gen_inter(flat_scores)
            k_dd_native = self.kernel_gen_dd_native(inp_native_dd, flat_scores)
            k_dc_native = self.kernel_gen_dc_native(inp_native_dc, flat_scores)
            k_self_native = self.kernel_gen_self_native(inp_native_self, flat_scores)

            w_static_dd = self.static_kernel_dd[i]
            w_static_dc = self.static_kernel_dc[i]
            w_static_self = self.static_kernel_self[i]

            g_dd_inter = self.gate_gen_dd_inter(inp_native_dd)
            delta_dd = (g_dd_inter * k_dd_native) + ((1 - g_dd_inter) * B_global)
            g_dd_final = self.gate_gen_dd_final(inp_native_dd)
            final_kernel_dd = (g_dd_final * delta_dd) + ((1 - g_dd_final) * w_static_dd.unsqueeze(0))

            g_dc_inter = self.gate_gen_dc_inter(inp_native_dc)
            delta_dc = (g_dc_inter * k_dc_native) + ((1 - g_dc_inter) * B_global)
            g_dc_final = self.gate_gen_dc_final(inp_native_dc)
            final_kernel_dc = (g_dc_final * delta_dc) + ((1 - g_dc_final) * w_static_dc.unsqueeze(0))

            g_self_inter = self.gate_gen_self_inter(inp_native_self)
            delta_self = (g_self_inter * k_self_native) + ((1 - g_self_inter) * B_global)
            g_self_final = self.gate_gen_self_final(inp_native_self)
            final_kernel_self = (g_self_final * delta_self) + ((1 - g_self_final) * w_static_self.unsqueeze(0))

            self_scores = torch.diagonal(raw_interaction_scores, dim1=-2, dim2=-1)
            self_interaction = torch.einsum('bn,bd->bnd', self_scores, final_kernel_self)

            score_dd = raw_interaction_scores[:, 0, 1]
            l_p_dd_term = torch.einsum('b,bd->bd', score_dd, final_kernel_dd)
            drug_drug_interaction = torch.zeros_like(current_head_features)
            drug_drug_interaction[:, 0, :] += l_p_dd_term
            drug_drug_interaction[:, 1, :] += l_p_dd_term

            score_dc1 = raw_interaction_scores[:, 0, 2]
            score_dc2 = raw_interaction_scores[:, 1, 2]
            l_p_dc_term1 = torch.einsum('b,bd->bd', score_dc1, final_kernel_dc)
            l_p_dc_term2 = torch.einsum('b,bd->bd', score_dc2, final_kernel_dc)
            drug_cell_interaction = torch.zeros_like(current_head_features)
            drug_cell_interaction[:, 0, :] += l_p_dc_term1
            drug_cell_interaction[:, 2, :] += l_p_dc_term1
            drug_cell_interaction[:, 1, :] += l_p_dc_term2
            drug_cell_interaction[:, 2, :] += l_p_dc_term2

            independent_projection = torch.einsum('bnd,ndh->bnh', current_head_features,
                                                  self.independent_projection_weights[i])
            pairwise_interaction_features = self_interaction + drug_drug_interaction + drug_cell_interaction
            fused_head_output = torch.matmul(independent_projection + pairwise_interaction_features,
                                             self.output_projection_weights[i])
            head_outputs.append(fused_head_output)

        combined_heads_output = torch.cat(head_outputs, dim=-1)
        mean_head_features = torch.mean(combined_heads_output, dim=1, keepdim=True).expand(-1, num_features, -1)
        gate_input = torch.cat([combined_heads_output, mean_head_features], dim=-1)
        gate = self.gate(gate_input)
        gated_interaction_output = gate * combined_heads_output + (1 - gate) * mean_head_features
        projected_output = self.output_linear(gated_interaction_output)

        pure_inter_output = self.layer_norm(projected_output)
        return pure_inter_output


class DecoupledStaticRawDynamicInterFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.rawnorm = nn.LayerNorm(embed_dim)
        # ============================================================
        # 1. Static Raw Reinforcement
        # ============================================================

        self.static_alpha_drug = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.static_alpha_cell = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.static_alpha_global = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        # ============================================================
        # 2.  Gate & Shift
        # ============================================================
        self.dynamic_gate_network = nn.Sequential(
            nn.Linear(embed_dim , embed_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim // 2, embed_dim * 2),
            nn.Tanh()  # Shared Tanh control ->  [-1, 1]
        )

        # Shift
        self.shift_gain = nn.Parameter(torch.zeros(1))

        # ============================================================
        # 3. Static Scaling Vector
        # ============================================================

        self.static_scale_drug = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.static_scale_cell = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.static_scale_global = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.shift_gain2 = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        # 1. Alpha
        nn.init.constant_(self.static_alpha_drug, 0.1)
        nn.init.constant_(self.static_alpha_cell, 0.1)
        nn.init.constant_(self.static_alpha_global, 0.1)

        # 2.
        for m in self.dynamic_gate_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        #  Gate/Shift
        #  Gate=0, Shift=0 -> Patch=0
        last_layer = self.dynamic_gate_network[-2]
        if isinstance(last_layer, nn.Linear):
            nn.init.constant_(last_layer.weight, 0)
            nn.init.constant_(last_layer.bias, 0)

        # 3. Scale
        nn.init.constant_(self.static_scale_drug, 0.1)
        nn.init.constant_(self.static_scale_cell, 0.1)
        nn.init.constant_(self.static_scale_global, 0.0)

        # 4. Shift Gain
        nn.init.constant_(self.shift_gain, 0.0)
        nn.init.constant_(self.shift_gain2, 0.01)
    def forward(self, raw, pure_inter):
        # raw, pure_inter shape: [Batch, 3, Dim]
        # Index 0: Drug1, Index 1: Drug2, Index 2: Cell

        # --- A. Static Base---
        # Drug/Cell
        # Drug: [Batch, 2, Dim]
        alpha_drug = self.static_alpha_drug.expand(raw.size(0), 2, -1)
        # Cell: [Batch, 1, Dim]
        alpha_cell = self.static_alpha_cell.expand(raw.size(0), 1, -1)

        # [Batch, 3, Dim]
        alpha_static = torch.cat([alpha_drug, alpha_cell], dim=1)

        # Base Term: (1 + alpha) * Raw
        term_base = (1.0 + alpha_static) * raw

        # --- B. Static Scaled + Dynamic Shift Patch ---

        combined = pure_inter + raw

        combined = self.dropout(combined)

        gate_out = self.dynamic_gate_network(combined)
        gate, dynamic_shift_raw = torch.chunk(gate_out, 2, dim=-1)
        #gate = torch.tanh(gate)
        #dynamic_shift_raw= torch.tanh(dynamic_shift_raw)
        # Shift Gain
        # dynamic_shift_raw
        dynamic_shift = dynamic_shift_raw * self.shift_gain2

        # 3. Static Scale

        scale_drug_param = self.static_scale_drug.expand(raw.size(0), 2, -1)
        scale_cell_param = self.static_scale_cell.expand(raw.size(0), 1, -1)

        # [Batch, 3, Dim]
        scale_combined_param = torch.cat([scale_drug_param, scale_cell_param], dim=1)

        # Tanh(param) -> [-1, 1]
        static_scale = torch.tanh(scale_combined_param)


        # Inter * (1 + Scale) + Shift
        inter_transformed = pure_inter * (1.0 + static_scale)

        #  Patch: Gate * Inter_Transformed + Dynamic_Shift
        term_patch = gate * inter_transformed + dynamic_shift

        final = term_base + term_patch

        return final

# ==============================================================================
# 5. 主模型
# ==============================================================================

class SynergyPredictionModel(torch.nn.Module):
    def __init__(self,  num_features_xd=78, num_features_xt=954, output_dim=128,
                 dropout=0.1, use_auxiliary=False,
                 top_k: int = 128,
                 query_dropout_p: float = 0.1):
        super(SynergyPredictionModel, self).__init__()

        self.use_auxiliary = use_auxiliary
        self.query_dropout_p = query_dropout_p
        self.top_k = top_k
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # Feature Extraction
        self.drug_conv1 = TransformerConv(78, num_features_xd * 2, heads=2)
        self.drug_conv2 = TransformerConv(num_features_xd * 4, num_features_xd * 4, heads=2)
        self.global_attention_branch = GlobalAttentionBranch(input_dim=78, hidden_dim=num_features_xd * 2,
                                                             output_dim=num_features_xd * 4)
        self.gate_network = nn.Sequential(nn.Linear(num_features_xd * 8, output_dim), nn.LeakyReLU(),
                                          nn.Linear(output_dim, 1), nn.Sigmoid())
        self.fusion_transform = nn.Sequential(nn.Linear(num_features_xd * 8, output_dim), nn.LeakyReLU(),
                                              nn.Dropout(dropout))
        self.reduction = nn.Sequential(nn.Linear(num_features_xt, 512), nn.LeakyReLU(), nn.Linear(512, 256),
                                       nn.LeakyReLU(), nn.Linear(256, output_dim))

        # Interaction Module
        self.interaction_module = PureContextAwareTripletInteraction(embed_dim=output_dim, num_heads=1, dropout=dropout)

        # Fusion Module
        self.fusion_module = DecoupledStaticRawDynamicInterFusion(embed_dim=output_dim)

        # PPI
        try:
            current_dir = Path(__file__).parent.resolve()
            protein_embeddings_path = current_dir / 'protein_embeddings.pt'
        except NameError:
            protein_embeddings_path = Path('./protein_embeddings.pt')

        if protein_embeddings_path.exists():
            print(f"  > Loading PPI embeddings from: {protein_embeddings_path}")
            protein_embeddings = torch.load(protein_embeddings_path)
            self.register_buffer('protein_embeddings', protein_embeddings)
            protein_feat_dim = self.protein_embeddings.shape[1]
            self.protein_projection = nn.Linear(protein_feat_dim, output_dim)
        else:
            self.protein_projection = None

        self.unified_bilinear_scorer = nn.Linear(output_dim, output_dim)

        final_dnn_input_dim = output_dim * 3
        hidden_units = [final_dnn_input_dim, 1024, 256]
        self.dnn_network = DNN(hidden_units, 0.2)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
        if self.use_auxiliary:
            self.auxiliary_net = nn.Sequential(
                nn.Linear(final_dnn_input_dim, 128), nn.LeakyReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1)
            )

    def process_drug(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        gnn_hidden = self.activate(self.drug_conv1(x, edge_index))
        gnn_hidden = self.activate(self.drug_conv2(gnn_hidden, edge_index))
        gnn_graph_rep = gmp(gnn_hidden, batch)
        global_graph_rep = self.global_attention_branch(x, batch)
        features_for_gate = gnn_graph_rep + global_graph_rep
        fusion_gate = self.gate_network(features_for_gate)
        gated_gnn_rep = fusion_gate * gnn_graph_rep
        gated_global_rep = (1 - fusion_gate) * global_graph_rep
        fused_drug_rep = gated_gnn_rep + gated_global_rep
        final_drug_rep = self.fusion_transform(fused_drug_rep)
        return final_drug_rep

    def get_ppi_vector_bilinear(self, query, projected_proteins, scorer):
        transformed_query = scorer(query)
        relevance_scores = torch.matmul(transformed_query, projected_proteins.T)
        k = min(self.top_k, projected_proteins.size(0))
        top_k_scores, top_k_indices = torch.topk(relevance_scores, k, dim=1)
        top_k_features = projected_proteins[top_k_indices]
        attention_weights = F.softmax(top_k_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), top_k_features).squeeze(1)
        return context_vector

    def forward(self, data1, data2):
        batch_size = data1.num_graphs
        x1 = self.process_drug(data1)
        x2 = self.process_drug(data2)
        cell_vector = self.reduction(F.normalize(data1.cell, 2, 1))

        # 1. Raw
        triplet_features = torch.stack([x1, x2, cell_vector], dim=1)

        if self.protein_projection is not None:
            if self.query_dropout_p > 0 and self.training:
                query_features = F.dropout(triplet_features, p=self.query_dropout_p, training=self.training)
                ppi_query = torch.mean(query_features, dim=1)
            else:
                ppi_query = torch.mean(triplet_features, dim=1)
            projected_proteins = self.protein_projection(self.protein_embeddings)
            ppi_context = self.get_ppi_vector_bilinear(
                ppi_query, projected_proteins, self.unified_bilinear_scorer
            )
        else:
            ppi_context = torch.zeros(batch_size, self.interaction_module.embed_dim, device=x1.device)

        # 2. Pure Inter
        inter_pure = self.interaction_module(triplet_features, ppi_context)

        # 3. Fusion
        final_features = self.fusion_module(triplet_features, inter_pure)

        flat_features = final_features.view(batch_size, -1)
        prediction_hidden = self.dnn_network(flat_features)
        logit = self.dense_final(prediction_hidden)
        prediction = torch.sigmoid(logit.squeeze(1))

        if self.training and self.use_auxiliary:
            # auxiliary_output = self.auxiliary_net(flat_features)
            return prediction
        else:
            return prediction