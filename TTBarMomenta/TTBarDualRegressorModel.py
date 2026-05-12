
# Imports
import torch
import torch.nn as nn
class TransformerEncoder(nn.Module):    
    """
    Super simple transformer encoder
    """
    def __init__(self, d_model, n_heads, num_layers, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=mask)
    
class JetEmbedder(nn.Module):
    
    def __init__(self, in_features_jets, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_jets, d_model)
    def forward(self, jets, jets_mask):
        embedded_jets = self.linear(jets)
        mask = jets_mask.unsqueeze(-1)
        embedded_jets = embedded_jets.masked_fill(mask, 0.0)
        return embedded_jets # (batch, tokens, d_model)
    
class LeptonEmbedder(nn.Module):
    def __init__(self, in_features_leptons, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_leptons, d_model)
    def forward(self, leptons, leptons_mask):
        embedded_leptons = self.linear(leptons)
        mask = leptons_mask.unsqueeze(-1)
        embedded_leptons = embedded_leptons.masked_fill(mask, 0.0)
        return embedded_leptons # (batch, tokens, d_model)
    
class METEmbedder(nn.Module):
    def __init__(self, in_features_met, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_met, d_model)
    def forward(self, met):
        embedded_met = self.linear(met)
        return embedded_met  # (batch, 1, d_model)
    
class EllipseNuEmbedder(nn.Module):
    def __init__(self, in_features_ellipse_nu, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_ellipse_nu, d_model)
    def forward(self, ellipse_nu):
        embedded_ellipse_nu = self.linear(ellipse_nu)
        return embedded_ellipse_nu  # (batch, 1, d_model)
    
class EllipseAntiNuEmbedder(nn.Module):
    def __init__(self, in_features_ellipse_anti_nu, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_ellipse_anti_nu, d_model)
    def forward(self, ellipse_anti_nu):
        embedded_ellipse_anti_nu = self.linear(ellipse_anti_nu)
        return embedded_ellipse_anti_nu  # (batch, 1, d_model)
    
# class Embedder(nn.Module):
#     def __init__(self, in_features, d_model):
#         super().__init__()
#         self.linear = nn.Linear(in_features, d_model)
#     def forward(self, x, padding_mask):
#         embedded = self.linear(x)
#         mask = padding_mask.unsqueeze(-1)
#         embedded = embedded.masked_fill(mask, 0.0)
#         return embedded

class TransformerRegressor(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu, output_dim=6):
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_ellipse_nu = in_features_ellipse_nu
        self.in_features_ellipse_anti_nu = in_features_ellipse_anti_nu
        self.has_ellipse = in_features_ellipse_nu > 0 and in_features_ellipse_anti_nu > 0
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        self.met_embedder = METEmbedder(in_features_met, embed_dim)
        if self.has_ellipse:
            self.ellipse_nu_embedder = EllipseNuEmbedder(in_features_ellipse_nu, embed_dim)
            self.ellipse_anti_nu_embedder = EllipseAntiNuEmbedder(in_features_ellipse_anti_nu, embed_dim)
        n_token_types = 5 if self.has_ellipse else 3
        self.type_embedding = nn.Embedding(n_token_types, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers, dropout=0.15)
        # Attention-weighted pooling
        self.attn_pool = nn.Linear(embed_dim, 1)
        # Separate heads for nu and antinu with deeper residual MLP
        h = embed_dim // 2
        self.antinu_proj = nn.Linear(embed_dim, h)
        self.antinu_block = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
        )
        self.antinu_out = nn.Linear(h, 3)  # antinu_px, antinu_py, antinu_pz
        self.nu_proj = nn.Linear(embed_dim, h)
        self.nu_block = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
        )
        self.nu_out = nn.Linear(h, 3)  # nu_px, nu_py, nu_pz
        self.last_embedder_output = None
    def forward(self, x, padding_mask=None, store_embedder_output=False):
        if padding_mask is None:
            padding_mask = (x == -99.0).all(dim=-1)
        
        leptons = x[:, :self.lepton_mask_size, :self.in_features_leptons]
        jets = x[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size, :self.in_features_jets]
        met = x[:, self.lepton_mask_size + self.jet_mask_size:self.lepton_mask_size + self.jet_mask_size + 1, :self.in_features_met]
        lepton_mask = padding_mask[:, :self.lepton_mask_size]
        jet_mask = padding_mask[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size]
        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)
        met_embedded = self.met_embedder(met)
        if self.has_ellipse:
            ellipse_nu = x[:, self.lepton_mask_size + self.jet_mask_size + 1:self.lepton_mask_size + self.jet_mask_size + 2, :self.in_features_ellipse_nu]
            ellipse_anti_nu = x[:, self.lepton_mask_size + self.jet_mask_size + 2:self.lepton_mask_size + self.jet_mask_size + 3, :self.in_features_ellipse_anti_nu]
            ellipse_nu_embedded = self.ellipse_nu_embedder(ellipse_nu)
            ellipse_anti_nu_embedded = self.ellipse_anti_nu_embedder(ellipse_anti_nu)
            embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded, ellipse_nu_embedded, ellipse_anti_nu_embedded), dim=1)
        else:
            embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded), dim=1)
        batch_size = x.size(0)
        if self.has_ellipse:
            type_ids = torch.cat([
                torch.zeros((batch_size, self.lepton_mask_size), dtype=torch.long, device=x.device),
                torch.ones((batch_size, self.jet_mask_size), dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 3, dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 4, dtype=torch.long, device=x.device),
            ], dim=1)
        else:
            type_ids = torch.cat([
                torch.zeros((batch_size, self.lepton_mask_size), dtype=torch.long, device=x.device),
                torch.ones((batch_size, self.jet_mask_size), dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=x.device),
            ], dim=1)
        type_embedded = self.type_embedding(type_ids)
        embedded = self.embed_dropout(embedded + type_embedded)

        if store_embedder_output:
            self.last_embedder_output = embedded.detach()

        x = self.transformer(embedded, mask=padding_mask)
        # Attention-weighted pooling
        attn_weights = self.attn_pool(x).squeeze(-1)           # (batch, tokens)
        attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)      # (batch, tokens)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)        # (batch, embed_dim)
        # Separate residual heads
        antinu_h = self.antinu_proj(x)
        antinu_h = antinu_h + self.antinu_block(antinu_h)
        antinu_pred = self.antinu_out(antinu_h)
        nu_h = self.nu_proj(x)
        nu_h = nu_h + self.nu_block(nu_h)
        nu_pred = self.nu_out(nu_h)
        out = torch.cat([antinu_pred, nu_pred], dim=1)  # (batch, 6)
        return out
