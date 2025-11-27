# Imports
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):    
    """
    Super simple transformer encoder
    """
    def __init__(self, d_model, n_heads, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
    def forward(self, x, mask=None):
        return self.encoder(x,src_key_padding_mask=mask)
    
class JetEmbedder(nn.Module):
    
    def __init__(self, in_features_jets, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_jets, d_model)
    def forward(self, jets, jets_mask):
        embedded_jets = self.linear(jets)
        mask = jets_mask.unsqueeze(-1)
        embedded_jets = embedded_jets.masked_fill(mask, 0.0)
        return embedded_jets
    
class LeptonEmbedder(nn.Module):
    def __init__(self, in_features_leptons, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_leptons, d_model)
    def forward(self, leptons, leptons_mask):
        embedded_leptons = self.linear(leptons)
        mask = leptons_mask.unsqueeze(-1)
        embedded_leptons = embedded_leptons.masked_fill(mask, 0.0)
        return embedded_leptons

class TransformerRegressor(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets):
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers)
        self.regressor = nn.Linear(embed_dim, 1)

        self.last_embedder_output = None
    def forward(self, x, padding_mask=None, store_embedder_output=False):
        if padding_mask is None:
            padding_mask = (x == -99.0).all(dim=-1)
        
        leptons = x[:, :self.lepton_mask_size, :]
        jets = x[:, self.lepton_mask_size:, :]
        lepton_mask = padding_mask[:, :self.lepton_mask_size]
        jet_mask = padding_mask[:, self.lepton_mask_size:]
        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)
        embedded = torch.cat((lepton_embedded, jet_embedded), dim=1)
        if store_embedder_output:
            self.last_embedder_output = embedded.detach()
        x = self.transformer(embedded, mask = padding_mask)
        valid = (~padding_mask).float()  # (batch, tokens)
        x = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        out = self.regressor(x)
        return out.squeeze(-1)
