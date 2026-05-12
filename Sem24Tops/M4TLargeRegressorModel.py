
# Imports
import torch
import torch.nn as nn
class TransformerEncoder(nn.Module):    
    """
    Super simple transformer encoder
    """
    def __init__(self, d_model, n_heads, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward = 1024, batch_first=True)
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
    
class NumbersEmbedder(nn.Module):
    def __init__(self, in_features_numbers, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_numbers, d_model)
    def forward(self, numbers):
        embedded_numbers = self.linear(numbers)
        return embedded_numbers  # (batch, 1, d_model)
    
class HTEmbedder(nn.Module):
    def __init__(self, in_features_ht, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features_ht, d_model)
    def forward(self, ht_features):
        embedded_ht = self.linear(ht_features)
        return embedded_ht  # (batch, 1, d_model)
    
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
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht):
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_numbers = in_features_numbers
        self.in_features_ht = in_features_ht
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        self.met_embedder = METEmbedder(in_features_met, embed_dim)
        self.numbers_embedder = NumbersEmbedder(in_features_numbers, embed_dim)
        self.ht_embedder = HTEmbedder(in_features_ht, embed_dim)
        # self.embedder = Embedder(in_features_leptons, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers)
        self.regressor_mass = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1))
        self.regressor_ht = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1))
        self.last_embedder_output = None
    def forward(self, x, padding_mask=None, store_embedder_output=False):
        if padding_mask is None:
            padding_mask = (x == -99.0).all(dim=-1)
        
        leptons = x[:, :self.lepton_mask_size, :self.in_features_leptons]
        jets = x[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size, :self.in_features_jets]
        met = x[:, self.lepton_mask_size + self.jet_mask_size:self.lepton_mask_size + self.jet_mask_size + 1, :self.in_features_met]
        numbers = x[:, self.lepton_mask_size + self.jet_mask_size + 1:self.lepton_mask_size + self.jet_mask_size + 2, :self.in_features_numbers]
        ht_features = x[:, self.lepton_mask_size + self.jet_mask_size + 2:self.lepton_mask_size + self.jet_mask_size + 3, :self.in_features_ht]
        lepton_mask = padding_mask[:, :self.lepton_mask_size]
        jet_mask = padding_mask[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size]
        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)
        met_embedded = self.met_embedder(met)
        numbers_embedded = self.numbers_embedder(numbers)
        ht_embedded = self.ht_embedder(ht_features)
        embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded, numbers_embedded, ht_embedded), dim=1)
        # embedded = self.embedder(x, padding_mask)
        if store_embedder_output:
            self.last_embedder_output = embedded.detach()
        x = self.transformer(embedded, mask = padding_mask)
        valid = (~padding_mask).float()  # (batch, tokens)
        x = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        out_mass = self.regressor_mass(x)
        out_ht = self.regressor_ht(x)
        out = torch.cat((out_mass, out_ht), dim=1) # (batch, 2)
        return out
