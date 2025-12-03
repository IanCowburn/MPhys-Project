# Imports
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Module consisting of multiple layers.
    """
    def __init__(self, d_model, n_heads, num_layers):
        """
        Initialize the Transformer Encoder.

        Args:
            d_model (int): The number of expected features in the input.
            n_heads (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.

        Returns:
            None
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Padding mask for the input.

        Returns:
            torch.Tensor: Output of the Transformer Encoder.
        """
        return self.encoder(x,src_key_padding_mask=mask)
    
class JetEmbedder(nn.Module):
    """
    Jet Embedder Module to embed jet features.
    """
    def __init__(self, in_features_jets, d_model):
        """
        Initialize the Jet Embedder.

        Args:
            in_features_jets (int): Number of input features for jets.
            d_model (int): Dimension of the embedding space.

        Returns:
            None
        """
        super().__init__()
        self.linear = nn.Linear(in_features_jets, d_model)

    def forward(self, jets, jets_mask):
        """
        Forward pass through the Jet Embedder.

        Args:
            jets (torch.Tensor): Input tensor for jets.
            jets_mask (torch.Tensor): Padding mask for the jets.

        Returns:
            torch.Tensor: Embedded jet features.
        """
        embedded_jets = self.linear(jets)
        mask = jets_mask.unsqueeze(-1)
        embedded_jets = embedded_jets.masked_fill(mask, 0.0)
        return embedded_jets
    
class LeptonEmbedder(nn.Module):
    """
    Lepton Embedder Module to embed lepton features.
    """
    def __init__(self, in_features_leptons, d_model):
        """
        Initialize the Lepton Embedder.

        Args:
            in_features_leptons (int): Number of input features for leptons.
            d_model (int): Dimension of the embedding space.

        Returns:
            None
        """
        super().__init__()
        self.linear = nn.Linear(in_features_leptons, d_model)

    def forward(self, leptons, leptons_mask):
        """
        Forward pass through the Lepton Embedder.

        Args:
            leptons (torch.Tensor): Input tensor for leptons.
            leptons_mask (torch.Tensor): Padding mask for the leptons.

        Returns:
            torch.Tensor: Embedded lepton features.
        """
        embedded_leptons = self.linear(leptons)
        mask = leptons_mask.unsqueeze(-1)
        embedded_leptons = embedded_leptons.masked_fill(mask, 0.0)
        return embedded_leptons
    
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
    """
    Transformer Regressor Module for processing embedded features and performing regression.
    """
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets):
        """
        Initialize the Transformer Regressor.

        Args:
            embed_dim (int): Dimension of the embedding space.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            lepton_mask_size (int): Maximum number of leptons allowed per event.
            jet_mask_size (int): Maximum number of jets allowed per event.
            in_features_leptons (int): Number of input features for leptons.
            in_features_jets (int): Number of input features for jets.

        Returns:
            None
        """
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        # self.embedder = Embedder(in_features_leptons, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers)
        self.regressor = nn.Linear(embed_dim, 2)
        self.last_embedder_output = None

    def forward(self, x, padding_mask=None, store_embedder_output=False):
        """
        Forward pass through the Transformer Regressor.

        Args:
            x (torch.Tensor): Input tensor containing lepton and jet features.
            padding_mask (torch.Tensor, optional): Padding mask for the input tensor. Defaults to None.
            store_embedder_output (bool, optional): Flag to store the output of the embedder. Defaults to False.

        Returns:
            torch.Tensor: Output tensor after regression.
        """
        if padding_mask is None:
            padding_mask = (x == -99.0).all(dim=-1)
        
        leptons = x[:, :self.lepton_mask_size, :]
        jets = x[:, self.lepton_mask_size:, :]

        lepton_mask = padding_mask[:, :self.lepton_mask_size]
        jet_mask = padding_mask[:, self.lepton_mask_size:]

        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)

        embedded = torch.cat((lepton_embedded, jet_embedded), dim=1)

        # embedded = self.embedder(x, padding_mask)

        if store_embedder_output:
            self.last_embedder_output = embedded.detach()

        x = self.transformer(embedded, mask = padding_mask)

        valid = (~padding_mask).float()  # (batch, tokens)

        x = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        
        out = self.regressor(x)
        return out
