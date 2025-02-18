import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
    "PReLU": nn.PReLU(),
    "RReLU": nn.RReLU(lower=0.1, upper=0.3),
    "ELU": nn.ELU(alpha=1.0),
    "SELU": nn.SELU(),
    "GELU": nn.GELU(),
    "SiLU (Swish)": nn.SiLU(),
    "Mish": nn.Mish(),
}

class ExULayer(nn.Module):
    """
    Implements the exp-centred (ExU) layer:
      ExU(x) = LeakyReLU((x - b) * exp(W))
    where b and W are learnable parameters.
    """
    def __init__(self, in_features, out_features, negative_slope=0.01):
        super(ExULayer, self).__init__()
        self.negative_slope = negative_slope
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, a=self.negative_slope, nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # x shape: (batch, in_features) where in_features == 1 typically.
        # Expand x to (batch, out_features)
        x_expanded = x.expand(-1, self.weight.size(0))
        transformed = (x_expanded - self.bias) * torch.exp(self.weight.squeeze(1))
        return F.leaky_relu(transformed, negative_slope=self.negative_slope)

class FeatureNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list, dropout_rate: float = 0.3):
        """
        Processes a single input feature.
        The first layer is an ExU layer. Subsequent layers are standard dense layers:
          LeakyReLU((X - b)W)
        """
        super(FeatureNetwork, self).__init__()
        # First layer: ExU layer
        first_hidden = hidden_units[0]
        self.exu = ExULayer(input_dim, first_hidden, negative_slope=0.01)
        
        # Build subsequent hidden layers.
        layers = []
        in_features = first_hidden
        for out_features in hidden_units[1:]:
            linear = nn.Linear(in_features, out_features)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='leaky_relu')
            layers.append(linear)
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(activations["LeakyReLU"])
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        
        # Final output layer produces a single scalar.
        linear = nn.Linear(in_features, 1)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='linear')
        layers.append(linear)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        out = self.exu(x)
        return self.mlp(out)

class CoxNAM(nn.Module):
    def __init__(self, num_features: int, input_dim: int, hidden_units: list, dropout_rate: float = 0.3):
        """
        CoxNAM consists of one FeatureNetwork per input feature.
        Each network processes a single feature and outputs a scalar.
        These outputs are then concatenated and combined using a final linear layer
        (weighted summation) to produce the risk score.
        """
        super(CoxNAM, self).__init__()
        self.num_features = num_features
        self.feature_networks = nn.ModuleList([
            FeatureNetwork(input_dim, hidden_units, dropout_rate) for _ in range(num_features)
        ])
        # Final linear layer to perform weighted summation.
        # No bias is used so that the final output remains scaled by the learned weights.
        self.final_linear = nn.Linear(num_features, 1, bias=False)
        
    def forward(self, x):
        # x is expected to have shape: (batch, num_features, input_dim)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
            
        contributions = []
        for i, network in enumerate(self.feature_networks):
            # Each feature network processes its corresponding feature.
            # Output shape: (batch, 1)
            contribution = network(x[:, i, :])
            contributions.append(contribution)
        
        # Concatenate contributions to shape (batch, num_features)
        contributions_cat = torch.cat(contributions, dim=1)
        # Compute weighted summation using the final linear layer.
        risk_scores = self.final_linear(contributions_cat)
        return risk_scores.squeeze(-1)


