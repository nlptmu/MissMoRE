import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

import math

class ResidualCell(nn.Module):
    """
    Tokenwise residual MLP:
        y = x + FF(LN(x))
    Operates on the last dim only. Shape preserved: [B, N, D] -> [B, N, D]
    """
    def __init__(self, dim: int = 1024, hidden_mult: float = 2.0, activation=nn.GELU):
        super().__init__()
        h = int(hidden_mult * dim)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, h),
            activation(),
            nn.Linear(h, dim),
        )
        # near-identity init for stability under recursion
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.ln(x))


class MixtureOfRecursion(nn.Module):
    """
    Mixture over multiple recursion depths of the SAME residual cell.

    Inputs:
        x: [B, N, D] with D=1024 (N can be 49 or any token count)
    Args:
        dim: feature size (1024 here)
        depths: iterable of positive ints, e.g. (1, 2, 4)
        gate_hidden_mult: width multiplier for the gate MLP

    Outputs:
        y: [B, N, D] mixed output
        alphas: [B, N, K] softmax weights over K depths (tokenwise)
    """
    def __init__(
        self,
        dim: int = 1024,
        depths=(1, 2, 4),
        gate_hidden_mult: float = 2.0,
        cell: nn.Module | None = None,
    ):
        super().__init__()
        assert all(d > 0 for d in depths), "Depths must be positive."
        self.dim = dim
        self.depths = tuple(sorted(set(int(d) for d in depths)))
        self.max_depth = max(self.depths)

        # one shared residual cell reused across depths
        self.cell = cell if cell is not None else ResidualCell(dim=dim)

        # tokenwise gate -> logits over depths
        gh = int(gate_hidden_mult * dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, gh),
            nn.GELU(),
            nn.Linear(gh, len(self.depths)),
        )

    def forward(self, x: torch.Tensor, return_alphas: bool = False):
        """
        x: [B, N, D] with D == self.dim
        """
        B, N, D = x.shape
        assert D == self.dim, f"Expected last dim {self.dim}, got {D}"

        # compute all recursion states up to max depth
        cache = {0: x}
        cur = x
        for t in range(1, self.max_depth + 1):
            cur = self.cell(cur)      # shape stays [B, N, D]
            cache[t] = cur

        # stack the requested depths along a new axis K
        # Y: [B, N, K, D]
        Y = torch.stack([cache[d] for d in self.depths], dim=2)

        # tokenwise gate from the ORIGINAL input x
        # logits: [B, N, K] -> softmax over K
        logits = self.gate(x)
        alphas = F.softmax(logits, dim=-1)

        # weighted sum over depths
        # result: [B, N, D]
        y = torch.sum(alphas.unsqueeze(-1) * Y, dim=2)

        return (y, alphas) if return_alphas else y

class Tower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Tower, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.Linear1(x)
        x = self.dropout(x)
        x = F.gelu(self.Linear2(x)) 
        x = self.dropout(x)
        x = self.Classifier(x.mean(dim=1))
        return x


class MMoRE(nn.Module):
    def __init__(self, dim=1024, gate_dim=256, hidden_dim=768, num_tasks=7, num_experts=3):
        super(MMoRE, self).__init__()
        
        # Recursion Experts
        self.experts = nn.ModuleList([
            MixtureOfRecursion(dim=dim, depths=(1, 3, 5)) 
            for _ in range(num_experts)
        ])
        
        # Define the Gates (one for each task)
        self.gates = nn.ModuleList([
            nn.Linear(gate_dim, num_experts) for _ in range(num_tasks)
        ])
        
        # Define the 7 Task Towers
        self.towers = nn.ModuleList([
            Tower(input_size=dim, hidden_size=hidden_dim, output_size=1) 
            for _ in range(num_tasks)
        ])

        self.latent_compressor = nn.Linear(dim, gate_dim)

    def forward(self, feature):
        
        expert_outputs = [expert(feature) for expert in self.experts]
        stacked_experts = torch.stack(expert_outputs, dim=1)
        
        routing_context = self.latent_compressor(feature.mean(dim=1)) # [B, D]
        
        final_outputs = []
        for i in range(len(self.towers)):

            gate_weights = F.softmax(self.gates[i](routing_context), dim=-1)
            task_specific_features = torch.einsum('be, besd -> bsd', gate_weights, stacked_experts)
            
            task_output = self.towers[i](task_specific_features)
            final_outputs.append(task_output)
            
        return final_outputs

class MissMoRE(torch.nn.Module):
    def __init__(self, model, num_labels=7, num_experts=3):
        super(MissMoRE, self).__init__()

        self.model = AutoModel.from_pretrained(model, image_size = (307, 614), output_attentions = False, hidden_act = 'gelu', output_hidden_states = True, ignore_mismatched_sizes=True)
        
        MMoRE_module = MMoRE(dim=171, gate_dim=256, hidden_dim=768, num_tasks=num_labels, num_experts=num_experts) # input dim [B, 1024, 171] 
        self.MMoRE = torch.compile(
            MMoRE_module, 
            mode='default', 
            backend='inductor', 
            dynamic=False
        )
        
    def forward(self, quadrant_pixels, interpolate_pos_encoding=True):

        last_hs = self.model(quadrant_pixels)["hidden_states"][-1] 
        last_hs = torch.flatten(last_hs, -2, -1).contiguous() 
        output = self.MMoRE(last_hs)
        return torch.cat(output, dim=1)
