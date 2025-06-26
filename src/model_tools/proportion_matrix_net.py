import torch.nn as nn
import torch.nn.functional as F

class ProportionMatrixNet(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

        # Multihead Attention (treat each row as a sequence)
        self.attn = nn.MultiheadAttention(embed_dim=self.max_delay, num_heads=1, batch_first=True)

        # FC Block
        self.fc1 = nn.Linear(self.max_delay, self.max_delay)
        self.fc2 = nn.Linear(self.max_delay**2, 2*self.max_delay)
        self.fc3 = nn.Linear(2*self.max_delay, self.max_delay)

        # Conv Block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        

        # SiLu activation
        self.silu = nn.SiLU()

        # Dropout to prevent over fitting
        self.drop1 = nn.Dropout(p=0.15)
        self.drop2 = nn.Dropout(p=0.1)


    def forward(self, x):
        x_add = x.clone()
        
        # Attention across rows
        x, _ = self.attn(x, x, x)  # shape: [B, D, D]
        x = self.drop1(x)

        # FF on rows
        x = self.silu(self.fc1(x))
        x = self.drop2(x)

        # Residual connection
        x = x + x_add

        # Conv layers treating as if image
        x = x.unsqueeze(1)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x))) # shape : [B, D, D]
        x = x.squeeze(1)

        # Final dense layers
        x = nn.flatten(x, start_dim=1, end_dim=-1) # shape : [B, D*D]
        x = self.silu(self.fc2(x)) # shape : [B, 2D]
        x = self.silu(self.fc3(x)) # Shape : [B, D]


        return F.softmax(x, dim=-1)

