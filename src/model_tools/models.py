import torch.nn as nn
import torch
from model_tools.NegativeBinomial import NegBin as NB
from sparsemax import Sparsemax


class NowcastPNN(nn.Module):
    def __init__(self, M = 40, D = 40, hidden_units = [16, 8], conv_channels = [16, 1], embedding_dim = 10, dropout_probs = [0.15, 0.1], device="cpu"):
        super().__init__()
        self.device = device
        self.M = M
        self.D = D
        self.final_dim = M
        self.conv1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.M, self.M)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])

        self.fcnb = nn.Linear(hidden_units[-1], 2)
        self.const = 10000
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(7, embedding_dim)
        self.fc_embed1, self.fc_embed2 = nn.Linear(embedding_dim, 2*embedding_dim), nn.Linear(2*embedding_dim, M)
        
        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.D), nn.BatchNorm1d(num_features=conv_channels[0])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.M for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
    
    def forward(self, rep_tri, dow): 
        x = rep_tri.float()

        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, M, D] -> [batch, D, M]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        x = torch.squeeze(x, 1)

        ## Addition of embedding of day of the week ##
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)

        # nn.Embedding only available on the cpu
        self.embed = self.embed.to("cpu")
        embedded = self.embed(dow.to("cpu"))
        embedded = embedded.to(self.device)
        x = x + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
        
        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        x = self.fcnb(self.bnorm_final(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = (self.const**2)*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)




class PropPNN(nn.Module):
    def __init__(self, M , D, hidden_units = [16, 8], conv_channels = [16, 1], embedding_dim = 10, dropout_probs = [0.15, 0.1], device="cpu"):
        super().__init__()
        self.device = device
        self.M = M
        self.D = D
        self.final_dim = M
        self.temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.lbda_scale = nn.Parameter(torch.tensor(10000, dtype=torch.float32))
        self.phi_scale = nn.Parameter(torch.tensor(10000**2, dtype=torch.float32))

        self.conv1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.M, self.M)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])

        self.fcnb = nn.Linear(hidden_units[-1], 2)
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(7, embedding_dim)
        self.fc_embed1, self.fc_embed2 = nn.Linear(embedding_dim, 2*embedding_dim), nn.Linear(2*embedding_dim, M)
        
        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.D), nn.BatchNorm1d(num_features=conv_channels[0])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.M for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()

        # Proportion
        self.conv_prop1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv_prop2 = nn.Conv1d(conv_channels[0], self.D, kernel_size=7, padding="same")
        
        self.fc_prop1 = nn.Linear(self.M, hidden_units[0])
        self.fc_prop2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_prop3 = nn.Linear(hidden_units[1], hidden_units[1])
        self.fc_prop4 = nn.Linear(hidden_units[1], 1)

        self.drop_prop1 = nn.Dropout(dropout_probs[0])
        self.drop_prop2 = nn.Dropout(dropout_probs[1])

        self.fc_temp1 = nn.Linear(1, 8)
        self.fc_temp2 = nn.Linear(8, 1)

    
    def forward(self, rep_tri, dow): 
        x = rep_tri.float().clone()

        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, M, D] -> [batch, D, M]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        x = torch.squeeze(x, 1)

        ## Addition of embedding of day of the week ##
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)

        # nn.Embedding only available on the cpu
        self.embed = self.embed.to("cpu")
        embedded = self.embed(dow.to("cpu"))
        embedded = embedded.to(self.device)
        x = x + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
        
        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        x = self.fcnb(self.bnorm_final(x))

        # Predict NB params
        lbda = self.lbda_scale * self.softplus(x[:, 0]) + 1e-5
        phi = self.phi_scale * self.softplus(x[:, 1]) + 1e-5

        ## Proportion Model##
        x_prop = rep_tri.float().clone()
        # Make time last dim
        x_prop = x_prop.permute(0, 2, 1)

        # Conv over delays
        x_res = x_prop.clone()
        x_prop = self.conv_prop1(x_prop)
        x_prop = self.conv_prop2(x_prop)
        x_prop = x_prop + x_res
        
        # Dense layers
        x_prop = self.act(self.fc_prop1(x_prop))
        x_prop = self.act(self.fc_prop2(x_prop))
        x_prop = self.drop_prop2(x_prop)
        x_prop = self.act(self.fc_prop3(x_prop))
        x_prop = self.drop_prop1(x_prop)
        x_prop = self.act(self.fc_prop4(x_prop))

        temperature = self.softplus(self.temperature_raw)
        # temp_low = 0.7
        # temp_high = 1.3

        
        # x_temp = self.act(self.fc_temp1(torch.log(lbda.unsqueeze(-1) + 1e-6)))
       

        # # Learn a value in [0, 1]
        # gate = torch.sigmoid(self.fc_temp2(x_temp))
        # print()

        # # Interpolate between low vs high temp
        # temperature = gate * temp_low + (1 - gate) * temp_high

        scaled_logits = x_prop.squeeze(-1) / temperature
        p = torch.softmax(scaled_logits, dim=-1) 
        
        phi = phi.unsqueeze(-1)
        lbda = lbda.unsqueeze(-1)
                                
        mu = p * lbda 
        dist = NB(lbda=mu, phi=phi) 

        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)


class SparsePropPNN(nn.Module):
    def __init__(self, M , D, hidden_units = [16, 8], conv_channels = [16, 1], embedding_dim = 10, dropout_probs = [0.15, 0.1], device="cpu"):
        super().__init__()
        self.device = device
        self.M = M
        self.D = D
        self.sparsemax = Sparsemax()
        self.final_dim = M
        self.temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.lbda_scale = nn.Parameter(torch.tensor(10000, dtype=torch.float32))
        self.phi_scale = nn.Parameter(torch.tensor(10000**2, dtype=torch.float32))

        self.conv1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.M, self.M)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])

        self.fcnb = nn.Linear(hidden_units[-1], 2)
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(7, embedding_dim)
        self.fc_embed1, self.fc_embed2 = nn.Linear(embedding_dim, 2*embedding_dim), nn.Linear(2*embedding_dim, M)
        
        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.D), nn.BatchNorm1d(num_features=conv_channels[0])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.M for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()

        # Proportion
        self.conv_prop1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv_prop2 = nn.Conv1d(conv_channels[0], self.D, kernel_size=7, padding="same")
        
        self.fc_prop1 = nn.Linear(self.M, hidden_units[0])
        self.fc_prop2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_prop3 = nn.Linear(hidden_units[1], hidden_units[1])
        self.fc_prop4 = nn.Linear(hidden_units[1], 1)

        self.drop_prop1 = nn.Dropout(dropout_probs[0])
        self.drop_prop2 = nn.Dropout(dropout_probs[1])

        self.fc_temp1 = nn.Linear(1, 8)
        self.fc_temp2 = nn.Linear(8, 1)

    
    def forward(self, rep_tri, dow): 
        x = rep_tri.float().clone()

        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, M, D] -> [batch, D, M]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        x = torch.squeeze(x, 1)

        ## Addition of embedding of day of the week ##
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)

        # nn.Embedding only available on the cpu
        self.embed = self.embed.to("cpu")
        embedded = self.embed(dow.to("cpu"))
        embedded = embedded.to(self.device)
        x = x + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
        
        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        x = self.fcnb(self.bnorm_final(x))

        # Predict NB params
        lbda = self.lbda_scale * self.softplus(x[:, 0]) + 1e-5
        phi = self.phi_scale * self.softplus(x[:, 1]) + 1e-5

        ## Proportion Model##
        x_prop = rep_tri.float().clone()
        # Make time last dim
        x_prop = x_prop.permute(0, 2, 1)

        # Conv over delays
        x_res = x_prop.clone()
        x_prop = self.conv_prop1(x_prop)
        x_prop = self.conv_prop2(x_prop)
        x_prop = x_prop + x_res
        
        # Dense layers
        x_prop = self.act(self.fc_prop1(x_prop))
        x_prop = self.act(self.fc_prop2(x_prop))
        x_prop = self.drop_prop2(x_prop)
        x_prop = self.act(self.fc_prop3(x_prop))
        x_prop = self.drop_prop1(x_prop)
        x_prop = self.act(self.fc_prop4(x_prop))

        temperature = self.softplus(self.temperature_raw)
        # temp_low = 0.7
        # temp_high = 1.3

        
        # x_temp = self.act(self.fc_temp1(torch.log(lbda.unsqueeze(-1) + 1e-6)))
       

        # # Learn a value in [0, 1]
        # gate = torch.sigmoid(self.fc_temp2(x_temp))
        # print()

        # # Interpolate between low vs high temp
        # temperature = gate * temp_low + (1 - gate) * temp_high

        scaled_logits = x_prop.squeeze(-1) / temperature
        p = torch.softmax(scaled_logits, dim=-1) 
        
        phi = phi.unsqueeze(-1).expand(-1, 40)
        lbda = lbda.unsqueeze(-1)
                                
        mu = p * lbda 

        active_idxs = p > 0.0

        mu_active = mu[active_idxs]
        phi_active = phi[active_idxs]


        dist = NB(lbda=mu_active, phi=phi_active)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1), active_idxs

