import torch.nn as nn
import torch
from sparsemax import Sparsemax
from PNN.NegativeBinomial import NegBin as NB
from PNN.NegativeBinomial import ZINegBin as ZINB


## For matrix-like (two-dimensional) input data
class NowcastPNN(nn.Module):
    def __init__(self, past_units = 30, max_delay = 40, hidden_units = [16, 8], conv_channels = [16, 1], dropout_probs = [0.15, 0.1]):
        super().__init__()
        self.past_units = past_units
        self.max_delay = max_delay
        self.final_dim = past_units# * (2**len(conv_channels))
        self.conv1 = nn.Conv1d(self.max_delay, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        #self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=7, padding="same")
        #self.conv4 = nn.Conv1d(conv_channels[2], conv_channels[3], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.past_units, self.past_units)#, nn.Linear(self.past_units, self.past_units)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])
        #self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[-1], 2)
        #self.fcpoi = nn.Linear(hidden_units[1], 1)
        self.const = 10000

        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.max_delay), nn.BatchNorm1d(num_features=conv_channels[0])#, nn.BatchNorm1d(num_features=conv_channels[1])#, nn.BatchNorm1d(num_features=conv_channels[2])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])#, nn.BatchNorm1d(num_features=hidden_units[2])
        #self.bnorm7 = nn.BatchNorm1d(num_features=hidden_units[1])
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.past_units for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.max_delay, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
    
    def forward(self, x): ## Feed forward function, takes input of shape [batch, past_units, max_delay]
        #x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = x.float() # maybe uncomment
        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        #x = self.act(self.conv3(self.bnorm3(x)))
        #x = self.act(self.conv4(self.bnorm4(x)))
        x = torch.squeeze(x, dim = 1) # only squeeze max_delay dimension in case a single obs (batch size of 1) is passed

        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        #x = self.drop3(x)
        #x = self.act(self.fc5(self.bnorm7(x)))
        #x = self.drop3(x)
        x = self.fcnb(self.bnorm_final(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = (self.const**2)*self.softplus(x[:, 1])+1e-5)
        #x = self.fcpoi(self.bnorm7(x))
        #dist = torch.distributions.Poisson(rate=self.const*self.softplus(x))
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)

"""## For summed (one-dimensional) input data
class PNNSumDaily(nn.Module):
    def __init__(self, past_units = 45, max_delay = 45, n_layers = 3, hidden_units = [64, 32, 16]):
        super().__init__()
        self.past_units = past_units
        self.max_delay = max_delay
        self.attfc1 = nn.Linear(self.past_units, self.past_units)
        self.attfc2 = nn.Linear(self.past_units, self.past_units)
        self.attfc3 = nn.Linear(self.past_units, self.past_units)
        self.attfc4 = nn.Linear(self.past_units, self.past_units)
        # Should iterate over n_layers for more robust solution and make ModuleList
        self.fc3 = nn.Linear(past_units, hidden_units[0])
        self.fc4 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcpoi = nn.Linear(hidden_units[2], 1)
        self.fcnb = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.lnorm1, self.lnorm2, self.lnorm3, self.lnorm4 = nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1])
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.2), nn.Dropout(0.4), nn.Dropout(0.2)
        self.softplus = nn.Softplus()
        self.relu, self.silu = nn.ReLU(), nn.SiLU()
    
    def forward(self, x):
        #print(x.size())
        #x = x + self.pos_embed(x)
        x = torch.unsqueeze(x, -1)#.permute(0, 2, 1)
        #print(f"Before att layers: {x.size()}")
        x_add = x.clone()
        x = self.lnorm1(x)
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.attfc1(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x + x_add
        x = x.permute(0, 2, 1) # [batch, past_units, 1] -> [batch, 1, past_units], so can take past_units
        x = torch.squeeze(x)
        x = self.silu(self.fc3(self.bnorm1(x)))
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcnb(self.bnorm4(x))
        #dist = torch.distributions.Poisson(rate=1000*self.softplus(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)"""
class NowcastPNNDOW(nn.Module):
    """ Still NowcastPNN, just this time processing the day of the week additionally to reporting triangle """
    def __init__(self, past_units = 40, max_delay = 40, hidden_units = [16, 8], conv_channels = [16, 1], embedding_dim = 10, dropout_probs = [0.15, 0.1], device="mps"):
        super().__init__()
        self.device = device
        self.past_units = past_units
        self.max_delay = max_delay
        self.final_dim = past_units
        self.conv1 = nn.Conv1d(self.max_delay, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.past_units, self.past_units)#, nn.Linear(self.past_units, self.past_units)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])
        #self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[-1], 2)
        self.const = 10000 # if not normalized, take constant out
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(7, embedding_dim)
        #self.embed.weight.requires_grad_(False)
        #self.embed.weight = nn.Parameter(torch.randn((7, embedding_dim))), can use to initialize, doesn't help
        self.fc_embed1, self.fc_embed2 = nn.Linear(embedding_dim, 2*embedding_dim), nn.Linear(2*embedding_dim, past_units)
        #self.fc_embed1 = nn.Linear(embedding_dim, past_units)

        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.max_delay), nn.BatchNorm1d(num_features=conv_channels[0])#, nn.BatchNorm1d(num_features=conv_channels[1])#, nn.BatchNorm1d(num_features=conv_channels[2])
        #self.bnorm3 = nn.BatchNorm1d(num_features=conv_channels[1])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])#, nn.BatchNorm1d(num_features=hidden_units[2])
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        #self.bnorm7 = nn.BatchNorm1d(num_features=hidden_units[1])
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.past_units for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.max_delay, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
    
    def save_embeddings(self):
        """ Allows the user to save the embeddings if trained with a different dimension
        to load later and allow for reproducible training runs. Usage: run model with load_embed = False,
        then use model.save_embeddings() after training and use the model with load_embed = True afterwards.
        """
        torch.save(self.embed.weight, f"../src/outputs/weights/embedding_weights_{self.embedding_dim}")
    
    def forward(self, rep_tri, dow): ## Feed forward function, takes input of shape [batch, past_units, max_delay]
        #x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = rep_tri.float()
        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        # Think about processing delay_dim, meaning indep for each time step, permute after
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        #x = self.act(self.conv3(self.bnorm3(x)))
        #x = self.act(self.conv4(self.bnorm4(x)))
        x = torch.squeeze(x, 1)
        ## Addition of embedding of day of the week ##
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)

        # nn.Embedding only available on the cpu
        self.embed = self.embed.to("cpu")
        embedded = self.embed(dow.to("cpu"))
        embedded = embedded.to(self.device)

        #print(embedded)
        x = x + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        #x = self.drop3(x)
        #x = self.act(self.fc5(self.bnorm7(x)))
        x = self.fcnb(self.bnorm_final(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = (self.const**2)*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
    

""" from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
count_parameters(npnn) """




class PropPNN(nn.Module):
    
    def __init__(self, max_val, past_units = 40, max_delay = 40, hidden_units = [16, 8], conv_channels = [16, 1], embedding_dim = 10, dropout_probs = [0.15, 0.1], device="cpu"):
        super().__init__()
        self.temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.device = device
        self.past_units = past_units
        self.max_delay = max_delay
        self.final_dim = past_units
        self.conv1 = nn.Conv1d(self.max_delay, conv_channels[0], kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        self.fc1 = nn.Linear(self.past_units, self.past_units)#, nn.Linear(self.past_units, self.past_units)
        self.fc3, self.fc4 = nn.Linear(self.final_dim, hidden_units[0]), nn.Linear(hidden_units[0], hidden_units[1])#, nn.Linear(hidden_units[1], hidden_units[2])
        #self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[-1], 2)
        self.const = 10000 # if not normalized, take constant out
        self.max_val = max_val
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(7, embedding_dim)
        #self.embed.weight.requires_grad_(False)
        #self.embed.weight = nn.Parameter(torch.randn((7, embedding_dim))), can use to initialize, doesn't help
        self.fc_embed1, self.fc_embed2 = nn.Linear(embedding_dim, 2*embedding_dim), nn.Linear(2*embedding_dim, past_units)
        #self.fc_embed1 = nn.Linear(embedding_dim, past_units)
        self.fc_embed_prop1 = nn.Linear(embedding_dim, hidden_units[1])
        self.fc_embed_prop2 = nn.Linear(hidden_units[1], self.past_units)
        self.bnorm_embed_prop = nn.BatchNorm1d(self.past_units)

        self.bnorm1, self.bnorm2 = nn.BatchNorm1d(num_features=self.max_delay), nn.BatchNorm1d(num_features=conv_channels[0])#, nn.BatchNorm1d(num_features=conv_channels[1])#, nn.BatchNorm1d(num_features=conv_channels[2])
        #self.bnorm3 = nn.BatchNorm1d(num_features=conv_channels[1])
        self.bnorm5, self.bnorm6  = nn.BatchNorm1d(num_features=self.final_dim), nn.BatchNorm1d(num_features=hidden_units[0])#, nn.BatchNorm1d(num_features=hidden_units[2])
        self.bnorm_embed = nn.BatchNorm1d(num_features=2*embedding_dim)
        #self.bnorm7 = nn.BatchNorm1d(num_features=hidden_units[1])
        self.bnorm_final = nn.BatchNorm1d(num_features=hidden_units[-1]) #hidden_units[1]/self.past_units for single model
        self.attn1 = nn.MultiheadAttention(embed_dim=self.max_delay, num_heads=1, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout_probs[0]), nn.Dropout(dropout_probs[1]) 
        self.softplus = nn.Softplus()
        self.act = nn.SiLU()

        self.fc_prop1 = nn.Linear(self.past_units, hidden_units[0])
        self.fc_prop2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_prop3 = nn.Linear(hidden_units[1], hidden_units[1])
        self.fc_prop4 = nn.Linear(hidden_units[1], 1)
        self.drop1 = nn.Dropout(0.1)
        self.bnorm_prop = nn.BatchNorm1d(num_features=hidden_units[1])

        self.norm1 = nn.LayerNorm(self.past_units)
        self.drop_prop = nn.Dropout(0.1)
        self.embed_prop = nn.Embedding(7, embedding_dim).to(device)
        self.conv_prop = nn.Conv1d(in_channels=self.past_units, out_channels=self.past_units, kernel_size=3, stride=1, padding=1)

        self.fc_temp1 = nn.Linear(1, 8)
        self.fc_temp2 = nn.Linear(8, 1)


    def save_embeddings(self):
        """ Allows the user to save the embeddings if trained with a different dimension
        to load later and allow for reproducible training runs. Usage: run model with load_embed = False,
        then use model.save_embeddings() after training and use the model with load_embed = True afterwards.
        """
        torch.save(self.embed.weight, f"../src/outputs/weights/embedding_weights_{self.embedding_dim}")
    
    def forward(self, rep_tri, dow): ## Feed forward function, takes input of shape [batch, past_units, max_delay]
        B, M, D = rep_tri.shape
        #x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = rep_tri.float()
        ## Attention Block ##
        x_add = x.clone()
        x = self.attn1(x, x, x, need_weights = False)[0]
        # Think about processing delay_dim, meaning indep for each time step, permute after
        x = self.act(self.fc1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add

        ## Convolutional Block ##
        x = x.permute(0, 2, 1) # [batch, past_units, max_delay] -> [batch, max_delay, past_units]
        x = self.act(self.conv1(self.bnorm1(x)))
        x = self.act(self.conv2(self.bnorm2(x)))
        #x = self.act(self.conv3(self.bnorm3(x)))
        #x = self.act(self.conv4(self.bnorm4(x)))
        x = torch.squeeze(x, 1)
        ## Addition of embedding of day of the week ##
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)

        # nn.Embedding only available on the cpu
        self.embed = self.embed.to("cpu")
        embedded = self.embed(dow.to("cpu"))
        embedded = embedded.to(self.device)

        #print(embedded)
        x = x + self.act(self.fc_embed2(self.bnorm_embed(self.act(self.fc_embed1(embedded))))) # self.bnorm_embed1(embedded)
        ## Fully Connected Block ##
        x = self.drop1(x)
        x = self.act(self.fc3(self.bnorm5(x)))
        x = self.drop2(x)
        x = self.act(self.fc4(self.bnorm6(x)))
        #x = self.drop3(x)
        #x = self.act(self.fc5(self.bnorm7(x)))
        x = self.fcnb(self.bnorm_final(x))

        lbda = self.max_val * self.softplus(x[:, 0])
        phi = (self.max_val**2) * self.softplus(x[:, 1])+1e-5

        lbda = lbda.unsqueeze(-1)                             
        phi = phi.unsqueeze(-1)
        

        #MLP layers
        x_prop = rep_tri.float()
        x_prop = self.act(self.fc_prop1(x_prop))
        x_prop = self.act(self.fc_prop2(x_prop))
        x_prop = self.act(self.fc_prop3(x_prop))
        x_prop = self.drop_prop(x_prop)
        x_prop = self.act(self.fc_prop4(x_prop))

        # temperature = self.softplus(self.temperature_raw)
        temp_low = 0.7
        temp_high = 1.3

        x_temp = self.act(self.fc_temp1(torch.log(lbda + 1e-6)))

        # Learn a value in [0, 1]
        gate = torch.sigmoid(self.fc_temp2(x_temp))

        # Interpolate between low vs high temp
        temperature = gate * temp_low + (1 - gate) * temp_high

        scaled_logits = x_prop.squeeze(-1) / temperature
        p = torch.softmax(scaled_logits, dim=-1) 
          
                                
        mu = p * lbda 
        dist = NB(lbda=mu, phi=phi) 

        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
class PropNet(nn.Module):
    def __init__(self, max_val, D, M, device="cpu", embedding_dim=10, conv_channels=[16, 1], hidden_units=[16, 8], dropout_probs=[0.3, 0.1]):
        super().__init__()
        self.max_val = max_val
        self.D = D
        self.M = M
        self.device = device

        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.lbda_scale = nn.Parameter(torch.tensor(self.max_val, dtype=torch.float32))
        self.phi_scale = nn.Parameter(torch.tensor(self.max_val**2, dtype=torch.float32))

        ## Total count model ##

        self.attn_count1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=1, batch_first=True)
        
        self.conv_count1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv_count2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        
        self.fc_count1 = nn.Linear(self.M, self.M)
        self.fc_count2 = nn.Linear(self.M, hidden_units[0])
        self.fc_count3 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_count4 = nn.Linear(hidden_units[1], 2)
        
        self.embed_day = nn.Embedding(7, embedding_dim)
        self.embed_week = nn.Embedding(53, embedding_dim)
        self.fc_embed_day1 = nn.Linear(embedding_dim, 2*embedding_dim)
        self.fc_embed_day2 = nn.Linear(2*embedding_dim, self.M)
        self.fc_embed_week1 = nn.Linear(embedding_dim, 2*embedding_dim)
        self.fc_embed_week2 = nn.Linear(2*embedding_dim, self.M)
        
        self.drop_count1 = nn.Dropout(dropout_probs[0])
        self.drop_count2 = nn.Dropout(dropout_probs[1])

        self.bnorm_count1 = nn.BatchNorm1d(num_features=self.D)
        self.bnorm_count2 = nn.BatchNorm1d(num_features=conv_channels[0])
        self.bnorm_count3 = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_count4 = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_count5 = nn.BatchNorm1d(num_features=self.M)
        self.bnorm_count6 = nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm_count7 = nn.BatchNorm1d(num_features=hidden_units[1])

        ## Proportion model ##
        self.fc_prop1 = nn.Linear(self.M, hidden_units[0])
        self.fc_prop2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_prop3 = nn.Linear(hidden_units[1], hidden_units[1])
        self.fc_prop4 = nn.Linear(hidden_units[1], 1)

        self.drop_prop1 = nn.Dropout(dropout_probs[0])
        self.drop_prop2 = nn.Dropout(dropout_probs[1])

        self.fc_temp1 = nn.Linear(1, 8)
        self.fc_temp2 = nn.Linear(8, 1)


    def forward(self, rep_tri, dow): 
        x = rep_tri.float()
        B, M, D = x.shape

        ## Predict total count ##

        # Attention block
        x_add = x.clone()
        x = self.attn_count1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc_count1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add
        x = x.permute(0, 2, 1)
        
        x = self.act(self.conv_count1(self.bnorm_count1(x)))
        x = self.act(self.conv_count2(self.bnorm_count2(x)))
        x = torch.squeeze(x, 1)

        # Day of the week effect
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)
        embedded_day = self.embed_day(dow)
        e_day = self.act(self.fc_embed_day2(self.bnorm_count3(self.act(self.fc_embed_day1(embedded_day)))))

        x = x + e_day

        # Final dense layers
        
        x = self.drop_count1(x)
        x = self.act(self.fc_count2(self.bnorm_count5(x)))
        x = self.drop_count2(x)
        x = self.act(self.fc_count3(self.bnorm_count6(x)))
        x = self.fc_count4(self.bnorm_count7(x))
        
        # Predict NB params
        lbda = self.lbda_scale * self.softplus(x[:, 0]) + 1e-5
        phi = self.phi_scale * self.softplus(x[:, 1]) + 1e-5

        lbda = lbda.unsqueeze(-1)                             
        phi = phi.unsqueeze(-1)                    

        ## Predict delay proportions ##
        x_prop = rep_tri.float()

        # Make time dim last dimension
        x_prop = x_prop.permute(0, 2, 1)

        x_prop = self.act(self.fc_prop1(x_prop))
        x_prop = self.act(self.fc_prop2(x_prop))
        x_prop = self.drop_prop1(x_prop)
        x_prop = self.act(self.fc_prop3(x_prop))
        x_prop = self.drop_prop2(x_prop)
        x_prop = self.act(self.fc_prop4(x_prop))

        ## Temperature ##
        # temperature = self.softplus(self.temperature_raw)
        temp_low = 0.4
        temp_high = 1.3

        print(x_prop.shape)
        print(lbda.shape)
        x_temp = self.act(self.fc_temp1(torch.log(lbda + 1e-6)))

        # Learn a value in [0, 1]
        gate = torch.sigmoid(self.fc_temp2(x_temp))

        # Interpolate between low vs high temp
        temperature = gate * temp_low + (1 - gate) * temp_high

        # Learn sharpness of proportion dist with temp
        # temperature = self.softplus(self.temperature_raw)
        scaled_logits = x_prop.squeeze(-1) / temperature 
        
        ## Final Distribution params, shape: (B,D) ##
        p = torch.softmax(scaled_logits, dim=-1)
        mu = p * lbda 
        

        dist = NB(lbda=mu, phi=phi)

        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        # x shape: (B, L, dim)
        B, L, D = x.shape
        out = self.fc1(x.view(B*L, D))
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out.view(B, L, D)
        return self.act(out + x)
 

class TypePNN(nn.Module):
    def __init__(self, max_val, D, M, T, D_t, n_type_name, device="cpu", embedding_dim=10, conv_channels=[16, 1], hidden_units=[16, 8], dropout_probs=[0.3, 0.1]):
        super().__init__()
        self.max_val = max_val
        self.D = D
        self.M = M
        self.T = T
        self.D_t = D_t
        self.n_type_name = n_type_name
        self.device = device

        self.softplus = nn.Softplus()
        self.act = nn.SiLU()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.lbda_scale = nn.Parameter(torch.tensor(self.max_val, dtype=torch.float32))
        self.phi_scale = nn.Parameter(torch.tensor(self.max_val**2, dtype=torch.float32))

        ## Total count model ##

        self.attn_count1 = nn.MultiheadAttention(embed_dim=self.D, num_heads=1, batch_first=True)
        
        self.conv_count1 = nn.Conv1d(self.D, conv_channels[0], kernel_size=7, padding="same")
        self.conv_count2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding="same")
        
        self.fc_count1 = nn.Linear(self.M, self.M)
        self.fc_count2 = nn.Linear(self.M, hidden_units[0])
        self.fc_count3 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc_count4 = nn.Linear(hidden_units[1], 2)
        
        self.embed_day = nn.Embedding(7, embedding_dim)
        self.embed_week = nn.Embedding(53, embedding_dim)
        self.fc_embed_day1 = nn.Linear(embedding_dim, 2*embedding_dim)
        self.fc_embed_day2 = nn.Linear(2*embedding_dim, self.M)
        self.bnorm_week = nn.BatchNorm1d(num_features=2*embedding_dim)
        
        self.drop_count1 = nn.Dropout(dropout_probs[0])
        self.drop_count2 = nn.Dropout(dropout_probs[1])

        self.bnorm_count1 = nn.BatchNorm1d(num_features=self.D)
        self.bnorm_count2 = nn.BatchNorm1d(num_features=conv_channels[0])
        self.bnorm_count3 = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_count4 = nn.BatchNorm1d(num_features=2*embedding_dim)
        self.bnorm_count5 = nn.BatchNorm1d(num_features=self.M)
        self.bnorm_count6 = nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm_count7 = nn.BatchNorm1d(num_features=hidden_units[1])

        ## Proportion model ##
        self.fc_phi_head = nn.Linear(1, self.n_type_name)
        self.fc_shared = nn.Linear(self.M, self.n_type_name)

        conv_out_channels = 16
        fc_hidden = 16
        self.conv_prop1 = nn.Conv1d(self.D_t, conv_out_channels, kernel_size=3, padding=1)
        self.conv_prop2 = nn.Conv1d(conv_out_channels, 1, kernel_size=3, padding=1)

        # Fully connected layers for final proportion prediction
        self.fc_prop1 = nn.Linear(self.T, fc_hidden)
        self.drop_prop1 = nn.Dropout(p=0.1)
        self.fc_prop2 = nn.Linear(fc_hidden, 1)


    def forward(self, rep_tri, dow,  type_name_tri): 
        x = rep_tri.float()

        B, M, D = x.shape
        B, n_type_name, T, D_type = type_name_tri.shape

        ## Predict total count ##

        # Attention block
        x_add = x.clone()
        x = self.attn_count1(x, x, x, need_weights = False)[0]
        x = self.act(self.fc_count1(x.permute(0,2,1)))
        x = x.permute(0,2,1) + x_add
        x = x.permute(0, 2, 1)
        
        x = self.act(self.conv_count1(self.bnorm_count1(x)))
        x = self.act(self.conv_count2(self.bnorm_count2(x)))
        x = torch.squeeze(x, 1)

        # Day of the week effect
        if len(dow.size()) == 0:
            dow = torch.unsqueeze(dow, 0)
        embedded_day = self.embed_day(dow)
        e_day = self.act(self.fc_embed_day2(self.bnorm_count3(self.act(self.fc_embed_day1(embedded_day)))))

        x = x + e_day
        x_shared = x.clone()

        # Final dense layers
        
        x = self.drop_count1(x)
        x = self.act(self.fc_count2(self.bnorm_count5(x)))
        x = self.drop_count2(x)
        x = self.act(self.fc_count3(self.bnorm_count6(x)))
        x = self.fc_count4(self.bnorm_count7(x))

        phi_raw = self.fc_phi_head(self.act(x[:, 1].unsqueeze(1)))
        
        # Predict NB params
        lbda = self.lbda_scale * self.softplus(x[:, 0]) + 1e-5
        phi = self.phi_scale * self.softplus(phi_raw) + 1e-5

        lbda = lbda.unsqueeze(-1)                             
                       

        ## Predict delay proportions ##
        #(B, n_type_name, T, D_g )
        x_prop = type_name_tri.float()
        B, N, T, D = x_prop.shape

        
        
        x_prop = x_prop.permute(0, 1, 3, 2).reshape(B * N, D, T)

        x_prop = self.act(self.conv_prop1(x_prop))
        x_prop = self.act(self.conv_prop2(x_prop))
        x_prop = x_prop.squeeze(1)
        # (B*N, T)


        x_shared = self.act(self.fc_shared(x_shared))
        x_shared = x_shared.reshape(B*self.n_type_name, 1)

        x_prop = x_prop + x_shared

        # Fully connected head
        x_prop = self.act(self.fc_prop1(x_prop))  # (B*N, hidden)
        x_prop = self.drop_prop1(x_prop)
        x_prop = self.fc_prop2(x_prop)            # (B*N, 1)
        x_prop = x_prop.view(B, N)                # (B, n_type_name)


        ## Temperature ##
        temperature = self.softplus(self.temperature_raw)
        scaled_logits = x_prop / temperature
        
        ## Final Distribution params, shape: (B,D) ##
        # p = self.sparsemax(scaled_logits)
        p = self.softmax(scaled_logits)
        mu = p * lbda

        active_type_names = p > 0.0

        mu_active = mu[active_type_names]
        phi_active = phi[active_type_names]
        p_active = p[active_type_names]


        dist = NB(lbda=mu_active, phi=phi_active)

        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1), p_active, active_type_names