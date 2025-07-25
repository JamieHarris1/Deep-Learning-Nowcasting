import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class SubsetSampler(Sampler):
    r""" Sampler for PyTorch that uses a given subset of indices to sample from. Not random, to use for reproducible samplings
    on the test set.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        for idx in self.indices:
            yield idx
    def __len__(self):
        return len(self.indices)

def nll(y_true, y_pred):
    nll_loss = -y_pred.log_prob(y_true)
    return nll_loss

def hybrid_loss(y_true, y_pred):
    """ Add negative log likelihood to percentage error to encourage
    accurate predictions of smaller values
    """
    return nll(y_true=y_true, y_pred=y_pred) + abs(y_true - y_pred.mode)

def mae(y_true, y_pred):
    return abs(y_true - y_pred.mode)

def mse(y_true, y_pred):
    return (y_true - y_pred.mode).pow(2)

def get_loss(y_true, y_pred, loss_fct):
    match loss_fct:
        case "nll":
            return nll(y_true, y_pred)
        case "hybrid":
            return hybrid_loss(y_true, y_pred)
        case "mae":
            return mae(y_true, y_pred)
        case "mse":
            return mse(y_true, y_pred)
    raise ValueError(f"Loss function {loss_fct} not supported. Choose one of hybrid, nll, mse or mae.")

def process_preds_observed(dist_pred, num_obs):
    """ Function to include information about the number of cases already observed.
    Any predicted values below this lower bound will be set to the lower bound
    
    Args:
        dist_pred[torch.tensor]: tensor of dimension (batch)
    """



def train(model, num_epochs, train_loader, val_loader, early_stopper, loss_fct = "nll", device = torch.device("mps"), dow = False, num_obs = False, label_type='y', alpha=0):
    model.to(device)
    model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003, weight_decay=1e-3)
    early_stopper.reset() # set counter to zero if same instance used for multiple training runs
    for e in range(num_epochs):
        batch_loss = 0.
        model.train()
        for mat, y in train_loader:
            optimizer.zero_grad()
            if num_obs:
                y, _ = y
            if dow:
                mat, dow_val = mat.copy()
                dist_pred = model(mat, dow_val)
            else:
                dist_pred = model(mat.to(device))
            loss = get_loss(y.to(device), dist_pred, loss_fct=loss_fct).mean()
            loss.retain_grad()
            loss.backward()
            #nn.utils.clip_grad_value_(model.parameters(), 10.0)

            ## Check for inf or nan gradients - stop updates in that case
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    #print(f"{name} - Gradient NaNs: {torch.isnan(param.grad).any()} - Max Gradient: {param.grad.abs().max()}")
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print("Detected inf or nan values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
        
            optimizer.step()
            batch_loss += loss.item()
        
        batch_loss /= len(train_loader)
        with torch.no_grad(): # performance on test/validation set
            model.eval()
            #model.drop1.train()
            #model.drop2.train()
            test_batch_loss = 0.
            for mat, y in val_loader:
                if num_obs:
                    y, _ = y
                if dow:
                    mat, dow_val = mat
                    test_pred = model(mat.to(device), dow_val.to(device))
                else:
                    test_pred = model(mat.to(device))
                test_loss = get_loss(y.to(device), test_pred, loss_fct=loss_fct).mean()
                if label_type == "z":
                    test_loss += alpha * torch.nn.functional.mse_loss(test_pred.mean.sum(-1), y.to(device).sum(-1).detach())
            
                test_batch_loss += test_loss.item()
            #test_batch_loss /= len(test_loader)
            if early_stopper.early_stop(test_batch_loss, model):
                model.train() # set back to train for sampling
                break
        model.train()
        #if e % 50 == 0 or e == num_epochs-1:
        print(f"Epoch {e+1} - Train loss: {batch_loss:.3} - Val loss: {test_batch_loss:.3} - ES count: {early_stopper.get_count()}")
    

class EarlyStopper:
    """ Class implementing early stopping. Theoretically, PyTorch lightning could be used, but this might be more rubust.
    
    As seen e.g. in https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch and adapted to include 
    restoration of best weights.
    """
    def __init__(self, past_units, max_delay, weeks = False, future_obs = 0, state = "SP", triangle = True, patience = 30, random_split = False, dow = False, n_training = None, biggest_outbreak = False):
        self.patience = patience
        self.counter = 0
        self.min_loss = float('inf')
        self.past_units = past_units
        self.max_delay = max_delay
        self.weeks = weeks
        self.future_obs = future_obs
        self.state = state
        self.triangle = triangle
        self.random_split = random_split
        self.dow = dow
        self.n_training = n_training
        self.biggest_outbreak = biggest_outbreak

    def early_stop(self, val_loss, model):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            ## Save best weights
            
            torch.save(model.state_dict(), f"../src/outputs/weights/weights-{self.past_units}-{self.max_delay}")
        elif val_loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def get_count(self):
        return self.counter
    
    def get_patience(self):
        return self.patience
    
    def reset(self):
        self.counter = 0






def train_type_name(model, num_epochs, train_loader, val_loader, early_stopper, loss_fct = "nll", device = torch.device("mps"), dow = False, num_obs = False, label_type='y', alpha=0):
    model.to(device)
    model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003, weight_decay=1e-3)
    early_stopper.reset() # set counter to zero if same instance used for multiple training runs
    for e in range(num_epochs):
        batch_loss = 0.
        model.train()
        for (rep_tri, dow, type_name_tri), (y, p_true) in train_loader:
            optimizer.zero_grad()
            dist_pred, p_pred, active_type_names = model(rep_tri, dow, type_name_tri)

            ll_loss = get_loss(y.to(device)[active_type_names], dist_pred, loss_fct=loss_fct).mean()
            # kl_loss = F.kl_div(p_pred.log(), p_true[active_type_names], reduction='batchmean')
            loss = ll_loss
            loss.retain_grad() 
            loss.backward()
            #nn.utils.clip_grad_value_(model.parameters(), 10.0)

            ## Check for inf or nan gradients - stop updates in that case
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    #print(f"{name} - Gradient NaNs: {torch.isnan(param.grad).any()} - Max Gradient: {param.grad.abs().max()}")
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print("Detected inf or nan values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
        
            optimizer.step()
            batch_loss += loss.item()
        
        batch_loss /= len(train_loader)
        with torch.no_grad(): # performance on test/validation set
            model.eval()
            test_batch_loss = 0.
            for (rep_tri, dow, type_name_tri), (y, p_true) in val_loader:
                test_dist_pred, test_p_pred, test_active_type_names = model(rep_tri.to(device), dow.to(device), type_name_tri.to(device))
                test_ll_loss = get_loss(y.to(device)[test_active_type_names], test_dist_pred, loss_fct=loss_fct).mean()
  
                # test_kl_loss = F.kl_div(test_p_pred.log(), p_true[test_active_type_names], reduction='batchmean')
                test_loss = test_ll_loss

                test_batch_loss += test_loss.item()
            #test_batch_loss /= len(test_loader)
            if early_stopper.early_stop(test_batch_loss, model):
                model.train() # set back to train for sampling
                break
        model.train()
        #if e % 50 == 0 or e == num_epochs-1:
        print(f"Epoch {e+1} - Train loss: {batch_loss:.3} - Val loss: {test_batch_loss:.3} - ES count: {early_stopper.get_count()}")
    
