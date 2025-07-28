import torch


class BaseTrain:
    def __init__(self, model_name, device="cpu", num_epochs=200, patience = 30, lr=0.0003, weight_decay=1e-3):
        self.model_name = model_name
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_loss = float('inf')

    def early_stop(self, val_loss, model):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0

            # Save best weights
            torch.save(model.state_dict(), f"../src/outputs/weights/weights-{self.model_name}")

        elif val_loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def nll(self, y_true, y_pred):
        nll_loss = -y_pred.log_prob(y_true)
        return nll_loss
    
    def batch_forward(self, train_loader, val_loader, optimizer, model):
        model.train()
        batch_loss = 0.
        for (obs, dow), y in train_loader:
            optimizer.zero_grad()
            dist_pred = model(obs, dow)
            loss = self.nll(y, dist_pred).mean()
            loss.retain_grad()
            loss.backward()

            # Check for valid gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print("Detected inf or nan values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
        
            optimizer.step()
            batch_loss += loss.item()

        batch_loss /= len(train_loader)

        # performance on test/validation set
        with torch.no_grad(): 
            model.eval()
            val_batch_loss = 0.
            for (obs, dow), y in val_loader:
                dist_pred = model(obs, dow)
                val_loss = self.nll(y, dist_pred).mean()
                val_batch_loss += val_loss.item()

            if self.early_stop(val_batch_loss, model):
                # Return True and end training
                model.train() 
                return batch_loss, val_batch_loss, True
        # Otherwise return False and keep training
        model.train()
        return batch_loss, val_batch_loss, False

    def train_model(self, model, train_loader, val_loader):
        model.to(self.device)
        model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        self.counter = 0

        for e in range(self.num_epochs):
            
            batch_loss, val_batch_loss, early_stop_cond = self.batch_forward(train_loader, val_loader, optimizer, model)
            print(f"Epoch {e+1} - Train loss: {batch_loss:.3} - Val loss: {val_batch_loss:.3} - ES count: {self.counter}")
            if early_stop_cond:
                break

class SparsePropTrain(BaseTrain):
    def __init__(self, model_name, device="cpu", num_epochs=200, patience = 30, lr=0.0003, weight_decay=1e-3):
        super().__init__(model_name, device="cpu", num_epochs=200, patience = 30, lr=0.0003, weight_decay=1e-3)

    def batch_forward(self, train_loader, val_loader, optimizer, model):
        model.train()
        batch_loss = 0.
        for (obs, dow), y in train_loader:
            optimizer.zero_grad()
            dist_pred, active_idxs = model(obs, dow)
            loss = self.nll(y[active_idxs], dist_pred).mean()
            loss.retain_grad()
            loss.backward()

            # Check for valid gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print("Detected inf or nan values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
        
            optimizer.step()
            batch_loss += loss.item()

        batch_loss /= len(train_loader)

        # performance on test/validation set
        with torch.no_grad(): 
            model.eval()
            val_batch_loss = 0.
            for (obs, dow), y in val_loader:
                dist_pred, active_idxs = model(obs, dow)
                val_loss = self.nll(y[active_idxs], dist_pred).mean()
                val_batch_loss += val_loss.item()

            if self.early_stop(val_batch_loss, model):
                # Return True and end training
                model.train() 
                return batch_loss, val_batch_loss, True
        # Otherwise return False and keep training
        model.train()
        return batch_loss, val_batch_loss, False

