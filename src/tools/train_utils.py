import torch
from pathlib import Path


class EarlyStopper:
    def __init__(self, patience, max_delay):

        self.project_dir = Path(__file__).resolve().parents[2]
        self.weight_dir = self.project_dir / "src" / "outputs" / "weights"
        self.patience = patience
        self.max_delay = max_delay
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, val_loss, model):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0

            # Save best weights
            torch.save(model.state_dict(), self.weight_dir / f"weights_max_delay_{self.max_delay}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def reset(self):
        self.counter = 0
    
    def get_count(self):
        return self.counter


def train(model, early_stopper, train_loader, val_loader, optimizer, loss_fn, device, num_epochs):
    model.to(device)
    early_stopper.reset()
    
    for epoch in range(num_epochs):
        # Put model into training mode
        model.train()

        #Initialise training loss
        train_loss = 0.0

        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass      
            outputs = model(inputs)    

            # Compute loss on batch      
            loss = loss_fn(outputs, target) 

            # Backpropagation
            loss.backward()

            # Optimizer update
            optimizer.step()

            # Potential to check for inf or nan grads here

            # Multiple batch loss by number of items in the batch
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Asses performance on val dataset
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1} - Train loss: {train_loss:.3} - Val loss: {val_loss:.3} - ES count: {early_stopper.get_count()}")

        if early_stopper.early_stop(val_loss, model):
            break