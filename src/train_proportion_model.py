import pandas as pd
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tools.data_utils import ReportingDataset
from tools.proportion_matrix_net import ProportionMatrixNet
from tools.train_utils import EarlyStopper, train



# Get the directory of the current script
project_dir = Path(__file__).resolve().parents[1]

# Load config vars
with open( project_dir / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

MAX_DELAY = config['model']['max_delay']
BATCH_SIZE = config['training']['batch_size']

# Read in training and validation data
train_df = pd.read_csv(project_dir / "data" / "model" / "training_data.csv", index_col=0)
val_df = pd.read_csv(project_dir / "data" / "model" / "validation_data.csv", index_col=0)

# Create PyTorch dataset objects
train_dataset = ReportingDataset(train_df, max_delay=MAX_DELAY)
val_dataset   = ReportingDataset(val_df,   max_delay=MAX_DELAY)

# Create PyTorch Dataloaders without shuffle since want to keep temporal relationships
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train model 
early_stopper = EarlyStopper(patience=config['training']['patience'], max_delay=MAX_DELAY)
model = ProportionMatrixNet(max_delay=MAX_DELAY)
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['adam_learning_rate'])
loss_fn = nn.MSELoss()
device = torch.device(config['training']['device'])

train(model, early_stopper, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=20)

# Load best set of weights on validation set
# model.load_state_dict(torch.load(project_dir / "src" / "outputs" / "weights" / f"weights_max_delay_{MAX_DELAY}"))

