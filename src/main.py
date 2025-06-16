import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split

from src.tools.data_utils import ReportingDataset
from src.tools.train_utils import SubsetSampler, SubsetRandomSampler
from src.tools.train_utils import train, EarlyStopper, 
from src.tools.NowcastPNN import NowcastPNN

# Load config vars
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SEED = config['training']['seed']
BATCH_SIZE = config['training']['batch_size']

PAST_UNITS = config['model']['past_units']
MAX_DELAY = config['model']['max_delay']
STATE = config['model']['state']
DEVICE = config['training']['device']

# set_seeds()
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True

# Load Dengue Data
path = "data/dengue-sp-reporting-delay.csv"
dengdf = pd.read_csv(path, index_col=0)
dengdf = np.array(dengdf.values, dtype = np.float32)
max_val = dengdf.max(axis = 1).max(axis=0)
dataset = ReportingDataset(dengdf, max_val=max_val, triangle=True, past_units=PAST_UNITS, max_delay=MAX_DELAY, future_obs=0, vector_y = False, dow = dow, return_number_obs = 0)

# Train/Test split
weeks = False
dow = False
random_split = True

all_idcs = range(dataset.__len__())
train_idcs, test_idcs = train_test_split(all_idcs, test_size=0.25, shuffle=True, random_state=SEED)
train_idcs, val_idcs = train_test_split(train_idcs, test_size=0.25, shuffle=True, random_state=SEED)
val_batch_size, test_batch_size = len(val_idcs), len(test_idcs)

g = torch.Generator()
g.manual_seed(SEED)

train_sampler = SubsetRandomSampler(train_idcs, generator=g)
val_sampler = SubsetRandomSampler(val_idcs)
test_sampler = SubsetSampler(test_idcs)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=val_sampler, shuffle=False)
test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler, shuffle=False)


# Train model
early_stopper = EarlyStopper(patience=30, past_units=PAST_UNITS, max_delay=MAX_DELAY, weeks=weeks, future_obs=0, random_split=random_split, dow = dow)
nowcast_pnn = NowcastPNN(past_units=PAST_UNITS, max_delay=MAX_DELAY, conv_channels=[16, 1], hidden_units=[16, 8], dropout_probs=[0.3, 0.1])
train(nowcast_pnn, num_epochs=200, train_loader=train_loader, val_loader=val_loader, early_stopper=early_stopper, loss_fct="nll", device = DEVICE, dow = dow)

# Load best set of weights on test/validation set
nowcast_pnn.load_state_dict(torch.load(f"src/outputs/weights/weights-{PAST_UNITS}-{MAX_DELAY}-{'week' if weeks else 'day'}-fut0{'-rec' if not random_split else ''}{'-dow' if dow else ''}"))


