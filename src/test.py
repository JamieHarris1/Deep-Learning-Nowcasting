import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from tools.data_utils import *



# Get the directory of the current script
project_dir = Path(__file__).resolve().parents[1]

# Load config vars
with open( project_dir / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

MAX_DELAY = config['model']['max_delay']
BATCH_SIZE = config['training']['batch_size']

reporting_delay_data_path = create_reporting_delay_matrix(
    start_year = 2013,
    end_year = 2020,
    max_delay = MAX_DELAY,
    input_data_filename = 'DENGSP.csv',
    output_data_filename = 'dengue-sp-reporting-delay.csv'
)

reporting_delay_prop_data_path = create_reporting_proportion_matrix(
    max_delay = MAX_DELAY,
    input_data_filename = 'dengue-sp-reporting-delay.csv',
    output_data_filename = 'dengue-sp-reporting-delay-proportion.csv'
)


# Read in raw data
dengdf = pd.read_csv(reporting_delay_prop_data_path, index_col=0)

# Drop total counts
propdf = dengdf.drop(["y"], axis=1)

# Drop last max_delay days as they are incomplete
propdf = propdf.iloc[0: -MAX_DELAY]

# Split into train, val, test datasets
train_prop, val_prop, test_prop = 0.7, 0.15, 0.15

# First max_delay days are not usable
n_usable_obs = len(propdf) - MAX_DELAY

# Split the number of usable obs over the 3 datasets
train_end_idx = MAX_DELAY + int(train_prop*n_usable_obs)
val_end_idx = train_end_idx + int(val_prop*n_usable_obs)

train_df = propdf.iloc[MAX_DELAY: train_end_idx]
val_df = propdf.iloc[train_end_idx - MAX_DELAY : val_end_idx]
test_df = propdf.iloc[val_end_idx - MAX_DELAY : ]

# Create PyTorch dataset objects
train_dataset = ReportingDataset(train_df, max_delay=MAX_DELAY)
val_dataset   = ReportingDataset(val_df,   max_delay=MAX_DELAY)
test_dataset  = ReportingDataset(test_df,  max_delay=MAX_DELAY)

# Create PyTorch Dataloaders without shuffle since want to keep temporal relationships
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

