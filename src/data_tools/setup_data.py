import pandas as pd
import yaml
from pathlib import Path

from src.data_tools.data_utils import create_reporting_delay_matrix, create_reporting_proportion_matrix




# Get the directory of the current script
project_dir = Path.cwd()

# Load config vars
with open( project_dir / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

MAX_DELAY = config['model']['max_delay']

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



# Drop last max_delay days as they are incomplete
propdf = dengdf.iloc[0: -MAX_DELAY]

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

# Write transformed data to csv in same folder
model_data_path = project_dir / "data" / "model"
train_df.to_csv(model_data_path / "training_data.csv", index=True)
val_df.to_csv(model_data_path / "validation_data.csv", index=True)
test_df.to_csv(model_data_path / "test_data.csv", index=True)