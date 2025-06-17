from cmdstanpy import CmdStanModel
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory of the current script
project_dir = Path.cwd()
train_df = pd.read_csv(project_dir / "data" / "model" / "training_data.csv", index_col=0)
y_train = train_df['y']



# Load config vars
with open( project_dir / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_path = project_dir / "src" / "model_tools" / "total_count_model" / "AR1.stan"
model = CmdStanModel(stan_file=model_path)
fit = model.sample(data={
    "T": len(y_train),
    "y": y_train,
    "D": config['model']['max_delay']})

fit.save_csvfiles(dir=str(project_dir / "src" / "outputs" / "stan_models" / "AR1_fit"))