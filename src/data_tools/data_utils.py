import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset


def create_reporting_delay_matrix(
    start_year, end_year, max_delay, data_folder_path,
    input_data_filename, output_data_filename
    ): 

    # Get the directory of the current script
    project_dir = Path.cwd()

    # Go up one level and into the data folder
    input_data_path = data_folder_path / "raw" / input_data_filename

    # Read in raw data
    dengdf_raw = pd.read_csv(input_data_path, index_col=0)
    
    # Filter for date of symptom onset and the reporting date
    dengdf = dengdf_raw[['DT_SIN_PRI', 'DT_NOTIFIC']].dropna()
    dengdf.columns = ['Date_Symptoms', 'Date_Reported']
    dengdf = dengdf.apply(pd.to_datetime).sort_values(by='Date_Symptoms')

    # Compute delay and convert to integer rather than Timedelta object
    dengdf['Delay'] = (dengdf['Date_Reported'] - dengdf['Date_Symptoms']).dt.days

    # Filter between start and end year
    dengdf = dengdf[
    (dengdf["Date_Symptoms"].dt.year >= start_year) & 
    (dengdf["Date_Symptoms"].dt.year <= end_year) &
    (dengdf["Date_Reported"].dt.year >= start_year) &
    (dengdf["Date_Reported"].dt.year <= end_year)
    ]

    # Filter out rows with delays greater than max_delay
    dengdf = dengdf[
        (dengdf['Delay'] < max_delay) &
        (dengdf['Delay'] >= 0)
        ]

    # Now want to create contingency table for every symptom date for each delay 
    deng_delays = pd.crosstab(dengdf['Date_Symptoms'], dengdf['Delay'])

    # Check for any NAs across all (Delay,Symptom combinations)
    (deng_delays.isna().sum() > 0).sum()

    # Ensure all days between start_year and end_year are present
    all_days = pd.DataFrame({
        "Date_Symptoms": pd.date_range(
            start=f"{start_year}-01-01", end=f"{end_year}-12-31")
    })
    deng_delays = all_days.merge(deng_delays, on="Date_Symptoms", how="left")
    deng_delays.fillna(0).astype(int)

    # Rename cols for clarity
    deng_delays.columns = [deng_delays.columns[0]] + [f"delay_{col}" for col in deng_delays.columns[1:]]

    # Write transformed data to csv in same folder
    output_data_path = data_folder_path / "transformed" / output_data_filename
    deng_delays.to_csv(output_data_path, index=False)
    return output_data_path

def create_reporting_proportion_matrix(
    max_delay, input_data_filename, output_data_filename
):
    # Get the directory of the current script
    project_dir = Path.cwd()

    # Go up one level and into the data folder
    input_data_path = project_dir / "data" / "transformed" / input_data_filename

    # Read in raw data
    dengdf = pd.read_csv(input_data_path, index_col=0)

    dengdf['y'] = dengdf.sum(axis=1)
    reporting_props = dengdf.copy()

    assert len(reporting_props.columns) == max_delay + 1, "Wrong number of cols"
    delay_cols = reporting_props.columns[:-1]
    reporting_props[delay_cols] = reporting_props[delay_cols].div(dengdf['y'], axis=0)

    # Note sum of props is never quite equal to one

    # Rename cols for clarity
    reporting_props.columns=  [f"delay_prop_{col_num}" for col_num in list(range(len(reporting_props.columns)))[:-1]] + [reporting_props.columns[-1]]

    # Write transformed data to csv in same folder
    output_data_path = project_dir / "data" / "transformed" / output_data_filename
    reporting_props.to_csv(output_data_path, index=True)
    return output_data_path
    
def create_data_split(
        start_year, end_year, D,
        data_folder_path, input_filename, train_prop, val_prop, test_prop):
    reporting_delay_path = create_reporting_delay_matrix(
        start_year, end_year, D, data_folder_path, input_filename, "reporting_delay.csv"
    )
    reporting_dealy_df = pd.read_csv(reporting_delay_path, index_col=0)

    # Drop last D days as they are incomplete
    complete_df = reporting_dealy_df.iloc[D-1: -D+1]

    # First max_delay days are not usable
    n_usable_obs = len(complete_df)

    # Split the number of usable obs over the 3 datasets
    train_end_idx = D + int(train_prop*n_usable_obs)
    val_end_idx = train_end_idx + int(val_prop*n_usable_obs)

    train_df = complete_df.iloc[0: train_end_idx]
    val_df = complete_df.iloc[train_end_idx - D : val_end_idx]
    test_df = complete_df.iloc[val_end_idx - D : ]

    # Write transformed data to csv in same folder
    # Get the directory of the current script
    model_data_path = data_folder_path / "model"
    train_df.to_csv(model_data_path / "training_data.csv", index=True)
    val_df.to_csv(model_data_path / "validation_data.csv", index=True)
    test_df.to_csv(model_data_path / "test_data.csv", index=True)
    complete_df.to_csv(model_data_path / "complete_data.csv", index=True)
    return None
    

class ReportingDataset(Dataset):
    def __init__(self, data, D, M, max_val, device='mps'):
        self.df = data.copy()
        self.D = D
        self.M = M
        self.device = device
        self.max_val = max_val
        
    def __len__(self):
        return len(self.df) - self.D - self.M
    
    def __getitem__(self, idx):
        # Ignore first D days to ensure full matrix
        t = idx + self.D + self.M

        # Create reporting triangle for day t
        matrix = self.df.iloc[t - self.D - self.M: t, :self.D].copy().values
        

        # Mask last D days
        matrix[-self.D:] = self.mask_delay_prop_matrix(matrix[-self.D:])

        # Create y label
        y = self.df.iloc[t].sum()

        # Convert to tensors
        matrix = torch.tensor(matrix, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Put tensors on device
        matrix.to(device=self.device)
        y.to(device=self.device)

        # Return in matrix, y in tensor format
        return (matrix / self.max_val, y)

    def mask_delay_prop_matrix(self, matrix):
        for i in range(self.D):
            for j in range(self.D):
                if i + j > self.D - 1:
                    matrix[i, j] = 0
        return matrix

