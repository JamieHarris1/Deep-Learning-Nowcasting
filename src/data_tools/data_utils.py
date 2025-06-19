import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset


def create_reporting_delay_matrix(
    start_year, end_year, max_delay,
    input_data_filename, output_data_filename
    ): 

    # Get the directory of the current script
    project_dir = Path.cwd()

    # Go up one level and into the data folder
    input_data_path = project_dir / "data" / "raw" / input_data_filename

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
    output_data_path = project_dir / "data" / "transformed" / output_data_filename
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
    


class ReportingDataset(Dataset):
    def __init__(self, data, max_delay):
        self.df = data.copy()
        self.max_delay = max_delay
        
    def __len__(self):
        return len(self.df) - self.max_delay
    
    def __getitem__(self, idx):
        # Ignore first max_delay days to ensure full matrix
        t = idx + self.max_delay

        # Create label
        prop_matrix = self.df.iloc[t - self.max_delay : t].copy()
        prop_matrix_np = prop_matrix.values

        # Mask lower right triangle of matrix with zeros, in line with what is observable at time t
        prop_masked = self.mask_delay_prop_matrix(prop_matrix_np.copy())

        # Return in tensor format
        return (
            torch.tensor(prop_masked, dtype=torch.float32),
            torch.tensor(prop_matrix_np, dtype=torch.float32)
        )

    def mask_delay_prop_matrix(self, delay_prop_matrix):
        for i in range(self.max_delay):
            for j in range(self.max_delay):
                if i + j > self.max_delay - 1:
                    delay_prop_matrix[i, j] = 0
        return delay_prop_matrix

