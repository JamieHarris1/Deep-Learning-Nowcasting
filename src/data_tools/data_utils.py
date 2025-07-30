import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os


def create_reporting_delay_matrix(
    start_year, end_year, max_delay, data_folder_path,
    input_data_filename, output_data_filename
    ): 

    

    project_dir = project_dir = Path.cwd().parent

    base_folder_path = project_dir / "data" / "raw" / "counts"
    files = [f for f in os.listdir(base_folder_path) if f.endswith('.csv')]

    df_list = [pd.read_csv(os.path.join(base_folder_path, file)) for file in files]
    dengdf_raw = pd.concat(df_list, ignore_index=True)

    
    # Filter for date of symptom onset and the reporting date
    dengdf = dengdf_raw[['DT_SIN_PRI', 'DT_NOTIFIC']].dropna()
    dengdf.columns = ['Date_Symptoms', 'Date_Reported']
    dengdf = dengdf.apply(pd.to_datetime, errors='coerce')

    # Filter between start and end year
    dengdf = dengdf[
    (dengdf["Date_Symptoms"].dt.year >= start_year) & 
    (dengdf["Date_Symptoms"].dt.year <= end_year) &
    (dengdf["Date_Reported"].dt.year >= start_year) &
    (dengdf["Date_Reported"].dt.year <= end_year)
    ]
    
    dengdf = dengdf.sort_values(by='Date_Symptoms')

    # Compute delay and convert to integer rather than Timedelta object
    dengdf['Delay'] = (dengdf['Date_Reported'] - dengdf['Date_Symptoms']).dt.days

    

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


class SeroDataset:
    def __init__(self, dataset, prop_dataset, T, period="Q"):
        self.dataset = dataset
        self.prop_dataset = prop_dataset
        self.T = T
        self.period = period
    
    def get_obs(self, date):
        # Get all samples submitted before date
        date = pd.to_datetime(date).to_period(self.period)
        obs = self.dataset.loc[self.dataset['Submission date'] < date].copy()

        # Compute difference in months between collection day and date
        obs['Date diff'] = (date - obs['Collection date']).apply(lambda x: x.n)
        obs = obs[["Sero", "Quarter", "Delay", "Date diff"]]
        
        obs = obs.groupby(["Sero", "Quarter", "Delay", "Date diff"]).size().reset_index(name="Count")


        # Keep only last T samples
        obs = obs.sort_values("Date diff")
        obs = obs.iloc[0:self.T, :]

        # Encode Sero type i.e DENV1 -> 1
        obs['Sero'], _ = pd.factorize(obs['Sero'])
        obs['Sero'] += 1    

        obs['Delay'] = np.log1p(obs['Delay'])
        obs['Date diff'] = np.log1p(obs['Date diff'])

        return np.array(obs)
    
    def get_prop_vec(self, date):
        date = pd.to_datetime(date).to_period(self.period)
        obs = self.prop_dataset.loc[self.prop_dataset['Collection date'] == date].copy()
        obs.drop(columns=['Collection date'], inplace=True)
        return obs.to_numpy().ravel()


class PartialCountDataset:

    def __init__(self, dataset, D, M, norm=True):
        self.dataset = dataset
        self.M = M
        self.D = D
        self.norm = norm
        self.max_val = dataset.select_dtypes(include='number').max().max()
    
    def get_obs(self, date):
        # Get all obs up to and including current day
        date = pd.to_datetime(date)
        obs = self.dataset.loc[self.dataset['Collection date'] <= date].copy()
        obs.drop(columns=['Collection date'], inplace=True)

        # Limit to past M days and max delay of D
        obs = obs.iloc[-self.M:, :self.D]

        # Create mask over last D days
        mask = self.get_mask()
        
        # Create reporting triangle
        obs.loc[obs.iloc[-self.D:].index, :] = obs.iloc[-self.D:, :].where(mask, 0)
        
        if self.norm:
            return np.array(obs) / self.max_val
        
        return np.array(obs)

    def get_mask(self):
        mask_matrix = np.ones(shape=(self.D, self.D), dtype=bool)
        for i in range(self.D):
            for j in range(self.D):
                if i + j > self.D - 1:
                    mask_matrix[i, j] = False
        return mask_matrix

class TrueCountDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_y(self, date):
        date = pd.to_datetime(date)
        y = self.dataset.loc[self.dataset['Collection date'] == date].copy()
        y.drop(columns=['Collection date'], inplace=True)
        return np.array(y.sum(1))

    def get_z(self, date):
        date = pd.to_datetime(date)
        z = self.dataset.loc[self.dataset['Collection date'] == date].copy()
        z.drop(columns=['Collection date'], inplace=True)
        return z.to_numpy().ravel()
    
    def get_y_prop(self, date, prop_vec):
        date = pd.to_datetime(date)
        y = self.get_y(date)
        return (y * prop_vec).round()



if __name__ == "__main__":
    # # Check SeroDataset working as expected
    # denv_df = pd.read_csv(Path("data") / "transformed" / "denv_df.csv")
    # sero_props = pd.read_csv(Path("data") / "transformed" / "sero_props.csv")

    # denv_df['Collection date'] = pd.to_datetime(denv_df['Collection date']).dt.to_period('M')
    # denv_df['Submission date'] = pd.to_datetime(denv_df['Submission date']).dt.to_period('M')



    # sero_props['Collection date'] = pd.to_datetime(sero_props['Collection date']).dt.to_period('M')



    # sero_dataset = SeroDataset(dataset=denv_df, prop_dataset=sero_props, T=10)
    # obs = sero_dataset.get_prop_vec("2016-07-01")
    # print(obs)

    # Check PatialCountDataset working as expected
    delays_df = pd.read_csv(Path("data") / "transformed" / "DENG_delays.csv")
    delays_df['Collection date'] = pd.to_datetime(delays_df['Collection date'])
    deng_dataset = PartialCountDataset(delays_df, D=40, M=50)
    print(deng_dataset.get_obs("2016-07-01"))
    
    # true_count_dataset = TrueCountDataset(delays_df)
    # print(true_count_dataset.get_y_prop("2016-01-01", [0.3, 0.3, 0.3, 0.1]))
