### This code was heavily inspired from the open source repositories:
# https://github.dev/AI4Finance-Foundation/FinRL
# and https://github.com/ranaroussi/yfinance
# for which we are grateful.

import os, sys
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"],".."))

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from tsExperiments.stock_market_data.data_processors.yahoofinance import Yahoofinance
from tsExperiments.stock_market_data.config_tickers import DOW_30_TICKER
import pandas as pd 
import numpy as np 
from typing import List 
from gluonts.dataset.common import MetaData, CategoricalFeatureInfo

class YahooFinanceDataset:
    def __init__(self,start_date:str,
                 end_date:str,
                 time_interval:str="1d",
                 adress_to_save_file:str="./data/dataset.csv",
                 ticker_list:List[str] = DOW_30_TICKER,
                 update_index = False,
                 load_from_csv = False,
                 load_from_csv_path = None
                 ):

        self.target_dim = len(ticker_list)        
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.adress_to_save_file = adress_to_save_file 
        self.ticker_list = ticker_list
        self.loaderYhaoo = Yahoofinance("yahoofinance",
                                        start_date=self.start_date,
                                        end_date=self.end_date,time_interval=self.time_interval)      
        self.loaderYhaoo.download_data(ticker_list=self.ticker_list,save_path=self.adress_to_save_file,load_from_csv=load_from_csv,load_from_csv_path=load_from_csv_path)
        self.data = self.loaderYhaoo.dataframe
        self.data_for_analyse=self.data
        self.data = self.data.set_index("date")

        if self.time_interval == "1d" and update_index:
            common_index = self.data[self.data["tic"]==self.data["tic"].iloc[0]].index 
           
            # Group by "tic" en excluant les colonnes de groupement de l'opération apply pour éviter le DeprecationWarning.
            for item_id, gdf in self.data.groupby("tic", group_keys=False):
                    assert (gdf.index == common_index).min(), f"error {item_id}: you don't have the same index for all the dates, we can't re-adjust dates"
            self.old_index = self.data.index 
            self.data = self._update_to_continuous_index(self.data,str(self.data.index[0]))
            self.new_index = self.data.index 

        if self.time_interval == "60m":
            self.time_interval = "h"

    def analyze_assets(self):
        df =self.data_for_analyse
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['tic', 'date'])  
        df['return'] = df.groupby('tic')['close'].pct_change()
           
        asset_stats = df.groupby('tic').agg(
            min_price=('close', 'min'),
            max_price=('close', 'max'),
            data_points=('close', 'count')
        ).reset_index()      
        pivot_returns = df.pivot(index='date', columns='tic', values='return')
        corr_matrix = pivot_returns.corr()
        
        return asset_stats, corr_matrix

    def _update_to_continuous_index(self,df, start_date="2000-01-03"):
        """
        Replace the dataframe's datetime index with a continuous daily index.
        
        For each unique date in the original index, a new date is generated, starting from `start_date`
        and increasing daily. Then, the original index (even with duplicates) is mapped to these new dates,
        so that the unique dates become continuous.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame with a datetime index (possibly with duplicates and gaps).
            start_date (str): The starting date for the new continuous index (default "2000-01-03").
        
        Returns:
            pd.DataFrame: A new DataFrame with the updated continuous datetime index.
        """
        # Extract the unique dates from the current index and sort them
        unique_dates = df.index.unique().sort_values()
        
        # Generate a continuous date range with the same number of unique dates
        continuous_dates = pd.date_range(start=start_date, periods=len(unique_dates), freq="D")
        
        # Create a mapping from the old unique dates to the new continuous dates
        date_mapping = dict(zip(unique_dates, continuous_dates))
        
        # Replace each date in the index with its corresponding new date
        new_index = df.index.map(lambda d: date_mapping.get(d, d))
        
        # Optionally, create a copy of the DataFrame to avoid modifying the original data
        df_new = df.copy()
        df_new.index = new_index
        
        return df_new
  
    def _converting_data_to_usable_dataset(self, data: pd.DataFrame):
        # Drop unnecessary columns
        data = data.drop(columns=['open', 'volume', 'adjusted_close', 'high', 'low', 'day'])
        common_index = data[data["tic"]==self.data["tic"].iloc[0]].index 
        dfs_dict = {}
        # Group by "tic" by excluding the grouping columns from the apply operation to avoid the DeprecationWarning.
        for item_id, gdf in data.groupby("tic", group_keys=False):
            if self.time_interval == "1d":
                assert (gdf.index == common_index).min(), "error: you don't have the same index for all the dates"
            # Use each group's own date range (from its first date to its last date)
            new_index = pd.date_range(start=gdf.index[0], end=gdf.index[-1], freq=self.time_interval)
            # Reindex the group, then forward-fill missing values, then drop the 'tic' column
            if self.time_interval =="h":
                nb_gaps, total_number_of_hour = self._check_hourly_index(gdf)
                print(f"The total number of gap in {item_id} is {nb_gaps} corresponding of a number of hours of {total_number_of_hour} on the {len(gdf)} total number of hours. We do a fill forward.")
            elif self.time_interval == "1d":
                nb_gaps, total_number_of_days = self._check_daily_index(gdf)
                print(f"The total number of gap in {item_id} is {nb_gaps} corresponding of a number of days of {total_number_of_days} on the {len(gdf)} total number of days. We do a fill forward.")
            min_price = gdf["close"].min()
            max_price = gdf["close"].max()
            print(f"Minimum level of price : {min_price}, maximum level of price: {max_price}")
            dfs_dict[item_id] = gdf.reindex(new_index).ffill().drop("tic", axis=1) #gdf.reindex(new_index).ffill().drop("tic", axis=1)
        
        ds = PandasDataset(dfs_dict, target="close")
        if len(dfs_dict)!= self.target_dim:
            raise ValueError("Issue with target dim of the ticker.")
        train_grouper = MultivariateGrouper(max_target_dim=len(dfs_dict))
        dataset_train = train_grouper(ds)
        return dataset_train

    def creating_train_or_val_dataset(self,start_date:str,end_date:str):
        data = self.data
        data = data.loc[start_date:end_date]

        return self._converting_data_to_usable_dataset(data)
    
    def creating_test_dataset(self,start_date:str,end_date:str,num_tests:int):
        """splitting between start_date and end_date in num_tests parts, and returning
        the test_dataset object you can directly use in you code, just like for electricity etc"""
        data = self.data
        data = data.loc[start_date:end_date]
        unique_dates = np.sort(data.index.unique())
        # Split the data into num_tests approximately equal parts using numpy's array_split.
        #segments = np.array_split(data, num_tests)
        date_groups = np.array_split(unique_dates, num_tests)
        # Convert each segment into a usable dataset object.
        out = []
     
        for group in date_groups:
          
            segment = data.loc[data.index.isin(group)]
            out += self._converting_data_to_usable_dataset(segment)
                
        return out 
    
    def generating_metaData(self):
        metadata = MetaData(
            freq=self.time_interval,  # sets freq field via alias
            feat_static_cat=[
                CategoricalFeatureInfo(name="stock markets", cardinality=self.target_dim),
            ],
            prediction_length=24 #arbitrary, associated with the dataset.
                )
        return metadata #the metadat associated with the dataset.
    
    def _check_hourly_index(self,df: pd.DataFrame):
        """just checking if there is any gap in the data
        return the number of "gaps", and the total number of hours missing/the total"""
      
        diffs = df.index.to_series().diff().dropna()
        
        gap_hours = (diffs / pd.Timedelta(hours=1) - 1).astype(int)

        gaps = gap_hours[gap_hours > 0]
        total_missing = gaps.sum()
        return len(gaps), total_missing
    def _check_daily_index(self, df: pd.DataFrame):
        """
        Check if there are gaps in the data indexed by days.
        Return the number of "gaps" and the total number of missing days.
        """
        # Calculate the differences between successive dates
        diffs = df.index.to_series().diff().dropna()
        
        # Calculate the number of missing days for each gap.
        # Divide by 1 day and subtract 1 to ignore the expected step.
        gap_days = (diffs / pd.Timedelta(days=1) - 1).astype(int)
        
        # Keep only the effective gaps (where the number of missing days is > 0)
        gaps = gap_days[gap_days > 0]
        total_missing = gaps.sum()
        
        return len(gaps), total_missing