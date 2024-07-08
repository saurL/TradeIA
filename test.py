from YahooDownloaderFile import YahooDownloader
from preprocessor import FeatureEngineer
import config_ticker
import config
import pandas as pd
import itertools
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'
     

data = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_ticker.CAC_40_TICKER).fetch_data()

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = config.INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(data)



list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)
processed_full['date_index'] = processed_full.groupby('date').ngroup()
processed_full.head()
     

processed_full.to_csv('data.csv')