### This code was heavily inspired from the open source repositories:
# https://github.dev/AI4Finance-Foundation/FinRL
# and https://github.com/ranaroussi/yfinance
# for which we are grateful.

import copy
import os
import urllib
import zipfile
from datetime import *
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import stockstats


from tsExperiments.stock_market_data.config import BINANCE_BASE_URL
from tsExperiments.stock_market_data.config import TIME_ZONE_BERLIN
from tsExperiments.stock_market_data.config import TIME_ZONE_JAKARTA
from tsExperiments.stock_market_data.config import TIME_ZONE_PARIS
from tsExperiments.stock_market_data.config import TIME_ZONE_SELFDEFINED
from tsExperiments.stock_market_data.config import TIME_ZONE_SHANGHAI
from tsExperiments.stock_market_data.config import TIME_ZONE_USEASTERN
from tsExperiments.stock_market_data.config import USE_TIME_ZONE_SELFDEFINED
from tsExperiments.stock_market_data.config_tickers import CAC_40_TICKER
from tsExperiments.stock_market_data.config_tickers import CSI_300_TICKER
from tsExperiments.stock_market_data.config_tickers import DAX_30_TICKER
from tsExperiments.stock_market_data.config_tickers import DOW_30_TICKER
from tsExperiments.stock_market_data.config_tickers import HSI_50_TICKER
from tsExperiments.stock_market_data.config_tickers import LQ45_TICKER
from tsExperiments.stock_market_data.config_tickers import MDAX_50_TICKER
from tsExperiments.stock_market_data.config_tickers import NAS_100_TICKER
from tsExperiments.stock_market_data.config_tickers import SDAX_50_TICKER
from tsExperiments.stock_market_data.config_tickers import SP_500_TICKER
from tsExperiments.stock_market_data.config_tickers import SSE_50_TICKER
from tsExperiments.stock_market_data.config_tickers import TECDAX_TICKER


class _Base:
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        self.data_source: str = data_source
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.time_interval: str = time_interval  # standard time_interval
        # transferred_time_interval will be supported in the future.
        # self.nonstandard_time_interval: str = self.calc_nonstandard_time_interval()  # transferred time_interval of this processor
        self.time_zone: str = ""
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.dictnumpy: dict = (
            {}
        )  # e.g., self.dictnumpy["open"] = np.array([1, 2, 3]), self.dictnumpy["close"] = np.array([1, 2, 3])

    def download_data(self, ticker_list: List[str]):
        pass

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"datetime": "time"}, inplace=True)
        if self.data_source == "ccxt":
            self.dataframe.rename(columns={"index": "time"}, inplace=True)

        if self.data_source == "ricequant":
            """RiceQuant data is already cleaned, we only need to transform data format here.
            No need for filling NaN data"""
            self.dataframe.rename(columns={"order_book_id": "tic"}, inplace=True)
            # raw df uses multi-index (tic,time), reset it to single index (time)
            self.dataframe.reset_index(level=[0, 1], inplace=True)
            # check if there is NaN values
            assert not self.dataframe.isnull().values.any()
        elif self.data_source == "baostock":
            self.dataframe.rename(columns={"code": "tic"}, inplace=True)

        self.dataframe.dropna(inplace=True)
        # adjusted_close: adjusted close price
        if "adjusted_close" not in self.dataframe.columns.values.tolist():
            self.dataframe["adjusted_close"] = self.dataframe["close"]
        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe = self.dataframe[
            [
                "tic",
                "time",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ]

    def fillna(self):
        df = self.dataframe

        dfcode = pd.DataFrame(columns=["tic"])
        dfdate = pd.DataFrame(columns=["time"])

        dfcode.tic = df.tic.unique()
        dfdate.time = df.time.unique()
        dfdate.sort_values(by="time", ascending=False, ignore_index=True, inplace=True)

        try:
            df1 = pd.merge(dfcode, dfdate, how="cross")
        except:
            print("Please wait for a few seconds...")
            df1 = pd.DataFrame(columns=["tic", "time"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1 = df1.append(
                        pd.DataFrame(
                            data={
                                "tic": dfcode.iat[i, 0],
                                "time": dfdate.iat[j, 0],
                            },
                            index=[(i + 1) * (j + 1) - 1],
                        )
                    )

        df = pd.merge(df1, df, how="left", on=["tic", "time"])

        # back fill missing data then front fill
        df_new = pd.DataFrame(columns=df.columns)
        for i in df.tic.unique():
            df_tmp = df[df.tic == i].fillna(method="bfill").fillna(method="ffill")
            df_new = pd.concat([df_new, df_tmp], ignore_index=True)

        df_new = df_new.fillna(0)

        # reshape dataframe
        df_new = df_new.sort_values(by=["time", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", df_new.shape)

        self.dataframe = df_new

    def get_trading_days(self, start: str, end: str) -> List[str]:
        if self.data_source in [
            "binance",
            "ccxt",
            "quantconnect",
            "ricequant",
            "tushare",
        ]:
            print(
                f"Calculate get_trading_days not supported for {self.data_source} yet."
            )
            return None

    # standard_time_interval  s: second, m: minute, h: hour, d: day, w: week, M: month, q: quarter, y: year
    # output time_interval of the processor
    def calc_nonstandard_time_interval(self) -> str:
        if self.data_source == "alpaca":
            pass
        elif self.data_source == "baostock":
            # nonstandard_time_interval: 默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。
            pass
            time_intervals = ["5m", "15m", "30m", "60m", "1d", "1w", "1M"]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            if (
                "d" in self.time_interval
                or "w" in self.time_interval
                or "M" in self.time_interval
            ):
                return self.time_interval[-1:].lower()
            elif "m" in self.time_interval:
                return self.time_interval[:-1]
        elif self.data_source == "binance":
            # nonstandard_time_interval: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
            time_intervals = [
                "1m",
                "3m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "4h",
                "6h",
                "8h",
                "12h",
                "1d",
                "3d",
                "1w",
                "1M",
            ]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            return self.time_interval
        elif self.data_source == "ccxt":
            pass
        elif self.data_source == "iexcloud":
            time_intervals = ["1d"]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            return self.time_interval.upper()
        elif self.data_source == "joinquant":
            # '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'
            time_intervals = [
                "1m",
                "5m",
                "15m",
                "30m",
                "60m",
                "120m",
                "1d",
                "1w",
                "1M",
            ]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            return self.time_interval
        elif self.data_source == "quantconnect":
            pass
        elif self.data_source == "ricequant":
            #  nonstandard_time_interval: 'd' - 天，'w' - 周，'m' - 月， 'q' - 季，'y' - 年
            time_intervals = ["d", "w", "M", "q", "y"]
            assert self.time_interval[-1] in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            if "M" in self.time_interval:
                return self.time_interval.lower()
            else:
                return self.time_interval
        elif self.data_source == "tushare":
            # 分钟频度包括1分、5、15、30、60分数据. Not support currently.
            # time_intervals = ["1m", "5m", "15m", "30m", "60m", "1d"]
            time_intervals = ["1d"]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            return self.time_interval
        elif self.data_source == "wrds":
            pass
        elif self.data_source == "yahoofinance":
            # nonstandard_time_interval: ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d","1wk", "1mo", "3mo"]
            time_intervals = [
                "1m",
                "2m",
                "5m",
                "15m",
                "30m",
                "60m",
                "90m",
                "1h",
                "1d",
                "5d",
                "1w",
                "1M",
                "3M",
            ]
            assert self.time_interval in time_intervals, (
                "This time interval is not supported. Supported time intervals: "
                + ",".join(time_intervals)
            )
            if "w" in self.time_interval:
                return self.time_interval + "k"
            elif "M" in self.time_interval:
                return self.time_interval[:-1] + "mo"
            else:
                return self.time_interval
        else:
            raise ValueError(
                f"Not support transfer_standard_time_interval for {self.data_source}"
            )

    # "600000.XSHG" -> "sh.600000"
    # "000612.XSHE" -> "sz.000612"
    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        return ticker

    def save_data(self, path):
        if ".csv" in path:
            path = path.split("/")
            filename = path[-1]
            path = "/".join(path[:-1] + [""])
        else:
            if path[-1] == "/":
                filename = "dataset.csv"
            else:
                filename = "/dataset.csv"

        os.makedirs(path, exist_ok=True)
        self.dataframe.to_csv(path + filename, index=False)

    def load_data(self, path):
        assert ".csv" in path  # only support csv format now
        self.dataframe = pd.read_csv(path)
        columns = self.dataframe.columns
        print(f"{path} loaded")

def calc_time_zone(
    ticker_list: List[str],
    time_zone_selfdefined: str,
    use_time_zone_selfdefined: int,
) -> str:
    assert isinstance(ticker_list, list)
    ticker_list = ticker_list[0]
    if use_time_zone_selfdefined == 1:
        time_zone = time_zone_selfdefined
    elif ticker_list in HSI_50_TICKER + SSE_50_TICKER + CSI_300_TICKER:
        time_zone = TIME_ZONE_SHANGHAI
    elif ticker_list in DOW_30_TICKER + NAS_100_TICKER + SP_500_TICKER:
        time_zone = TIME_ZONE_USEASTERN
    elif ticker_list == CAC_40_TICKER:
        time_zone = TIME_ZONE_PARIS
    elif ticker_list in DAX_30_TICKER + TECDAX_TICKER + MDAX_50_TICKER + SDAX_50_TICKER:
        time_zone = TIME_ZONE_BERLIN
    elif ticker_list == LQ45_TICKER:
        time_zone = TIME_ZONE_JAKARTA
    else:
        # hack needed to have this working with vix indicator
        # fix: unable to set time_zone_selfdefined from top-level dataprocessor class
        time_zone = TIME_ZONE_USEASTERN
        # raise ValueError("Time zone is wrong.")
    return time_zone


def check_date(d: str) -> bool:
    assert (
        len(d) == 10
    ), "Please check the length of date and use the correct date like 2020-01-01."
    indices = [0, 1, 2, 3, 5, 6, 8, 9]
    correct = True
    for i in indices:
        if not d[i].isdigit():
            correct = False
            break
    if not correct:
        raise ValueError("Please use the correct date like 2020-01-01.")
    return correct
