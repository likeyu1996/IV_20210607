#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Created by Klaus Lee on 2021-06-07
-------------------------------------------------
"""

import datetime
from functools import wraps
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'Data')
RESULT_PATH = os.path.join(ROOT_PATH, 'Result')


def time_decorator(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        end = datetime.datetime.now()
        print(f'“{func.__name__}” run time: {end - start}.')
        return result

    return timer


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


@time_decorator
def easy_read_csv(path, file_name, dedicated_filter=True, **kwargs):
    read_conds = {
        'encoding': 'utf-8',
        'engine': 'python',
        'index_col': None,
        'skiprows': None,
        'na_values': np.nan,
        **kwargs
    }
    data = pd.read_csv(os.path.join(path, file_name), **read_conds)
    if dedicated_filter:
        # 筛选出50ETF和俩300ETF
        filter_list_0 = [i or j or k for i, j, k in zip(data['ContractCode'].str.startswith('510050'),
                                                        data['ContractCode'].str.startswith('510300'),
                                                        data['ContractCode'].str.startswith('159919'))]
        data = data.loc[filter_list_0]
        # 日期格式化
        data['TradingDate'] = pd.to_datetime(data['TradingDate'])
        data['ExerciseDate'] = pd.to_datetime(data['ExerciseDate'])
        # 筛选出行权日为202106和202109的期权
        filter_list_1 = [i and (j or k) for i, j, k in zip(
            data['ExerciseDate'].dt.year == 2021,
            data['ExerciseDate'].dt.month == 6,
            data['ExerciseDate'].dt.month == 9)]
        data = data.loc[filter_list_1]
        # 按日期由近到远排序
        data.sort_values(by='TradingDate', ascending=False, inplace=True)
        # 重置index
        data.reset_index(inplace=True, drop=True)
    return data


database = easy_read_csv(DATA_PATH, 'SO_PricingParameter.csv')
# database.to_csv(os.path.join(RESULT_PATH, 'database.csv'), encoding='gb18030', index=False)


class Data:
    # DB_PATH = os.path.join(DATA_PATH, 'SO_PricingParameter.csv')
    UNDERLYING_DICT = {'sse_50etf': '510050',
                       'sse_300etf': '510300',
                       'szse_300etf': '159919'
                       }

    def __init__(self, date=datetime.datetime(2021, 6, 4), ignore_xd=False):
        self.db_raw = database
        self.date = date
        self.ignore_xd = ignore_xd
        self.sse_50etf, self.sse_300etf, self.szse_300etf = self.__option_filter()
        self.DATA_DICT = {'sse_50etf': self.sse_50etf,
                          'sse_300etf': self.sse_300etf,
                          'szse_300etf': self.szse_300etf
                          }
        self.res_sse_50etf, self.res_sse_300etf, self.res_szse_300etf = self.__implied_volatility()
        self.RESULT_DICT = {'sse_50etf': self.res_sse_50etf,
                            'sse_300etf': self.res_sse_300etf,
                            'szse_300etf': self.res_szse_300etf
                            }

    def __option_filter(self):
        # 按日期筛选
        data = self.db_raw
        date = self.date
        data = data.loc[data['TradingDate'] == date]
        # 按标的物筛选
        sse_50etf = data.loc[data['ContractCode'].str.startswith(self.UNDERLYING_DICT['sse_50etf'])]
        sse_300etf = data.loc[data['ContractCode'].str.startswith(self.UNDERLYING_DICT['sse_300etf'])]
        szse_300etf = data.loc[data['ContractCode'].str.startswith(self.UNDERLYING_DICT['szse_300etf'])]
        # 重置index
        sse_50etf.reset_index(inplace=True, drop=True)
        sse_300etf.reset_index(inplace=True, drop=True)
        szse_300etf.reset_index(inplace=True, drop=True)
        return sse_50etf, sse_300etf, szse_300etf

    @time_decorator
    def __implied_volatility(self):
        result_cache_dict = {}
        for key in self.DATA_DICT:
            # print(key)
            data = self.DATA_DICT[key]
            iv_list = []
            UnderlyingScrtClose = data['UnderlyingScrtClose'].to_numpy()
            StrikePrice = data['StrikePrice'].to_numpy()
            RisklessRate = (data['RisklessRate'] * 0.01).to_numpy()
            TradingDate = data['TradingDate']
            ExerciseDate = data['ExerciseDate']
            Delta_days = (ExerciseDate - TradingDate).dt.days
            Tao = (Delta_days / 365).to_numpy()
            ClosePrice = data['ClosePrice'].to_numpy()
            CallOrPut = np.array([0 if i == 'C' else 1 for i in data['CallOrPut']])
            for i in range(data.index.size):
                s = UnderlyingScrtClose[i]
                k = StrikePrice[i]
                r = RisklessRate[i]
                t = Tao[i]
                o = ClosePrice[i]
                cop = CallOrPut[i]
                sigma_est = 0.2

                # print(s, k, r, t, o, cop, sigma_est)

                # 此处感谢三年前的自己
                def func_bsm(sigma):
                    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
                    d2 = (np.log(s / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
                    return np.array(s * (stats.norm.cdf(d1, 0.0, 1.0) + cop * (-1)) -
                                    k * np.exp(-r * t) * (stats.norm.cdf(d2, 0.0, 1.0) + cop * (-1)) - o)

                iv = optimize.root(func_bsm, np.array(sigma_est), tol=1e-10)
                # print(iv)
                iv_list.append(iv.get('x')[0])
            data['tao'] = Tao
            data['cop'] = CallOrPut
            data['iv'] = iv_list
            result_dict = {'ContractCode': data['ContractCode'].to_numpy(),
                           'ShortName': data['ShortName'].to_numpy(),
                           'CallOrPut': data['CallOrPut'].to_numpy(),
                           'cop': CallOrPut,
                           'ExerciseDate': data['ExerciseDate'].to_numpy(),
                           'TradingDate': data['TradingDate'].to_numpy(),
                           'DeltaDays': Delta_days,
                           't': Tao,
                           'o': ClosePrice,
                           's': UnderlyingScrtClose,
                           'k': StrikePrice,
                           'r': RisklessRate,
                           'iv': iv_list,
                           }
            result_df = pd.DataFrame(result_dict)
            result_df.to_csv(os.path.join(RESULT_PATH, (key + '.csv')), encoding='gb18030', index=False)
            result_cache_dict[key] = result_df
        return result_cache_dict['sse_50etf'], result_cache_dict['sse_300etf'], result_cache_dict['szse_300etf']

    def draw(self):
        for key, data in self.RESULT_DICT.items():
            axis_x = data['k'].to_numpy()
            axis_y = data['iv'].to_numpy()
            delta_x = np.mean(axis_x) / 10.0
            delta_y = np.mean(axis_y) / 10.0
            min_x = 2 if np.min(axis_x) > 2 else np.min(axis_x) - delta_x
            max_x = 8 if np.max(axis_x) < 8 else np.max(axis_x) + delta_x
            min_y = 0 if np.min(axis_y) > 0 else np.min(axis_y) + delta_y
            max_y = 1 if np.max(axis_y) < 1 else np.max(axis_y) + delta_y
            sns.set(rc={'figure.figsize': (16, 9)})
            # plt.xlim(min_x, max_x)
            # plt.ylim(min_y, max_y)
            if self.ignore_xd:
                data = data.loc[~ data['ShortName'].str.startswith('XD')]
                data.reset_index(inplace=True, drop=True)
            sns.lineplot(x='k', y='iv', data=data, hue='CallOrPut', hue_order=['C', 'P'], sort=True, estimator=None)

            plt.xlabel('StrikePrice')
            plt.ylabel('ImpliedVolatility')
            standard = 'standard' if self.ignore_xd else 'all'
            plt.title('Vol_Curve of {0} on {1}({2})'.format(key, self.date.date(), standard))
            plt.savefig(os.path.join(RESULT_PATH, '{1}_{0}({2})'.format(key, self.date.date(), standard)))
            plt.close('all')


custom_date = datetime.datetime(2021, 6, 4)
a = Data(date=custom_date, ignore_xd=True)
a.draw()
