import json
import requests
import pandas as pd
from tqdm import tqdm
from typing import NamedTuple
from datetime import datetime



class Currency_info(NamedTuple):
    names: tuple
    periods: tuple


def Get_maturity_date_from_option_name(option_name: str):
    date_str = option_name.split("-")[1]
    maturity_date = datetime.strptime(date_str, '%d%b%y')
    return maturity_date 


def Get_days_till_maturity(option_name: str) -> int:
    date_str = option_name.split("-")[1]
    maturity_date = datetime.strptime(date_str, '%d%b%y')
    # to format YYYY-MM-DD 00:00:00 
    date_now = datetime.strptime(datetime.today().strftime("%Y-%m-%d"), "%Y-%m-%d")   
    return ((maturity_date - date_now).days)


def Get_options_information(currency_name: str) -> Currency_info:
    """
    Get instrument_name and settlement_period for all options of selected currency from deribit (public)
    """
    request = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + currency_name + "&kind=option")
    request_text = json.loads(request.text)
    currency_info = Currency_info(tuple(pd.json_normalize(request_text['result'])['instrument_name']),
                                  tuple(pd.json_normalize(request_text['result'])['settlement_period']))

    return currency_info


def Get_options_data(currency_name: str) -> pd.DataFrame:
    """
    Get currency data for instrument names
    """
    currency_info = Get_options_information(currency_name)
    progress_bar = tqdm(total = len(currency_info.names))
    currency_data = []
    for i in range(len(currency_info.names)):
        request = requests.get('https://test.deribit.com/api/v2/public/get_order_book?instrument_name=' + currency_info.names[i])
        request_text = json.loads(request.text)
        add_data = pd.json_normalize(request_text['result'])
        add_data['settlement_period'] = currency_info.periods[i]
        add_data['option_name'] = currency_info.names[i]
        add_data['strike'] = int(currency_info.names[i].split("-")[2])
        add_data['maturity_date'] = Get_maturity_date_from_option_name(currency_info.names[i])
        add_data['days_till_maturity'] = Get_days_till_maturity(currency_info.names[i])        
        currency_data.append(add_data)
        progress_bar.update(1)
    currency_dataframe = pd.concat(currency_data)
    """currency_dataframe = currency_dataframe[['underlying_price', 'timestamp', 'settlement_price',
                        'open_interest', 'min_price', 'max_price', 'mark_price', 'mark_iv',
                        'last_price', 'interest_rate', 'instrument_name', 'index_price', 'bids',
                        'bid_iv', 'best_bid_price', 'best_bid_amount','best_ask_price', 
                        'best_ask_amount', 'asks', 'ask_iv', 'stats.volume','stats.price_change',
                        'stats.low', 'stats.high','settlement_period','strike', 'option_name']].copy"""
    currency_dataframe = currency_dataframe[['instrument_name', 'underlying_price', 'timestamp', 'mark_price', 'mark_iv',
                         'best_bid_price', 'best_ask_price', 'strike', 'option_name', 'maturity_date', 'days_till_maturity']].copy()                    
    progress_bar.close()
    return currency_dataframe



