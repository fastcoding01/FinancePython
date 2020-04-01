import datetime as dt
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import style
import pandas as pd
import numpy as np
import os
import pandas_datareader.data as web
import pandas_datareader.data as pdr
import bs4 as bs
import pickle
import requests
import fix_yahoo_finance as yf

# style of graphic
style.use('ggplot')

# graph type
grapht_type = int(input('enter the type of graph(1:,2:,3:)'))

# start/end time of stocks taken
start = dt.datetime(2017,1,1)
end = dt.datetime.now()

# FOR 100MA
def show_100ma(stocks):
    ax1.plot(df.index,df['Adj Close'])
    ax1.plot(df.index,df['100ma'])
    ax2.bar(df.index,df['Volume'])
    plt.title('{}'.format(stocks))
    plt.show()

# FOR OHLC
def show_OHLC(df_ohlc,df_volume,stocks):
    candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup = 'g')
    ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
    plt.title('{}'.format(stocks))
    plt.show()
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("/Users/fedecech/Finance Python/sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("/Users/fedecech/Finance Python/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('/Users/fedecech/Finance Python/stock_dfs'):
        os.makedirs('/Users/fedecech/Finance Python/stock_dfs')
    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('/Users/fedecech/Finance Python/stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('/Users/fedecech/Finance Python/stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
            
def compile_data():
    with open("/Users/fedecech/Finance Python/sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('/Users/fedecech/Finance Python/stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('/Users/fedecech/Finance Python/sp500_joined_closes.csv')

def visualize_data():
    df = pd.read_csv('/Users/fedecech/Finance Python/sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('/Users/fedecech/Finance Python/sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

if grapht_type == 1:
    stocks = input('Enter stock name :')
    df = web.DataReader(stocks,'yahoo',start,end)
    df.to_csv('/Users/fedecech/Finance Python/{}.csv'.format(stocks))
    df = pd.read_csv('/Users/fedecech/Finance Python/{}.csv'.format(stocks), parse_dates = True, index_col=0)
    df ['100ma'] = df['Adj Close'].rolling(window = 100,min_periods=0).mean()
    df.dropna(inplace=True)  
    ax1 = plt.subplot2grid((6,1),(0,0),rowspan = 5,colspan = 1)
    ax2 = plt.subplot2grid((6,1),(5,0),rowspan = 1,colspan = 1, sharex = ax1)
    ax1.xaxis_date()
    show_100ma(stocks)

elif grapht_type == 2:
    stocks = input('Enter stock name :')
    df = web.DataReader(stocks,'yahoo',start,end)
    df.to_csv('/Users/fedecech/Finance Python/{}.csv'.format(stocks))
    df = pd.read_csv('/Users/fedecech/Finance Python/{}.csv'.format(stocks), parse_dates = True, index_col=0)
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    df_volume = df['Volume'].resample('10D').sum()
    df_ohlc.reset_index(inplace = True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    ax1 = plt.subplot2grid((6,1),(0,0),rowspan = 5,colspan = 1)
    ax2 = plt.subplot2grid((6,1),(5,0),rowspan = 1,colspan = 1, sharex = ax1)
    ax1.xaxis_date()
    show_OHLC(df_ohlc,df_volume,stocks)

elif grapht_type == 3:
    yf.pdr_override
    save_sp500_tickers()
    get_data_from_yahoo()
    compile_data()
    visualize_data()
else:
    print('No type aviable')













