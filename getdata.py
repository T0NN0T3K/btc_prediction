import yfinance as yf
import ta
import requests
import pandas as pd

# Scarica i dati storici di BTC
btc = yf.Ticker("BTC-USD")
close_data = btc.history(period="max")[['Close', 'Volume']]
close_data.rename(columns={'Close': 'BTC_Close', 'Volume': 'BTC_Volume'}, inplace=True)

# Rimuove il fuso orario dall'indice di close_data
close_data.index = close_data.index.tz_localize(None)
close_data.reset_index(inplace=True)
close_data.rename(columns={'Date': 'timestamp'}, inplace=True)

# Calcola l'RSI a 14 giorni
rsi = ta.momentum.RSIIndicator(close_data['BTC_Close'], window=14).rsi()
close_data['RSI_14d'] = rsi

# Scarica il Fear & Greed Index
data = requests.get("https://api.alternative.me/fng/?limit=0").json()
df_fgi = pd.DataFrame(data["data"])
df_fgi["timestamp"] = pd.to_datetime(pd.to_numeric(df_fgi["timestamp"], errors='coerce'), unit="s")
df_fgi["timestamp"] = df_fgi["timestamp"].dt.tz_localize(None)
df_fgi.sort_values(by="timestamp", inplace=True)
df_fgi.rename(columns={'value': 'Fear_Greed_Index'}, inplace=True)
df_fgi['Fear_Greed_Index'] = pd.to_numeric(df_fgi['Fear_Greed_Index'], errors='coerce')

# Scarica i dati di Hash Rate e Mining Difficulty da Blockchain.com
url_hashrate = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"
url_difficulty = "https://api.blockchain.info/charts/difficulty?timespan=all&format=json"

data_hashrate = requests.get(url_hashrate).json()
data_difficulty = requests.get(url_difficulty).json()

df_hashrate = pd.DataFrame(data_hashrate["values"])
df_difficulty = pd.DataFrame(data_difficulty["values"])

df_hashrate["timestamp"] = pd.to_datetime(df_hashrate["x"], unit="s")
df_difficulty["timestamp"] = pd.to_datetime(df_difficulty["x"], unit="s")

df_hashrate.rename(columns={'y': 'Hash_Rate'}, inplace=True)
df_difficulty.rename(columns={'y': 'Mining_Difficulty'}, inplace=True)

df_hashrate.drop(columns=['x'], inplace=True, errors='ignore')
df_difficulty.drop(columns=['x'], inplace=True, errors='ignore')

df_fgi.drop(columns=['x', 'y','time_until_update','value_classification'], inplace=True, errors='ignore')
close_data.drop(columns=['x', 'y','time_until_update','value_classification'], inplace=True, errors='ignore')

# Scarica i dati di USDX (DXY), EFFR e S&P 500, GOLD
dxy = yf.Ticker("DX-Y.NYB").history(start="2009-01-03")[['Close']]
dxy.rename(columns={'Close': 'USDX'}, inplace=True)
dxy.index = dxy.index.tz_localize(None)
dxy.reset_index(inplace=True)
dxy.rename(columns={'Date': 'timestamp'}, inplace=True)

sp500 = yf.Ticker("^GSPC").history(start="2009-01-03")[['Close']]
sp500.rename(columns={'Close': 'S&P500'}, inplace=True)
sp500.index = sp500.index.tz_localize(None)
sp500.reset_index(inplace=True)
sp500.rename(columns={'Date': 'timestamp'}, inplace=True)

effr = yf.Ticker("^IRX").history(start="2009-01-03")[['Close']]
effr.rename(columns={'Close': 'EFFR'}, inplace=True)
effr.index = effr.index.tz_localize(None)
effr.reset_index(inplace=True)
effr.rename(columns={'Date': 'timestamp'}, inplace=True)

gold = yf.Ticker("GC=F").history(start="2009-01-03")[['Close']]
gold.rename(columns={'Close': 'Gold'}, inplace=True)
gold.index = gold.index.tz_localize(None)
gold.reset_index(inplace=True)
gold.rename(columns={'Date': 'timestamp'}, inplace=True)

# Merge dei dati
final_df = close_data.merge(df_fgi, on="timestamp", how='outer')
final_df = final_df.merge(df_hashrate, on="timestamp", how='outer')
final_df = final_df.merge(df_difficulty, on="timestamp", how='outer')
final_df = final_df.merge(dxy, on="timestamp", how='outer')
final_df = final_df.merge(sp500, on="timestamp", how='outer')
final_df = final_df.merge(effr, on="timestamp", how='outer')
final_df = final_df.merge(gold, on="timestamp", how='outer')

# Rimuove eventuali colonne duplicate
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Filtra i dati rimuovendo le righe fino al 16/09/2014
cutoff_date = pd.to_datetime("2014-09-17")
final_df = final_df[final_df['timestamp'] >= cutoff_date]

# Salva il dataset finale
final_df.to_csv("btc_input_data.csv", index=False)

print("Dati combinati salvati")
