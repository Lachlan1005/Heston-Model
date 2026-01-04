import yfinance as yf

stock = yf.Ticker("TSLA")
A = stock.fast_info["currency"]

print(A)