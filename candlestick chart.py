import mplfinance as fplt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ticker = input("TICKER : ")
start = input("START : ")
end = input("END : ")

data = yf.download(ticker,start,end)
print(data)
mc = fplt.make_marketcolors(
                            up='tab:blue',down='tab:red',
                            wick={'up':'green','down':'red'},
                            volume='lawngreen',
                           )

s  = fplt.make_mpf_style(base_mpl_style="seaborn", marketcolors=mc, mavcolors=["red"])

fplt.plot(
        data,
        type="candle",
        title = ticker + "Chart",
        ylabel='Candlestick',
        mav= (5,20,60),
        volume=True,
        tight_layout = False,
        ylabel_lower='Volume',
        show_nontrading = True,
        width_adjuster_version='v0',
        style='yahoo',
    )
