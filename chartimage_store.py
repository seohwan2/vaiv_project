import cv2
from matplotlib.pyplot import savefig, tight_layout
import mplfinance as fplt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ticker = input("TICKER : ")
start = input("START : ")
end = input("END : ")
mav = []
num = 1
while(num):
    temp = int(input("mav(Enter 0 if you want end) : "))
    if temp == 0:
        break;
    mav.append(temp)

mavtuple = tuple(mav)
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
        title = "Samsung Chart",
        ylabel='Candlestick',
        mav=mavtuple,
        volume=True,
        tight_layout = False,
        ylabel_lower='Volume',
        show_nontrading = True,
        width_adjuster_version='v0',
        style='yahoo',
        savefig = 'fig.png'
    )
