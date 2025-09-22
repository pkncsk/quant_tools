from datetime import datetime
import dukascopy_python
import pandas as pd

def fetch_fx(symbol, start: datetime, end: datetime, interval=dukascopy_python.INTERVAL_MIN_5):
    df = dukascopy_python.fetch(symbol, interval, dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
    df.index = df.index.tz_convert('Asia/Bangkok')
    return df
