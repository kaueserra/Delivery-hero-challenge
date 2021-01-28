import pandas as pd


def day_time_func(x):
    try:
        h = int(x)
        if h <= 6:
            h_t = 'night'
        elif h >6 and h <=12:
            h_t = 'morning'
        elif h >12 and h <=18:
            h_t = 'afternoon'
        else:
            h_t = 'evening'
        return h_t
    except:
        return pd.NaT