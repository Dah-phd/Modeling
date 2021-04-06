from modules.predict import ARIMA, AutoReg, MovingAvg, LinearProjection


def csv_go(path, lags=30, col=0, old_new=False):
    """
    Uses the first column of csv, or otherwise specified,
    expects data to be from newest(top) to oldest(bot).
    path (string) req, path to the csv;
    lags (int) optional, specify number of tested lags;
    col (int) optional, spec other column to use;
    old_new (bool) optional, True if data is from oldest to newst.
    """
    from pandas import read_csv
    df = read_csv(path)
    data = list(df[df.columns[col]])
    if old_new:
        data = data[::-1]
    model = ARIMA(data=data, lags=lags)
    model.build()
    print('returns object, object.predict() to see result')
    return model
