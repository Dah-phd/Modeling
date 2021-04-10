try:
    from TS_analysis.modules.predict import ARIMA, AutoReg, MovingAvg, LinearProjection
    from TS_analysis.modules.regression import causality
except Exception:
    from modules.predict import ARIMA, AutoReg, MovingAvg, LinearProjection
    from modules.regression import causality

from pandas import read_csv, DataFrame


"""
bin is contains all, it includes both predict and regression modules also some extra functions allowing
easier connections to the modules. Still, more resource efficient is the usage of TS_analysis.modules and
then import what is necessery.
"""


def csv_to_predict(path, model, lags=30, col=0, old_new=False):
    """
    Uses the first column of csv, or otherwise specified,
    expects data to be from newest(top) to oldest(bot).
        model, string - ARIMA, AR(AutoRegression), MA(Moving Averages), LP(Linear Projection);
        path, string - req, path to the csv;
        lags (int) optional, specify number of tested lags;
        col (int) optional, spec other column to use;
        old_new (bool) optional, True if data is from oldest to newst.
    """
    df = read_csv(path)
    data = list(df[df.columns[col]])
    if old_new:
        data = data[::-1]
    model = ARIMA(data=data, lags=lags)
    model = AutoReg(data=data, lags=lags)
    model = MovingAvg(data=data, lags=lags)
    model = LinearProjection(data=data)
    model.build()
    model.predict()
    if verb:
        print(model.prediction)
    return model


def csvs_to_causality(path_Y, path_X, col_Y=0, col_X=0, test_lags=7, result_df=False, verb=True):
    df_Y = read_csv(path_Y)
    Y = list(df_Y[df_Y.columns[col_Y]])
    df_X = read_csv(path_X)
    X = list(df_X[df_X.columns[col_X]])
    model = _causality(Y, X, test_lags, verb)
    if model == None:
        return 'Data cannot be processed!'
    if result_df:
        return DataFrame(model.result)
    else:
        return model


def csv_to_causality(path,  col=(0, 1), test_lags=7, reverse=False, result_df=False, verb=True):
    """
    col(tuple, int) - first is index of Y column (result), next index of X column (factor);
    """
    df = read_csv(path)
    Y = list(df[df.columns[0]])
    X = list(df[df.columns[1]])
    model = _causality(Y, X, test_lags, verb)
    if model == None:
        return 'Data cannot be processed!'
    if result_df:
        return DataFrame(model.result)
    else:
        return model


def _causality(Y, X, test_lags, verb):
    model = causality(Y, X)
    model.fit(test_lags=test_lags)
    print(model.X)
    print(model.Y)
    if model.integrations == None:
        print('COULD NOT INTEGRATE DATA!!')
        return None
    if verb:
        print(model.result)
        if reverse:
            print()
            print('REVERSED')
            print(model.reversed_xy)
    return model


if __name__ == '__main__':
    a = csvs_to_causality(
        path_Y='/home/dah/Documents/3_python_projects/0_ETF_data/used/DAX.csv',
        path_X='/home/dah/Documents/3_python_projects/0_ETF_data/used/DAX.csv',
        col_Y=1, col_X=1
    )
    print(a)
