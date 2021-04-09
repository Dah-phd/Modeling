import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
if __name__ == '__main__':
    import data_tests
else:
    from . import data_tests


class causality:
    def __init__(self, Y, X):
        """
        Solves Grainger causality test to derive if fariable is causing the change in another.
        Data should start from the newst to the oldest observation.
            Y [list of float or numpy array] - the varaible row that is to be affcted;
            X [list of float or numpy array] - the factor variable row;
        """
        self.Y = Y
        self.X = X

    def fit(self, test_lags=5, integrate=True, reverse=False):
        """
        Fits the data to the model.
            test_lags [int, default=5] - lags to test for causality;
            integrate [boolean, default=True] - integrates data to reach stationarity;
            reverse [boolean, default=False] - test reverse causality.
        """
        self.fix_data()
        if integrate:
            self.integrations, self.Y, self.X = data_tests.stationarity.forceSTATxy(
                self.Y, self.X)
            if not self.integrations:
                print(self.Y)
                return
            print(self.integrations)
        self.result = self.build(test_lags)
        if reverse:
            self.X, self.Y = self.Y, self.X
            self.result_reverse = self.build(test_lags)
            self.Y, self.X = self.X, self.Y

    def build(self, test_lags):
        for lag in range(1, test_lags+1):
            lag_counts = lag
            passed_lags = 0
            print(self.Y)
            base = np.copy(self.Y)[:-lag]
            auto_regressor = np.copy(self.Y)[lag:]
            regressor = np.hstack((auto_regressor, self.X[lag:]))
            while lag_counts > 1:
                lag_counts -= 1
                passed_lags -= 1
                auto_regressor = np.hstack(
                    (auto_regressor, self.Y[lag_counts:passed_lags]))
                regressor = np.hstack(
                    (regressor, self.Y[lag_counts:passed_lags], self.X[lag_counts:passed_lags]))
            # auto_regressor is first!!!
            # auto_model = OLS(base, auto_regressor, missing='drop')
            # auto_model.fit()
            # full_model = OLS(base, regressor, missing='drop')
            # full_model.fit()
            print('======', lag, '======')
            print('Y', base.shape)
            print('Y-auto', auto_regressor.shape)
            print('X', regressor.shape)
            print()

    def fix_data(self):
        self.X = np.array(self.X).reshape(-1, 1)
        self.Y = np.array(self.Y).reshape(-1, 1)


def linear_regression(Y, X, alfa=True, fix_nan=True):
    """
    Using a package, it rebuild here for ease of use.
    It is 100% statsmodels OLS.
    Y (list[float]) - dependent variable;
    X ([list[float]|float]) - independent variable;
    """
    valid = check_X(X, len(Y))
    if not valid:
        return False
    if valid == 'stack':
        Y = np.array(Y).reshape(-1, 1)
        X = np.array(X)
        shape = X.shape
        X = X.reshape(shape[1], shape[0])
    if alfa:
        add_constant(X)
    model = OLS(Y, X, missing='drop' if fix_nan else 'none')
    return model.fit()


def check_X(X, values):
    if isinstance(X[0], list):
        if len(X[0]) == values:
            return 'stack'
    elif len(X) != values:
        return False
    return True

    # class dummy var regression
    # class rolling regression
