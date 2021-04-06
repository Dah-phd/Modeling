import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
if __name__ == '__main__':
    import data_tests
else:
    from . import data_tests


# class dummy var regression
# class rolling regression
# class causality Grainger
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
    if validate == 'stack':
        Y = np.array(Y).reshape(-1, 1)
        X = np.array(X)
        shape = X.shape
        X = X.reshape(shape[1], shape[0])
    if alfa:
        add_constant(self.X)
    model = OLS(Y, X, missing='drop' if fix_nan else 'none')
    return model.fit()


def check_X(X, values):
    if isinstance(X[0], list):
        if len(X[0]) == values:
            return 'stack'
    elif len(X) != values:
        return False
    return True
