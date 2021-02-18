from statsmodels.tsa.stattools import adfuller
import numpy as np


class stationarity:
    @staticmethod
    # Augmented DF test
    def ADF(x):
        result = adfuller(x)
        if result[1] < 0.05 and result[4]['5%'] > result[0]:
            return True
        else:
            return False

    @staticmethod
    # performs simple integration
    def integration(x):
        x = np.array(x)
        x = np.diff(x)
        return x

    @staticmethod
    # reaches stationarity
    def forceSTAT(x, n_integrations=True):
        inte = 0
        x = np.array(x)
        stat = stationarity.ADF(x)
        while not stat and inte < 6:
            inte += 1
            x = stationarity.integration(x)
            stat = stationarity.ADF(x)
        else:
            if stat and n_integrations:
                return (inte, x)
            elif stat:
                return x
            else:
                return False, None

    @staticmethod
    def forceSTATxy(x, y, n_integrations=True):
        inte = 0
        x = np.array(x)
        stat_x = stationarity.ADF(x)
        y = np.array(y)
        stat_y = stationarity.ADF(y)
        while not stat_x and not stat_y and inte < 6:
            inte += 1
            x = stationarity.integration(x)
            y = stationarity.integration(y)
            stat_x = stationarity.ADF(x)
            stat_y = stationarity.ADF(y)
        else:
            if stat_x and stat_y and n_integrations:
                return (inte, x, y)
            elif stat_x and stat_y:
                return(x, y)
            else:
                return False, None, None
