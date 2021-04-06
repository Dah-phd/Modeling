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

    @staticmethod
    def reintegrate(inte, x_legacy, x_integrated):
        """
        If you are using integrted datasets, the fuction easily converts data back to normal set.
        ========================================
        inte(int): number of integrations.
        x_legacy(list of float [newst:oldest]): base for the integrated list, it needs another value for every level of integration.
            If inte=1 needs one datapoint, if inte=2 it will need a scond datapoint (next value).
        x_integrated(list of float [newst:oldest]) the integrated data.
        """
        result = []
        for x in x_integrated[::-1]:
            reints = inte
            base = []
            data = np.array(x_legacy)
            while reints > 1:
                base.append(x_legacy[0])
                data = np.diff(data)
                reints -= 1
            else:
                new_x = sum(base)+x
                x_legacy.pop()
                x_legacy.insert(0, new_x)
                result.insert(0, new_x)
