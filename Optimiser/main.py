
import numpy as np
import scipy.optimize as opt
import quant_risk as qr
import pandas as pd

class objectiveFunction:

    def sharpe(weights: np.array, prices: pd.DataFrame, **kwargs):
        """Objective function that returns the negative sharpe ratio (since it is a minimisation problem), given a price dataframe.
        Additional arguements can be provided such as riskFreeRate and periodsPerYear to account for annualisation etc.

        Parameters
        ----------
        weights : np.array
            numpy array of our weights that will be optimised
        prices : pd.DataFrame
            Dataframe of prices of our tickers with dates as the index

        Returns
        -------
        float
            Returns the negative of the sharpe ratio
        """

        portfolio = (prices * weights).sum(axis=1)
        result = -qr.statistics.financial_ratios.sharpe_ratio(portfolio, **kwargs)

        if isinstance(result, np.array):
            result = result[0]

        return result
