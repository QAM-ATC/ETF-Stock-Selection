
import numpy as np
import scipy.optimize as opt
import quant_risk as qr
import pandas as pd

class ObjectiveFunction:

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

    def minimum_drawdown(weights: np.array, prices: pd.DataFrame):
        """Objective function that returns the maximum drawdown, given a price dataframe along with weights.

        Parameters
        ----------
        weights : np.array
            numpy array of our weights that will be optimised
        prices : pd.DataFrame
            Dataframe of prices of our tickers with dates as the index

        Returns
        -------
        float
            Returns the maximum drawdown of the portfolio
        """
        portfolio = (prices * weights).sum(axis=1)
        result = qr.statistics.stats.maximum_drawdown(portfolio)

        return result

    def sortino(weights: np.array, prices: pd.DataFrame, **kwargs):
        """Objective function that returns the negative sortino ratio (since it is a minimisation problem), given a price dataframe.
        Additional arguements can be provided such as riskFreeRate, periodsPerYear, reqReturn to account for annualisation etc.

        Parameters
        ----------
        weights : np.array
            numpy array of our weights that will be optimised
        prices : pd.DataFrame
            Dataframe of prices of our tickers with dates as the index

        Returns
        -------
        float
            Returns the negative of the sortino ratio
        """

        portfolio = (weights * prices).sum(axis=1)
        result = -qr.statistics.financial_ratios.sortino_ratio(portfolio, **kwargs)

        return result

    def minimum_volatility(weights: np.array, prices: pd.DataFrame, **kwargs):
        """Objective function that minimises the annualised volatility of the portfolio. Can take additional arguements such as periodsPerYear for annualisation.

        Parameters
        ----------
        weights : np.array
            numpy array of our weights that will be optimised
        prices : pd.DataFrame
            Dataframe of prices of our tickers with dates as the index

        Returns
        -------
        float
            Returns the annualised volatility of the portfolio
        """

        portfolio = (weights * prices).sum(axis=1)
        portfolio_returns = portfolio.pct_change().dropna()
        result = qr.statistics.annualize.annualised_volatility(portfolio_returns, **kwargs)

        return result
