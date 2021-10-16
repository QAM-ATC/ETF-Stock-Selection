
import numpy as np
import scipy.optimize as opt
import quant_risk as qr
import pandas as pd

class ObjectiveFunction:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

class Constraint:

    @staticmethod
    def industry_constraints(weights: np.array, industries: dict, tickers: list, industryWeights: dict):
        """This constraint function allocates weights to specific sectors by providing a maxWeight for each sector.
        It returns a list of constraints for each sector

        Parameters
        ----------
        weights : np.array
            weights of our portfolio
        industries : dict
            dictionary with list of tickers as values, the key is the industry and the values are the tickers in that industry
        tickers : list
            all possible tickers in our portfolio, used to verify weight indices
        industryWeights : dict
            dictionary with industry as key and value is the max weight to allocate to that industry. keys must be the same as 'industries' dictionary

        Returns
        -------
        list
            Returns a list of constraints for each sector
        """

        @staticmethod
        def _industry_constraints_(weights: np.array, industry: list, tickers: list, maxWeight: float):
            """Inner function to enforce the industry constraint for each industry

            Parameters
            ----------
            weights : np.array
                weights of our portfolio
            industry : list
                tickers present in a given industry in the form of a list
            tickers : list
                all possible tickers in our portfolio, used to verify weight indices
            maxWeight : float
                maximum possible weight to allocate to the current industry

            Returns
            -------
            float
                Returns the constraint value which must be non-negative
            """

            weightsInIndustry = []

            for idx, firm in enumerate(tickers):
                if firm in industry:
                    weightsInIndustry.append(idx)

            return (maxWeight - np.sum([weights[i] for i in weightsInIndustry]))

        result = []

        for industry in industries.keys():

            args = (industry, tickers, industryWeights[industry])

            constraint = {
                        'type': 'ineq',
                        'fun': _industry_constraints_,
                        'args': args
                        }

            result.append(constraint)

        return result

    @staticmethod
    def weights_constraint(weights: np.array):
        """Constraint to ensure that the weights in our portfolio always sum to one.

        Parameters
        ----------
        weights : np.array
            weights of our portfolio

        Returns
        -------
        list
            Returns a list containing a dictionary of our constraint
        """

        constraint = [{
                    'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1
                        }]

        return constraint
