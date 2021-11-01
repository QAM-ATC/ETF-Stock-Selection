
import numpy as np
import scipy.optimize as opt
import quant_risk as qr
import pandas as pd
from pandas.tseries.offsets import QuarterEnd, BusinessDay
from tqdm import tqdm
import datetime as dt

global count
count = 0

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
        result = -qr.statistics.stats.maximum_drawdown(portfolio)

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

class Constraints:
    def industry_constraint(weights: np.array, weightsInIndustry: list, maxWeight: float):
            """Constraint to enforce the industry constraint for each industry

            Parameters
            ----------
            weights : np.array
                weights of our portfolio
            weightsInIndustry : list
                weights of tickers present in industry
            maxWeight : float
                maximum possible weight to allocate to the current industry

            Returns
            -------
            float
                Returns the constraint value which must be non-negative
            """

            return (maxWeight - np.sum([weights[i] for i in weightsInIndustry]))

    def weights_constraint(weights: np.array):
        """Constraint to check weights add up to 1

        Parameters
        ----------
        weights : np.array
            Weights of the tickers

        Returns
        -------
        float
            Constraint value
        """
        return np.sum(weights) - 1

    def turnover_constraint(weights: np.array, pastWeights: np.array, *args, **kwargs):
        """Constraint to check that turnover is a maximum of 50% for each ticker

        Parameters
        ----------
        weights : np.array
            Weights of the tickers
        pastWeights : np.array
            Previous weights of the tickers

        Returns
        -------
        float
            Constraint value
        """

        MAX_TURNOVER = 0.5
        change_in_portfolio = MAX_TURNOVER - np.sum([abs(weights[i] - pastWeights[i]) for i in range(len(weights))])


        return change_in_portfolio

class ConstraintWrappers:

    def industry_constraints(industries: dict, tickers: list, industryWeights: dict):
        """This constraint function allocates weights to specific sectors by providing a maxWeight for each sector.
        It returns a list of constraints for each sector

        Parameters
        ----------
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

        result = []

        for industry, firms in industries.items():

            weightsInIndustry = []
            for idx, firm in enumerate(tickers):
                if firm in firms:
                    weightsInIndustry.append(idx)

            args = (weightsInIndustry, industryWeights[industry])

            constraint = {
                        'type': 'ineq',
                        'fun': Constraints.industry_constraint,
                        'args': args
                        }

            result.append(constraint)

        return result

    def weights_constraint():
        """Constraint to ensure that the weights in our portfolio always sum to one.

        Returns
        -------
        list
            Returns a list containing a dictionary of our constraint
        """

        constraint = [{
                    'type': 'eq',
                    'fun': Constraints.weights_constraint
                        }]

        return constraint

    def turnover_constraint(past_weights, **kwargs):

        constraint = [{
            'type': 'ineq',
            'fun': Constraints.turnover_constraint,
            'args': ([past_weights])
                }]

        return constraint


class Optimisation(ObjectiveFunction, ConstraintWrappers):

    def __init__(self, prices: pd.DataFrame, objective_function = None, constraint: list = []):

        if objective_function is not None:
            self.objective_function = objective_function
        if constraint:
            self.constraint = constraint
        self.prices = prices
        self.bounds = tuple((0.0, 1.0) for _ in range(len(self.prices.columns)))

    def optimise(self, **kwargs):

        if not self.objective_function:

            obj_funcs = [method for method in dir(ObjectiveFunction) if method.startswith('__') is False]
            self.objective_function = ObjectiveFunction.minimum_drawdown

        self.initial_guess = np.random.random(size=len(self.prices.columns))
        self.initial_guess /= sum(self.initial_guess)

        args = (self.prices)

        result = opt.minimize(self.objective_function, self.initial_guess, args = args, method = 'SLSQP',
                    bounds=self.bounds, options={'disp': False}, constraints=self.constraint)

        return result

class RollingOptimisation(Optimisation):

    def __init__(self, prices: pd.DataFrame, objective_function = None, constraint: list = [], rollback: int = 63):

        if objective_function is not None:
            self.objective_function = eval("ObjectiveFunction."+objective_function)
        if constraint:
            self.constraint = constraint
        self.prices = prices
        self.rollback = rollback

    def backtest(self, **kwargs):

        first = True
        dates = pd.date_range(start=self.prices.index[0], end=dt.date.today(), freq='B').tolist()
        empty_dataframe = pd.DataFrame(index=dates)
        self.prices = pd.concat([self.prices, empty_dataframe], axis=1, join='outer').ffill()
        pastweights = []

        tickers = self.prices.columns.tolist()

        for date in tqdm(dates[self.rollback:]):

            train = self.prices.loc[date - BusinessDay(self.rollback): date - BusinessDay(1), :]
            test = pd.DataFrame(self.prices.loc[date, :]).T

            constraints = []
            for cons in self.constraint:
                if cons=="industry_constraints":
                    items=ConstraintWrappers.industry_constraints(kwargs['industries'],tickers,kwargs['industryWeights'])
                elif cons=="weights_constraint":
                    items=ConstraintWrappers.weights_constraint()
                elif cons=="turnover_constraint" and first==False:
                    items=ConstraintWrappers.turnover_constraint(pastweights[-1])
                for item in items:
                        constraints.append(item)

            if first:
                weights = Optimisation(train, self.objective_function,constraints).optimise()['x']
                first = False

            else:
                weights = Optimisation(train, self.objective_function,constraints).optimise()['x']
                
            pastweights.append(weights)

            # if first:

            #     portfolio = pd.DataFrame((test*weights).sum(axis=1) / (test*weights).sum(axis=1).iloc[0])
            #     portfolio.columns = ['Portfolio']
            #     first = False

        pastweights = pd.DataFrame(pastweights, index=dates[self.rollback:])
        pastweights.columns = self.prices.columns

        return pastweights
