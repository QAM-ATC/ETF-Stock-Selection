
import numpy as np
import scipy.optimize as opt
import quant_risk as qr
import pandas as pd
from pandas.tseries.offsets import QuarterEnd, BusinessDay
from tqdm import tqdm
import datetime as dt

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

class Constraint:

    @staticmethod
    def industry_constraints(industries: dict, tickers: list, industryWeights: dict):
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
        def _industry_constraints_(weights: np.array, weightsInIndustry: list, maxWeight: float):
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

            return (maxWeight - np.sum([weights[i] for i in weightsInIndustry]))

        result = []

        for industry, firms in industries.items():

            weightsInIndustry = []
            for idx, firm in enumerate(tickers):
                if firm in firms:
                    weightsInIndustry.append(idx)

            args = (weightsInIndustry, industryWeights[industry])

            constraint = {
                        'type': 'ineq',
                        'fun': _industry_constraints_,
                        'args': args
                        }

            result.append(constraint)

        return result

    @staticmethod
    def weights_constraint():
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

    @staticmethod
    def turnover_constraint(past_weights, **kwargs):

        constraint = [{
            'type': 'ineq',
            'fun': lambda weights, past_weights: 0.3 - np.sum([(abs(weight[i] - past_weights[i])
                                             for i in range(len(weights)))
                                             ]),
            'args': (past_weights)
                }]

        return constraint


class Optimisation(ObjectiveFunction, Constraint):

    def __init__(self, prices: pd.DataFrame, objective_function = None, constraint: list = []):

        self.objective_function = objective_function
        self.constraint = constraint
        self.prices = prices
        self.bounds = tuple((0.0, 1.0) for _ in range(len(self.prices.columns)))

    def optimise(self, **kwargs):

        if not self.objective_function:

            obj_funcs = [method for method in dir(ObjectiveFunction) if method.startswith('__') is False]

            # print("Which of the following default objective functions would you like to use: ")

            # for idx, func in enumerate(obj_funcs):
            #     print(f" {idx}. {func}")

            # input_func = int(input("Please enter the function number you would like to choose: "))

            # self.objective_function = eval(f"ObjectiveFunction.{obj_funcs[input_func]}")
            self.objective_function = ObjectiveFunction.minimum_drawdown

        weights_constraint = Constraint.weights_constraint()
        turnover_constraint = Constraint.turnover_constraint(**kwargs)

        self.initial_guess = np.random.random(size=len(self.prices.columns))
        self.initial_guess /= sum(self.initial_guess)

        # industry_constraint = Constraint.industry_constraints(**kwargs)

        self.constraint.extend(weights_constraint)
        # self.constraint.extend(industry_constraint)
        self.constraint.extend(turnover_constraint)
        args = (self.prices)

        result = opt.minimize(self.objective_function, self.initial_guess, args = args, method = 'SLSQP',
                    bounds=self.bounds, options={'disp': False})

        return result

class RollingOptimisation(Optimisation):

    def __init__(self, prices: pd.DataFrame, objective_function = None, constraint: list = [], rollback: int = 63):

        self.objective_function = objective_function
        self.constraint = constraint
        self.prices = prices
        self.rollback = rollback

    def backtest(self, **kwargs):

        first = True
        pastweights = []
        dates = pd.date_range(start=self.prices.index[0], end=dt.date.today(), freq='B').tolist()
        empty_dataframe = pd.DataFrame(index=dates)
        self.prices = pd.concat([self.prices, empty_dataframe], axis=1, join='outer').ffill()

        for date in tqdm(dates[self.rollback:]):

            train = self.prices.loc[date - BusinessDay(self.rollback): date - BusinessDay(1), :]
            test = pd.DataFrame(self.prices.loc[date, :]).T

            if first:
                weights = Optimisation(train, self.objective_function, self.constraint).optimise(
                                                past_weights=[1/len(self.prices.columns)]*len(self.prices.columns))['x']
            else:
                weights = Optimisation(train, self.objective_function, self.constraint).optimise(
                                                                past_weights=pastweights[-1])['x']
            pastweights.append(weights)

            if first:

                portfolio = pd.DataFrame((test*weights).sum(axis=1) / (test*weights).sum(axis=1).iloc[0])
                portfolio.columns = ['Portfolio']
                first = False
        pastweights = pd.DataFrame(pastweights, index=dates[self.rollback:])
        pastweights.columns = self.prices.columns

        return portfolio, pastweights
        # Update daily and take previous 3m data
