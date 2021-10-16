
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

class Optimisation(ObjectiveFunction, Constraint):

    def __init__(self, prices: pd.DataFrame, objective_function = None, constraint: list = []):

        self.objective_function = objective_function
        self.constraint = constraint
        self.prices = prices
        self.bounds = tuple((0.0, 1.0) for _ in range(len(self.prices.columns)))

    def optimise(self, **kwargs):

        if not self.objective_function:

            obj_funcs = [method for method in dir(ObjectiveFunction) if method.startswith('__') is False]

            print("Which of the following default objective functions would you like to use: ")

            for idx, func in enumerate(obj_funcs):
                print(f" {idx}. {func}")

            input_func = int(input("Please enter the function number you would like to choose: "))

            self.objective_function = eval(f"ObjectiveFunction.{obj_funcs[input_func]}")

        weights_constraint = Constraint.weights_constraint()

        self.initial_guess = np.random.random(size=len(self.prices.columns))
        self.initial_guess /= sum(self.initial_guess)

        industry_constraint = Constraint.industry_constraints(**kwargs)

        self.constraint.extend(weights_constraint)
        self.constraint.extend(industry_constraint)

        args = (self.prices)

        result = opt.minimize(self.objective_function, self.initial_guess, args = args, method = 'SLSQP',
                    bounds=self.bounds, options={'disp': True})

        return result
