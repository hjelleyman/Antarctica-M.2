"""
Functions from week 9
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats, linalg
import statsmodels.api as sm

# First we are going to rewrite the regression and trend functions to calculate linear regressions with p-values.


def regression(dependant, independant):
    independant = independant.reshape(-1, 1)
    if all(independant == 0):
        slope = np.array([np.nan])
        intercept = np.array([np.nan])
        prediction = np.array([np.nan]*18)
        p_value = np.nan
    else:
        slope, intercept = linalg.lstsq(independant, dependant)[:2]
        prediction = independant[0] * slope + intercept
        newX = pd.DataFrame({"Constant": np.ones(len(independant))}).join(
            pd.DataFrame(independant))
        MSE = (sum((dependant-prediction)**2))/(len(newX)-len(newX.columns))
        var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = slope / sd_b
        p_values = [2*(1-stats.t.cdf(np.abs(i), (len(newX)-len(newX[0]))))
                    for i in ts_b]
        p_value = p_values[1]

    return slope, prediction, p_value


def fast_single_regression(dependant, independant):

    print(f'Running regression for {dependant.name} from {independant.name}')

    dependant = dependant.sortby('time')
    independant = independant.sortby('time')
    slope, intercept, r_value, p_value, std_err = xr.apply_ufunc(stats.linregress, independant, dependant,
                                                                 input_core_dims=[
                                                                     ['time'], ['time']],
                                                                 output_core_dims=[
                                                                     [], [], [], [], []],
                                                                 vectorize=True)
    print('Regression finished')

    out = xr.Dataset()
    out['dependant'] = dependant
    out['indepenant'] = independant
    out['regression'] = slope
    out['pvalues'] = p_value
    out['correlation'] = r_value
    out['intercept'] = intercept
    out['prediction'] = intercept + slope * independant

    return out


def calculate_trend(da):
    independant = da.time.astype(float).values * 1e9*60*60*24*365

    def trend(dependant): return stats.linregress(dependant, independant)

    slope, intercept, r_value, p_value, std_err = xr.apply_ufunc(trend, da,
                                                                 input_core_dims=[
                                                                     ['time'], ['time']],
                                                                 output_core_dims=[
                                                                     [], [], [], [], []],
                                                                 vectorize=True)
    print('Regression finished')
    print(regression_result)

# Correlations


def fast_single_correlation(dependant, independant):

    print(f'Running correlation for {dependant.name} and {independant.name}')

    correlation, pvalues = xr.apply_ufunc(stats.pearsonr, dependant, independant,
                                          input_core_dims=[
                                              ['time'], ['time']],
                                          output_core_dims=[
                                              [], []],
                                          vectorize=True)

    print('correlation finished')

    out = xr.Dataset()
    out['correlation'] = correlation
    out['pvalues'] = pvalues

    return out
