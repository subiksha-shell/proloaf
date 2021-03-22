# Copyright 2021 The ProLoaF Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ==============================================================================

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

class loss:
    def __init__(self, func, num_params):
        self._func = func
        self.num_params = num_params

    def __call__(self, *args,**kwargs):
        return self._func(*args,**kwargs)

def nll_gauss(target, predictions:list, total = True):
    assert len(predictions) == 2
    expected_value = predictions[0]
    log_variance = predictions[1]
    """
    Calculates gaussian neg likelihood score

    Parameters
    ----------
    target          : torch.Tensor
                    true values of the target variable
    expected_value  : torch.Tensor
                    predicted expected values of the target variable
    variance        : torch.Tensor
                    predicted variance values of the target variable
    log_variance    : approx. equal to log(2*pi*sigma^2)
    total           : bool
                    if true returns 1) otherwise returns 2)

    Returns
    -------
    1)float    : total gaussian negative log likelihood loss (lower the better)
    2)1d-array :  gaussian negative log likelihood loss along the horizon (lower the better)
                --- it is expected to increase as we move along the horizon

    """
    # y, y_pred, var_pred must have the same shape
    assert target.shape == expected_value.shape # target.shape = torch.Size([batchsize, horizon, # of target variables]) e.g.[64,40,1]
    assert target.shape == log_variance.shape

    squared_errors = (target - expected_value) ** 2
    if total:
        return torch.mean(squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance)
    else:
        return torch.mean(squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance, dim=0)

def pinball_loss(target, predictions:list, quantiles:list, total = True):
    #assert (len(predictions) == (len(quantiles) + 1))
    #quantiles = options

    """
        Calculates pinball loss or quantile loss against the specified quantiles

        Parameters
        ----------
        quantiles    : float list
                        quantiles that we are estimating for
        target       : torch.Tensor
                       true values of the target variable
        predictions  : list(torch.Tensor)
                       predicted expected values of the target variable

        Returns
        -------
        1)float    : total quantile loss (lower the better)


        """
    if not total:
        raise NotImplementedError("Pinball_loss does not support loss over the horizon")
    loss = 0.0

    for i, quantile in enumerate(quantiles):
        assert 0 < quantile
        assert quantile < 1
        assert target.shape == predictions[i].shape
        errors = (target - predictions[i])
        loss += (torch.mean(torch.max(quantile * errors, (quantile - 1) * errors)))

    return loss

def quantile_score(target, predictions:list, quantiles:list, total = True):
    #the quantile score builds upon the pinball loss,
    # we use the MSE to adjust the mean. one could also use 0.5 as third quantile,
    # but further code adjustments would be necessary then
    loss1 = pinball_loss(target, predictions, quantiles, total)
    #loss2 = pinball_loss(target, [predictions[2]], [0.5], total)
    loss2 = rmse(target, [predictions[len(quantiles)]])

    return (loss1+loss2)

def crps_gaussian(target, predictions:list, total = True):
    assert len(predictions) == 2
    mu = predictions[0]
    log_variance = predictions[1]

    """
    Computes normalized CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)

    Code Source: https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    target : scalar or torch.Tensor
        The observation or set of observations.
    mu : scalar or torch.Tensor
        The mean of the forecast normal distribution
    log_variance : scalar or torch.Tensor
        The standard deviation of the forecast distribution

    Returns
    -------
    crps : scalar or torch.Tensor or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules. (lower the better )

    """
    if not total:
        raise NotImplementedError("crps_gaussian does not support loss over the horizon")
    sig = torch.exp(log_variance * 0.5)
    norm_dist = torch.distributions.normal.Normal(0, 1)
    # standadized x
    sx = (target - mu) / sig
    pdf = torch.exp(norm_dist.log_prob(sx))
    cdf = norm_dist.cdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    return torch.mean(crps)

def residuals(target, predictions:list, total = True):
    """
        Calculates mean squared error

        Parameters
        ----------
        target          : torch.Tensor
                          true values of the target variable
        predictions     : torch.Tensor
                          predicted expected values of the target variable

        Returns
        -------
        1)float    : total mse (lower the better)
        2)1d-array :  mse loss along the horizon (lower the better)
                    --- it is expected to increase as we move along the horizon

        """
    if predictions[0].shape != target.shape:
        raise ValueError('dimensions of predictions and targets need to be compatible')

    error = target - predictions[0]
    if total:
        return torch.mean(error)
    else:
        return torch.mean(error, dim=0)

def mse(target, predictions:list, total = True):
    """
        Calculates mean squared error

        Parameters
        ----------
        target          : torch.Tensor
                          true values of the target variable
        predictions     : torch.Tensor
                          predicted expected values of the target variable

        Returns
        -------
        1)float    : total mse (lower the better)
        2)1d-array :  mse loss along the horizon (lower the better)
                    --- it is expected to increase as we move along the horizon

        """
    if predictions[0].shape != target.shape:
        raise ValueError('dimensions of predictions and targets need to be compatible')

    squared_errors = (target - predictions[0]) ** 2
    if total:
        return torch.mean(squared_errors)
    else:
        return torch.mean(squared_errors, dim=0)

def rmse(target, predictions:list, total = True):
    """
        Calculates root mean squared error

        Parameters
        ----------
        target : torch.Tensor
            true values of the target variable
        predictions : torch.Tensor
            predicted expected values of the target variable

        Returns
        -------
        1)float    : total rmse (lower the better)
        2)1d-array : rmse loss along the horizon (lower the better)
                    --- it is expected to increase as we move along the horizon

        """
    if predictions[0].shape != target.shape:
        raise ValueError('dimensions of predictions and targets need to be compatible')

    squared_errors = (target - predictions[0]) ** 2
    if total:
        return torch.mean(squared_errors).sqrt()
    else:
        return torch.mean(squared_errors, dim=0).sqrt()

def mape(target, predictions:list, total = True):
    """
        Calculates root mean absolute error

        Parameters
        ----------
        targets          : torch.Tensor
                           true values of the target variable
        predictions     : torch.Tensor
                          predicted expected values of the target variable

        Returns
        -------
        float    : mean absolute percentage error in % (lower the better)


    """
    
    if not total:
        raise NotImplementedError("MAPE does not support loss over the horizon")

    return torch.mean(torch.abs((target - predictions[0]) / target)) * 100

def mase(target, predictions:list, freq=1, total = True, insample_target=None):

    """
        Calculates Mean Absolute scaled error(https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
        For more clarity, please refer to the following paper
        https://www.nuffield.ox.ac.uk/economics/Papers/2019/2019W01_M4_forecasts.pdf


        Parameters
        ----------
        target   : torch.Tensor
                     series input data for predicting a forecast
        y_test     : torch.Tensor
                     true values of the target variable
        y_hat_test : torch.Tensor
                     predicted values of the target variable
        freq        : int scalar
                     frequency of season type considered, default: 1-step

        Returns
        -------
        1)float    : total mase (lower the better)


    """
    
    if not total:
        raise NotImplementedError("mase does not support loss over the horizon")
    
    y_hat_test = predictions[0]
    if insample_target==None: y_hat_naive = torch.roll(target,freq,0)# shift all values by frequency, so that at time t,
    # y_hat_naive returns the value of insample [t-freq], as the first values are 0-freq = negative,
    # all values at the beginning are filled with values of the end of the tensor. so to not falsify the evaluation,
    # exclude all terms before freq
    else: y_hat_naive = insample_target
    
    masep = torch.mean(torch.abs(target[freq:] - y_hat_naive[freq:]))
    # denominator is the mean absolute error of the "seasonal naive forecast method"
    return torch.mean(torch.abs(target[freq:] - y_hat_test[freq:])) / masep

def sharpness(target, predictions:list, total = True):
    """
        Calculates mean size of the intervals called as sharpness (lower the better)

        Parameters
        ----------
        y_pred_upper     : torch.Tensor
                           predicted upper limit of the target variable
        y_pred_lower     : torch.Tensor
                        predicted lower limit of the target variable

        Returns
        -------
        1)float    : total sharpness (lower the better)
        2)1d-array : sharpness along the horizon (lower the better)
                        --- it is expected to increase as we move along the horizon

        """
    assert len(predictions) == 2
    y_pred_upper = predictions[0]
    y_pred_lower = predictions[1]
    if total:
        return torch.mean(y_pred_upper - y_pred_lower)
    else:
        return torch.mean(y_pred_upper - y_pred_lower, dim=0)


def picp(target, predictions:list, total = True):
    """
    Calculates PICP : prediction interval coverage probability
    or simply the % of true values in the predicted intervals

    Parameters
    ----------
    targets   : torch.Tensor
                       true values of the target variable
    y_pred_upper     : torch.Tensor
                       predicted upper limit of the target variable
    y_pred_lower     : torch.Tensor
                       predicted lower limit of the target variable

    Returns
    -------
    1)float    : PICP in % (higher the better, for significance level alpha = 0.05, PICP should >= 95%)
    2)1d-array : PICP along the horizon (higher the better)
                  --- it is expected to decrease as we move along the horizon

    """

    # coverage_horizon = torch.zeros(targets.shape[1], device= targets.device,requires_grad=True)
    # for i in range(targets.shape[1]):
    #     # for each step in forecast horizon, calcualte the % of true values in the predicted interval
    #     coverage_horizon[i] = (torch.sum((targets[:, i] > y_pred_lower[:, i]) & 
    #                             (targets[:, i] <= y_pred_upper[:, i])) / targets.shape[0]) * 100
    assert len(predictions) == 2
    #torch.set_printoptions(precision=5)
    y_pred_upper = predictions[0]
    y_pred_lower = predictions[1]
    coverage_horizon = 100. * (torch.sum((target > y_pred_lower) &
                                         (target <= y_pred_upper), dim=0)) / target.shape[0]

    coverage_total = torch.sum(coverage_horizon) / target.shape[1]
    if total:
        return coverage_total
    else:    
        return coverage_horizon

def picp_loss(target, predictions, total = True):
    return 1-picp(target, predictions, total)

def mis(target, predictions:list, alpha=0.05, total = True):
    """
       Calculates MIS : mean interval score without scaling by seasonal difference
       -- It combines both the sharpness and PICP metrics into a scalar value
       For more,please refer to https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf

       Parameters
       ----------
       target           : torch.Tensor
                          true values of the target variable
       y_pred_upper     : torch.Tensor
                          predicted upper limit of the target variable
       y_pred_lower     : torch.Tensor
                          predicted lower limit of the target variable
       alpha            : float
                          significance level for the prediction interval

       Returns
       -------
       1)float    : MIS  (lower the better)
       2)1d-array : MIS along the horizon (lower the better)
                     --- it is expected to increase as we move along the horizon

       """

    assert len(predictions) == 2
    y_pred_upper = predictions[0]
    y_pred_lower = predictions[1]

    mis_horizon = torch.zeros(target.shape[1])

    for i in range(target.shape[1]):
        # calculate penalty for large prediction interval
        large_PI_penalty = torch.sum(y_pred_upper[:, i] - y_pred_lower[:, i])

        # calculate under estimation penalty
        diff_lower = y_pred_lower[:, i] - target[:, i]
        under_est_penalty = (2 / alpha) * torch.sum(diff_lower[diff_lower>0])
                            
        # calcualte over estimation penalty
        diff_upper = target[:, i] - y_pred_upper[:, i]
        over_est_penalty = (2 / alpha) * torch.sum(diff_upper[diff_upper>0])
                            
        # combine all the penalties
        mis_horizon[i] = (large_PI_penalty + under_est_penalty + over_est_penalty) / target.shape[0]

    mis_total = torch.sum(mis_horizon) / target.shape[1]

    if total:
        return mis_total
    else:
        return mis_horizon

def rae(target, predictions: list, total=True):
    """
        Calculates Relative Absolute Error
        compared to a naive forecast that only assumes that the future will produce the average of the past observations

        Parameters
        ----------
        target   : torch.Tensor
                     series input data for predicting a forecast
        y_test     : torch.Tensor
                     true values of the target variable
        y_hat_test : torch.Tensor
                     predicted values of the target variable

        Returns
        -------
        1)float    : total rae (lower the better)


    """
    y_hat_test = predictions[0]
    y_hat_naive = target
    y_hat_naive = torch.mean(target)

    if not total:
        raise NotImplementedError("rae does not support loss over the horizon")

    # denominator is the mean absolute error of the preidicity dependent "naive forecast method"
    # on the test set -->outsample
    return torch.mean(torch.abs(target - y_hat_test)) / torch.mean(torch.abs(target - y_hat_naive))


def nmae(target, predictions: list, total=True):
    """
        Calculates normalized absolute error
        nMAE is different from MAPE in that the average of mean error is normalized over the average of all the actual values

        Parameters
        ----------
        target   : torch.Tensor
                     series input data for predicting a forecast
        y_test     : torch.Tensor
                     true values of the target variable
        y_hat_test : torch.Tensor
                     predicted values of the target variable

        Returns
        -------
        1)float    : total nmae (lower the better)


    """

    if not total:
        raise NotImplementedError("nmae does not support loss over the horizon")

    y_hat_test = predictions[0]

    return torch.sum(torch.abs(target - y_hat_test)) / torch.sum(torch.abs(target))

def results_table(models, mse, rmse, mase, rae, mae, sharpness, coverage, mis, quantile_score=0, save_to_disc=False):
    data = {
        'MSE': mse,
        'RMSE': rmse,
        'MASE':mase,
        'RAE':rae,
        'nMAE':mae,
        'Mean sharpness': sharpness,
        'Mean PICP': coverage,
        'Mean IS': mis,
        'Quantile Score': quantile_score}

    results_df = pd.DataFrame(data, index=[models])
    if(save_to_disc):
        results_df.to_csv(save_to_disc+models+'.csv', sep=';', index=True)

    return results_df

def evaluate_hours(target, pred, y_pred_upper, y_pred_lower, hour, OUTPATH, limit, actual_hours=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(target, '.-k', label="Truth")  # true values
    ax.plot(pred, 'b', label='Predicted')
    # insert actual time
    if(actual_hours.dt.hour.any()):
        ax.set_title(actual_hours.iloc[0].strftime("%a, %Y-%m-%d"), fontsize=20)
    else:
        ax.set_title('Forecast along horizon', fontsize=22)
    ax.fill_between(np.arange(pred.shape[0]), pred.squeeze(), y_pred_upper.squeeze(), alpha=0.1, color='g')
    ax.fill_between(np.arange(pred.shape[0]), y_pred_lower.squeeze(), pred.squeeze(), alpha=0.1, color='g')

    ax.set_xlabel("Hour", fontsize=18)
    ax.set_ylabel("Scaled Residual Load (-1,1)", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(b=True, linestyle='-')
    if(limit):
        plt.axhline(linewidth=2, color='r', y=limit)
    ax.grid()
    positions = range(0, len(pred), 2)
    labels = actual_hours.dt.hour.to_numpy()
    new_labels = labels[positions]
    plt.xticks(positions, new_labels)
    ax.set_xlabel("Hour of Day", fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(OUTPATH + 'eval_hour{}'.format(hour))
    plt.close(fig)

def plot_metrics(rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon, OUTPATH, title):
    with plt.style.context('seaborn'):
        fig = plt.figure(figsize=(16, 12))
        st = fig.suptitle(title, fontsize=25)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        ax_rmse = plt.subplot(2, 2, 1)
        ax_sharpness = plt.subplot(2, 2, 3)
        ax_PICP = plt.subplot(2, 2, 2)
        ax_MSIS = plt.subplot(2, 2, 4)

        ax_rmse.plot(rmse_horizon, label='rmse')
        ax_rmse.set_title('RMSE along horizon', fontsize=22)
        ax_rmse.set_xlabel("Hour", fontsize=18)
        ax_rmse.set_ylabel("RMSE", fontsize=20)
        ax_rmse.legend(fontsize=20)
        ax_rmse.grid(b=True, linestyle='-')

        ax_sharpness.plot(sharpness_horizon, label='sharpness')
        ax_sharpness.set_title('sharpness along horizon', fontsize=22)
        ax_sharpness.set_xlabel("Hour", fontsize=18)
        ax_sharpness.set_ylabel("sharpness", fontsize=20)
        ax_sharpness.legend(fontsize=20)
        ax_sharpness.grid(b=True, linestyle='-')

        ax_PICP.plot(coverage_horizon, label='coverage')
        ax_PICP.set_title('coverage along horizon', fontsize=22)
        ax_PICP.set_xlabel("Hour", fontsize=18)
        ax_PICP.set_ylabel("coverage in %", fontsize=20)
        ax_PICP.legend(fontsize=20)
        ax_PICP.grid(b=True, linestyle='-')

        ax_MSIS.plot(mis_horizon, label='MIS')
        ax_MSIS.set_title('Mean Interval score', fontsize=22)
        ax_MSIS.set_xlabel("Hour", fontsize=18)
        ax_MSIS.set_ylabel("MIS", fontsize=20)
        ax_MSIS.legend(fontsize=20)
        ax_MSIS.grid(b=True, linestyle='-')

        st.set_y(1.08)
        fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.savefig(OUTPATH + title+'_metrics_plot')
        # plt.show()