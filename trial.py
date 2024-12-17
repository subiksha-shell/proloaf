import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.utils import validation
from IPython.display import display
import ipywidgets as widgets

from functools import partial

MAIN_PATH = os.path.abspath(os.path.join(__file__, '../src'))
sys.path.append(MAIN_PATH)

from proloaf.event_logging import create_event_logger
logger = create_event_logger('evaluate')    ##########

from proloaf import metrics
import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl

from proloaf.cli import parse_with_loss

import subprocess

#subprocess.run('python src/preprocess.py -s opsd')  ##########

#subprocess.run('python src/train.py -s opsd --smoothed_quantiles 0.025 0.5 0.975')  ##########

config_path = 'opsd'
PAR = ch.read_config(config_path)
torch.manual_seed(1)
model_name = PAR["model_name"]
data_path = PAR["data_path"]
INFILE = data_path  # input factsheet
INMODEL = os.path.join("", PAR["output_path"], model_name)
OUTDIR = os.path.join("", PAR["evaluation_path"])
DEVICE = "cpu"
print('hi')

target_id = PAR["target_id"]
SPLIT_RATIO = PAR["validation_split"] # Data that has not been used *at all* during training, also not for testing
HISTORY_HORIZON = PAR["history_horizon"]
FORECAST_HORIZON = PAR["forecast_horizon"]
feature_groups = PAR["feature_groups"]

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# Read load data
df = pd.read_csv(INFILE, sep=";", index_col=0, parse_dates=True)

#Originally we have trained the model to predict 24h ahead.
adj_forecast_horizon = widgets.IntSlider(value=PAR["forecast_horizon"])
display(adj_forecast_horizon)

FORECAST_HORIZON = adj_forecast_horizon.value

with torch.no_grad():
    net = mh.ModelHandler.load_model(f"{INMODEL}.pkl", locate=DEVICE)
    
    # Here we split the data still in pandas dataframe format into two parts. 
    # The split ratio is here really to seperate the data that was not seen during training, also not for validation purposes.
    train_df, test_df = dh.split(df, [SPLIT_RATIO])
    ## Now we create a tensor that also still contains the original dataframe. 
    test_data = tl.TimeSeriesData(
        test_df,
        device=DEVICE,
        ## Before the tensor object is created, all listed preparation steps are applied to the pandas dataframe
        ## Depending on the underlying data, some steps can be skipped, by simply removing the item from the list
        preparation_steps=[
                partial(dh.set_to_hours, freq="1H"),# Here you can specify a datetime frequency. Default: '1H'
                partial(dh.fill_if_missing, periodicity=PAR.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                net.scalers.transform,
                dh.check_continuity,
            ],
        **PAR,
    )
    prepped_data = test_data.make_data_loader(batch_size=None, shuffle=False)

prepped_data.dataset.data

prepped_data.dataset.get_as_frame

prepped_data.dataset.get_as_frame("2019-01-01 00:00:00")

print("Encoder Features:", prepped_data.dataset.encoder_features, "\n", )

print("Decoder Features:", prepped_data.dataset.decoder_features,"\n")

print("Targets:", prepped_data.dataset.target_id, "\n")

forecast_sample=prepped_data.dataset.data.index.get_loc("2019-01-01 00:00:00")
inputs_enc, inputs_dec, targets = prepped_data.dataset[forecast_sample]

print("Historic encoder inputs with shape ([n_timesteps, n_features]):",inputs_enc.shape, "\nand future decoder inputs with shape ([n_timesteps, n_features]):", inputs_dec.shape)

with torch.no_grad():
    prediction = net.predict(inputs_enc.unsqueeze(0), inputs_dec.unsqueeze(0))
prediction

print("Trained", net.name, "with", net.loss_metric.id, "resulting in following outputs:", net.output_labels)

quantile_prediction = net.loss_metric.get_quantile_prediction(
    predictions=prediction, target=targets.unsqueeze(0)
)

expected_values = quantile_prediction.get_quantile(0.5)
y_pred_upper = quantile_prediction.select_upper_bound().values.squeeze(
    dim=2
)
y_pred_lower = quantile_prediction.select_lower_bound().values.squeeze(
    dim=2
)

##The lower and upper bound can now be returned seperately: 
#quantile_prediction.select_lower_bound().values
#quantile_prediction.select_upper_bound().values

#or together with the mean value:
quantile_prediction.values

import proloaf.plot as plot

# Fetch the actual time from the datetimeindex in the pandas dataframe
actual_time = pd.Series(pd.to_datetime(df.index), index=df.index, dtype=object)
actuals = actual_time[forecast_sample : forecast_sample + FORECAST_HORIZON]
plot.plot_timestep(
    targets.detach().squeeze().numpy(),
    expected_values.detach().numpy()[0],
    y_pred_upper.detach().numpy()[0],
    y_pred_lower.detach().numpy()[0],
    forecast_sample,
    OUTDIR,
    PAR["cap_limit"],
    actuals,
)

# Plot for days with possible congestion
print("One can also plot a desired limit, to visually indicate congestion in high load situations.")
plot.plot_timestep(
    targets.detach().squeeze().numpy(),
    expected_values.detach().numpy()[0],
    y_pred_upper.detach().numpy()[0],
    y_pred_lower.detach().numpy()[0],
    forecast_sample,
    OUTDIR,
    PAR["cap_limit"],
    actuals,
    draw_limit=True
)

test_metrics_total = [
            metrics.NllGauss(),
            metrics.Mse(),
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Rae(),
            metrics.Mae(),
            metrics.Mis(),
            metrics.Mase(),
            metrics.PinnballLoss(),
            metrics.Residuals(),
        ]

results_total_per_forecast = (
    mh.ModelHandler.benchmark(
        test_data,
        [net],
        test_metrics=test_metrics_total,
        avg_over="all",
    )
    .iloc[0]
    .unstack()
)
results_total_per_forecast

## Specify which metrics to look into per sample
test_metrics_sample = [metrics.Mse(), metrics.Rmse()]

## RESULTS PER SAMPLE
results_per_sample_per_forecast = mh.ModelHandler.benchmark(
    test_data,
    [net],
    test_metrics=test_metrics_sample,
    avg_over="time",
)

print("A boxplot will serve as the most practical plot to show how much the error metrics deviate\nfrom one forecast situation to another. We refer to each forecast situation as one sample.")
# BOXPLOTS
with torch.no_grad():
    plot.plot_boxplot(
        metrics_per_sample=results_per_sample_per_forecast[net.name],
        sample_frequency=24, ## choose here the sample-frequencyy. It is set to 24 steps, here 24 hours = daily
        save_to=OUTDIR,
    )
print("Remember that the number of test samples is", len(test_data), ".")
print("The period ranges from", test_data.data.index[0].strftime("%a, %Y-%m-%d"), "to", test_data.data.index[-1].strftime("%a, %Y-%m-%d"), ".")
print("For sake of clarity we have plotted the samples in 24h intervals resulting in:", int(len(test_data)/24), "days in the test set.")

#first fetch the metrics per time step on the forecast horizon (=24 hours in our case)
test_metrics_timesteps = [
    metrics.NllGauss(),
    metrics.Rmse(),
    metrics.Sharpness(),
    metrics.Picp(),
    metrics.Mis(),
]

results_per_timestep_per_forecast = mh.ModelHandler.benchmark(
    test_data,
    [net],
    test_metrics=test_metrics_timesteps,
    avg_over="sample",
)

rmse_values = pd.DataFrame(
    data=results_per_timestep_per_forecast.xs(
        "Rmse", axis=1, level=1, drop_level=True
    ),
    columns=[net.name],
)
sharpness_values = pd.DataFrame(
    data=results_per_timestep_per_forecast.xs(
        "Sharpness", axis=1, level=1, drop_level=True
    ),
    columns=[net.name],
)
picp_values = pd.DataFrame(
    data=results_per_timestep_per_forecast.xs(
        "Picp", axis=1, level=1, drop_level=True
    ),
    columns=[net.name],
)
mis_values = pd.DataFrame(
    data=results_per_timestep_per_forecast.xs(
        "Mis", axis=1, level=1, drop_level=True
    ),
    columns=[net.name],
)

plot.plot_metrics(
    rmse_values,
    sharpness_values,
    picp_values,
    mis_values,
    OUTDIR,
    title="How does the forecast perform on average over the forecast period (=24 hours)?"
)