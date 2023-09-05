import numpy as np
import pandas as pd
from root import ROOT_DIR, Path
import plotly.graph_objects as go


PLOT_DIR = ROOT_DIR / 'reports/figures'

def plot_skill(df):

    df['sort'] = df['type'] + ' ' + df['target'] + ' ' + df['horizon']

    fig = go.Figure()

    metrics = ['MAE', 'MBE', 'RMSE']
    marker_styles = ['square', 'circle', 'triangle-up']


    for i, metric in enumerate(metrics):
        for model in df['model'].unique():
            subset_df = df[(df['model'] == model)]
            scatter = go.Scatter(x=subset_df['sort'], y=subset_df[metric], mode='markers',
                                name=f'{model} - {metric}', marker_symbol=marker_styles[i], marker_size=10)
            fig.add_trace(scatter)


    # Update the layout
    fig.update_layout(
        title='Metrics vs. Type and Target',
        xaxis_title='Type and Target',
        yaxis_title='Metrics',
        legend_title='Metric'
    )
    fig.write_html(PLOT_DIR / f"results.html")
    fig.show()


def plot_forecast(inputs, predictions, name):
    inputs['predictions'] = predictions
    fig = inputs.plot(backend='plotly')
    fig.write_html(PLOT_DIR / f"{name}.html")
    # fig.show()
