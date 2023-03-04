from models import Model, Distribution, myModel, myDistribution
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_samples(dates_to_predict: List[str], samples: Dict[str, float], axis, start_index=0):
    xdates = range(start_index, start_index+len(dates_to_predict))
    axis.plot(xdates, samples.keys(), color="pink")

def plot_predictions(dates_to_predict: List[str], predictions: Dict[str, Distribution], axis, start_index=0):
    xdates = range(start_index, start_index+len(dates_to_predict))
    mean_t = [predictions[date].get_mean() for date in dates_to_predict]
    percentile_10 = [predictions[date].get_percentile(10) for date in dates_to_predict]
    percentile_25 = [predictions[date].get_percentile(25) for date in dates_to_predict]
    percentile_75 = [predictions[date].get_percentile(75) for date in dates_to_predict]
    percentile_90 = [predictions[date].get_percentile(90) for date in dates_to_predict]
    
    axis.plot(xdates, mean_t)
    axis.fill_between(x=xdates, y1=percentile_10, y2=percentile_90, alpha=0.1)
    axis.fill_between(x=xdates, y1=percentile_25, y2=percentile_75, alpha=0.4)

def plot_series(typ, actual_vol, dates_to_predict, predictions, samples, axis):
    x_actual = range(0, len(actual_vol))
    axis.plot(x_actual, actual_vol, color="k")
    axis.set_title(f"VOLUME PREDICTIONS: {typ}")
    x_test = range(len(actual_vol), len(actual_vol) + len(dates_to_predict))
    axis.plot(x_test, samples.values(), color='pink')
    plot_predictions(dates_to_predict, predictions, axis, start_index=len(actual_vol))
    
def run(data: pd.DataFrame, model: Model) -> None:

    model.train(data, start_date="20210420", end_date="20220901")
    
    types = ["TOTAL", "A", "B", "C"]
    dates_to_predict = [f"{dt:%Y%m%d}" for dt in pd.date_range(start="20220902", end="20220923")]
    predictions = {}
    samples = {}
    fig, ax = plt.subplots(4, 1, figsize=(12, 10))
    for typ, axis in zip(types, ax):
        predictions[typ] = model.predict(dates_to_predict, type_=typ)

        samples[typ] = model.sample(dates_to_predict, type_=typ)
        
        if typ == "TOTAL":
            actual_vol = data.groupby("DATE").agg({"VOLUME": "sum"}).reset_index()["VOLUME"]
        else:
            actual_vol = data[data["TYPE"] == typ].reset_index()["VOLUME"]
        
        plot_series(typ, actual_vol, dates_to_predict, predictions[typ], samples[typ], axis)
    
    fig.tight_layout()
    plt.savefig("model_output.png")
    return
    


if __name__ == "__main__":
    
    path_to_data = "data.txt"
    data = pd.read_csv(
        path_to_data, 
        delimiter="\t", 
        parse_dates=['DATE'], 
        index_col='DATE',
    ).sort_index()
    
    random_forest_params = {
        'n_estimators':100,
        'max_features':30,
        'max_depth':8,
        'random_state':42,
        'n_jobs':-1,
        }
    
    model = myModel(random_forest_params)

    run(data, model)