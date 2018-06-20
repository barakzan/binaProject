# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import datetime
import AIWeatherForecast


def experiment(max_days=14):
    df = pd.DataFrame(np.zeros([8, max_days]), columns=[str(i) for i in range(1, max_days + 1)],
                      index=['madrid 0', 'austin 0', 'madrid 1', 'austin 1', 'madrid 2', 'austin 2',
                             'madrid 3', 'austin 3'])
    for i in range(1, max_days + 1):
        res = AIWeatherForecast.forecast(i)
        madrid_res = res[0]
        austin_res = res[1]
        for j in range(0, 4):
            df.loc['madrid ' + str(j), str(i)] = round(madrid_res[j], 2)
            df.loc['austin ' + str(j), str(i)] = round(austin_res[j], 2)

    df.to_csv("results.csv")


if __name__ == "__main__":
    experiment()

