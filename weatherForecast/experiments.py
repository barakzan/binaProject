# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import datetime
import AIWeatherForecast


def experiment(max_days=10):
    df = pd.DataFrame(np.zeros([16, max_days]), columns=[str(i) for i in range(1, max_days + 1)],
                      index=['madrid forest 0', 'austin forest 0', 'madrid tree 0', 'austin tree 0',
                             'madrid forest 1', 'austin forest 1', 'madrid tree 1', 'austin tree 1',
                             'madrid forest 2', 'austin forest 2', 'madrid tree 2', 'austin tree 2',
                             'madrid forest 3', 'austin forest 3', 'madrid tree 3', 'austin tree 3'])
    for i in range(1, max_days + 1):
        forest_res = AIWeatherForecast.forecast(i, use_forest_instead_of_tree=True)
        madrid_forest_res = forest_res[0]
        austin_forest_res = forest_res[1]
        tree_res = AIWeatherForecast.forecast(i, use_forest_instead_of_tree=False)
        madrid_tree_res = tree_res[0]
        austin_tree_res = tree_res[1]
        for j in range(0, 4):
            df.loc['madrid forest ' + str(j), str(i)] = round(madrid_forest_res[j], 2)
            df.loc['austin forest ' + str(j), str(i)] = round(austin_forest_res[j], 2)
            df.loc['madrid tree ' + str(j), str(i)] = round(madrid_tree_res[j], 2)
            df.loc['austin tree ' + str(j), str(i)] = round(austin_tree_res[j], 2)

    df.to_csv("results.csv")


if __name__ == "__main__":
    experiment()

