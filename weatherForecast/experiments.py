# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import AIWeatherForecast
import dataPreparation


def experiment(max_days=14):
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


def plot_results():
    df = pd.read_csv("results.csv", index_col=0).transpose()
    temp_df = pd.DataFrame()
    degree_sign = u'\N{DEGREE SIGN}'

    # plot madrid[0-3] vs days tree
    plt.figure(3)
    temp_df['0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['1' + degree_sign + ' accuracy'] = df['madrid tree 1']
    temp_df['2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['3' + degree_sign + ' accuracy'] = df['madrid tree 3']
    temp_df['0' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['1' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['2' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['3' + degree_sign + ' accuracy'].plot(legend=True)
    plt.title('Madrid accuracy for number of days used with tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)


    # # plot madrid[0-3] vs days forest
    # plt.figure(1)
    # temp_df['0' + degree_sign + ' accuracy'] = df['madrid forest 0']
    # temp_df['1' + degree_sign + ' accuracy'] = df['madrid forest 1']
    # temp_df['2' + degree_sign + ' accuracy'] = df['madrid forest 2']
    # temp_df['3' + degree_sign + ' accuracy'] = df['madrid forest 3']
    # temp_df['0' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['1' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['2' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['3' + degree_sign + ' accuracy'].plot(legend=True)
    # plt.title('Madrid accuracy for number of days used with forest clf')
    # plt.xlabel('Days')
    # plt.xticks(range(14), range(1, 15))
    # plt.ylabel('Accuracy[%]')
    # plt.legend(loc=4)

    # plot madrid[0,2] vs days - tree vs forest
    temp_df['tree 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['forest 0' + degree_sign + ' accuracy'] = df['madrid forest 0']
    temp_df['tree 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['forest 2' + degree_sign + ' accuracy'] = df['madrid forest 2']
    temp_df['tree 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['forest 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['tree 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['forest 2' + degree_sign + ' accuracy'].plot(style='g')

    # # plot austin[0-3] vs days
    # plt.figure(2)
    # temp_df['0' + degree_sign + ' accuracy'] = df['austin forest 0']
    # temp_df['1' + degree_sign + ' accuracy'] = df['austin forest 1']
    # temp_df['2' + degree_sign + ' accuracy'] = df['austin forest 2']
    # temp_df['3' + degree_sign + ' accuracy'] = df['austin forest 3']
    # temp_df['0' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['1' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['2' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['3' + degree_sign + ' accuracy'].plot(legend=True)
    # plt.title('Austin accuracy for number of days used with forest clf')
    # plt.xlabel('Days')
    # plt.xticks(range(14), range(1, 15))
    # plt.ylabel('Accuracy[%]')
    # plt.legend(loc=4)


    # # plot austin[0-3] vs days
    # plt.figure(4)
    # temp_df['0' + degree_sign + ' accuracy'] = df['austin tree 0']
    # temp_df['1' + degree_sign + ' accuracy'] = df['austin tree 1']
    # temp_df['2' + degree_sign + ' accuracy'] = df['austin tree 2']
    # temp_df['3' + degree_sign + ' accuracy'] = df['austin tree 3']
    # temp_df['0' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['1' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['2' + degree_sign + ' accuracy'].plot(legend=True)
    # temp_df['3' + degree_sign + ' accuracy'].plot(legend=True)
    # plt.title('Austin accuracy for number of days used with tree clf')
    # plt.xlabel('Days')
    # plt.xticks(range(14), range(1, 15))
    # plt.ylabel('Accuracy[%]')
    # plt.legend(loc=4)

    # plot madrid, austin temp histogram
    madrid_df = pd.read_csv(dataPreparation.raw_madrid_data_file, index_col=0)
    austin_df = pd.read_csv(dataPreparation.raw_austin_data_file, index_col=0)
    fig, ax = plt.subplots()
    madrid_temps, madrid_bins = np.histogram(madrid_df["Mean TemperatureC"].dropna(), bins=40, normed=True)
    austin_temps, austin_bins = np.histogram(austin_df["Mean TemperatureC"].dropna(), bins=40, normed=True)
    width = (madrid_bins[1] - madrid_bins[0]) / 2
    ax.bar(madrid_bins[:-1], madrid_temps, width=width, facecolor='cornflowerblue')
    ax.bar(austin_bins[:-1] + width, austin_temps, width=width, facecolor='seagreen')
    plt.title('Madrid (blue) and Austin (green) temperature histogram')
    plt.xlabel('temperature[C'+ degree_sign + ']')

    # plot madrid[0,2], austin[0,2] vs days
    plt.figure(5)
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['austin 0' + degree_sign + ' accuracy'] = df['austin tree 0']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['austin 2' + degree_sign + ' accuracy'] = df['austin tree 2']
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['austin 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['austin 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Mdrid and Austin accuracy comparisson with tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    plt.show()


if __name__ == "__main__":
    #experiment()
    plot_results()

