# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import AIWeatherForecast
import dataPreparation


def experiment(max_days=14, max_depth=10):
    df = pd.DataFrame(np.zeros([48, max_days]), columns=[str(i) for i in range(1, max_days + 1)],
                      index=['madrid forest 0', 'austin forest 0', 'joined forest 0','madrid kfold forest 0',
                             'madrid tree 0', 'austin tree 0', 'joined tree 0', 'madrid kfold tree 0',
                             'madrid linear 0', 'austin linear 0', 'joined linear 0', 'madrid kfold linear 0',
                             'madrid forest 1', 'austin forest 1', 'joined forest 1', 'madrid kfold forest 1',
                             'madrid tree 1', 'austin tree 1', 'joined tree 1', 'madrid kfold tree 1',
                             'madrid linear 1', 'austin linear 1', 'joined linear 1', 'madrid kfold linear 1',
                             'madrid forest 2', 'austin forest 2', 'joined forest 2', 'madrid kfold forest 2',
                             'madrid tree 2', 'austin tree 2', 'joined tree 2', 'madrid kfold tree 2',
                             'madrid linear 2', 'austin linear 2', 'joined linear 2', 'madrid kfold linear 2',
                             'madrid forest 3', 'austin forest 3', 'joined forest 3', 'madrid kfold forest 3',
                             'madrid tree 3', 'austin tree 3', 'joined tree 3', 'madrid kfold tree 3',
                             'madrid linear 3', 'austin linear 3', 'joined linear 3', 'madrid kfold linear 3'])
    for i in range(1, max_days + 1):
        forest_res = AIWeatherForecast.forecast(i, classifier_type='forest', max_depth=max_depth)
        madrid_forest_res = forest_res[0]
        austin_forest_res = forest_res[1]
        joined_forest_res = forest_res[2]
        madrid_kfold_forest_res = forest_res[3]
        tree_res = AIWeatherForecast.forecast(i, classifier_type='tree', max_depth=max_depth)
        madrid_tree_res = tree_res[0]
        austin_tree_res = tree_res[1]
        joined_tree_res = tree_res[2]
        madrid_kfold_tree_res = tree_res[3]
        linear_res = AIWeatherForecast.forecast(i, classifier_type='linear', max_depth=max_depth)
        madrid_linear_res = linear_res[0]
        austin_linear_res = linear_res[1]
        joined_linear_res = linear_res[2]
        madrid_kfold_linear_res = linear_res[3]

        for j in range(0, 4):
            df.loc['madrid forest ' + str(j), str(i)] = round(madrid_forest_res[j], 2)
            df.loc['austin forest ' + str(j), str(i)] = round(austin_forest_res[j], 2)
            df.loc['joined forest ' + str(j), str(i)] = round(joined_forest_res[j], 2)
            df.loc['madrid kfold forest ' + str(j), str(i)] = round(joined_forest_res[j], 2)
            df.loc['madrid tree ' + str(j), str(i)] = round(madrid_tree_res[j], 2)
            df.loc['austin tree ' + str(j), str(i)] = round(austin_tree_res[j], 2)
            df.loc['joined tree ' + str(j), str(i)] = round(joined_tree_res[j], 2)
            df.loc['madrid kfold tree ' + str(j), str(i)] = round(joined_tree_res[j], 2)
            df.loc['madrid linear ' + str(j), str(i)] = round(madrid_linear_res[j], 2)
            df.loc['austin linear ' + str(j), str(i)] = round(austin_linear_res[j], 2)
            df.loc['joined linear ' + str(j), str(i)] = round(joined_linear_res[j], 2)
            df.loc['madrid kfold linear ' + str(j), str(i)] = round(joined_linear_res[j], 2)

    df.to_csv("results.csv")


def plot_results():
    df = pd.read_csv("results.csv", index_col=0).transpose()
    temp_df = pd.DataFrame()
    degree_sign = u'\N{DEGREE SIGN}'

    # plot madrid[0-3] vs days tree
    plt.figure(1)
    temp_df['0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['1' + degree_sign + ' accuracy'] = df['madrid tree 1']
    temp_df['2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['3' + degree_sign + ' accuracy'] = df['madrid tree 3']
    temp_df['0' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['1' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['2' + degree_sign + ' accuracy'].plot(legend=True)
    temp_df['3' + degree_sign + ' accuracy'].plot(legend=True)
    plt.title('Madrid accuracy degees comparisson for number of days used with tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    # plot madrid[0,2] vs days - tree vs forest
    plt.figure(2)
    temp_df['tree 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['forest 0' + degree_sign + ' accuracy'] = df['madrid forest 0']
    temp_df['linear 0' + degree_sign + ' accuracy'] = df['madrid linear 0']

    temp_df['linear 1' + degree_sign + ' accuracy'] = df['madrid linear 1']

    temp_df['tree 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['forest 2' + degree_sign + ' accuracy'] = df['madrid forest 2']
    temp_df['linear 2' + degree_sign + ' accuracy'] = df['madrid linear 2']
    temp_df['tree 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['forest 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['linear 0' + degree_sign + ' accuracy'].plot(style='b^-')

    temp_df['linear 1' + degree_sign + ' accuracy'].plot(style='b+')

    temp_df['tree 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['forest 2' + degree_sign + ' accuracy'].plot(style='g')
    temp_df['linear 2' + degree_sign + ' accuracy'].plot(style='b')

    plt.title('Madrid accuracy comparision tree clf vs forest clf vs linear clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    # plot madrid when trained on both  vs days tree clf
    plt.figure(3)
    temp_df['joined 0' + degree_sign + ' accuracy'] = df['joined tree 0']
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['joined 2' + degree_sign + ' accuracy'] = df['joined tree 2']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['joined 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['joined 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Madrid accuracy comparisson between trained on austin and madrid data vs only madrid data\n for number of days used with tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    # plot madrid when trained on both  vs days forest clf
    plt.figure(4)
    temp_df['joined 0' + degree_sign + ' accuracy'] = df['joined forest 0']
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid forest 0']
    temp_df['joined 2' + degree_sign + ' accuracy'] = df['joined forest 2']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid forest 2']
    temp_df['joined 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['joined 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Madrid accuracy comparisson between trained on austin and madrid data vs only madrid data\n for number of days used with forest clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    # plot madrid[0,2], austin[0,2] vs days tree clf
    plt.figure(5)
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['austin 0' + degree_sign + ' accuracy'] = df['austin tree 0']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['austin 2' + degree_sign + ' accuracy'] = df['austin tree 2']
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['austin 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['austin 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Madrid and Austin accuracy comparision when trained on madrid data only\nwith tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    # plot madrid[0,2], austin[0,2] vs days forest clf
    plt.figure(6)
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid forest 0']
    temp_df['austin 0' + degree_sign + ' accuracy'] = df['austin forest 0']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid forest 2']
    temp_df['austin 2' + degree_sign + ' accuracy'] = df['austin forest 2']
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['austin 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['austin 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Madrid and Austin accuracy comparision when trained on madrid data only\n with forest clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

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

    # plot madrid[0,2], austin[0,2] vs days tree clf
    plt.figure(8)
    temp_df['madrid 0' + degree_sign + ' accuracy'] = df['madrid tree 0']
    temp_df['madrid kfold 0' + degree_sign + ' accuracy'] = df['madrid kfold tree 0']
    temp_df['madrid 2' + degree_sign + ' accuracy'] = df['madrid tree 2']
    temp_df['madrid kfold 2' + degree_sign + ' accuracy'] = df['madrid kfold tree 2']
    temp_df['madrid 0' + degree_sign + ' accuracy'].plot(style='r^-')
    temp_df['madrid kfold 0' + degree_sign + ' accuracy'].plot(style='g^-')
    temp_df['madrid 2' + degree_sign + ' accuracy'].plot(style='r')
    temp_df['madrid kfold 2' + degree_sign + ' accuracy'].plot(style='g')
    plt.title('Madrid accuracy comparision when trained on with or without kfold\nwith tree clf')
    plt.xlabel('Days')
    plt.xticks(range(14), range(1, 15))
    plt.ylabel('Accuracy[%]')
    plt.legend(loc=4)

    plt.show()


if __name__ == "__main__":
    # experiment()
    # plot_results()
    AIWeatherForecast.final_project_results(days_before=3, classifier_type='tree')
