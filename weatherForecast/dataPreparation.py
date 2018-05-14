import pandas as pd
import numpy as np


def data_preparator():
    important_features = ['CET', 'Max TemperatureC', 'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', ' Mean Humidity', ' Min Humidity',
        ' Max Sea Level PressurehPa', ' Mean Sea Level PressurehPa', ' Min Sea Level PressurehPa', ' Max VisibilityKm',
        ' Mean VisibilityKm', ' Min VisibilitykM', ' Max Wind SpeedKm/h', ' Mean Wind SpeedKm/h', 'Precipitationmm',
        ' CloudCover', 'WindDirDegrees']
    raw_data = pd.read_csv('../madridDataBase/weather_madrid_LEMD_1997_2015.csv', sep=',', low_memory=False)
    raw_data = raw_data[important_features]
    raw_data.to_csv(path_or_buf='../madridDataBase/FilteredRawDate.csv', index=False)


if __name__ == "__main__":
    data_preparator()

