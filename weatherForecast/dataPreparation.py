import pandas as pd
import numpy as np


def data_preparator():
    important_features = ['ELEVATION', 'LATITUDE', 'LONGITUDE', 'DATE', 'DAILYMaximumDryBulbTemp',
        'DAILYMinimumDryBulbTemp', 'DAILYAverageDryBulbTemp', 'DAILYDeptFromNormalAverageTemp',
        'DAILYAverageRelativeHumidity', 'DAILYAverageDewPointTemp', 'DAILYAverageWetBulbTemp', 'DAILYHeatingDegreeDays',
        'DAILYCoolingDegreeDays', 'DAILYSunrise', 'DAILYSunset', 'DAILYPrecip', 'DAILYSnowfall', 'DAILYSnowDepth',
        'DAILYAverageStationPressure', 'DAILYAverageWindSpeed', 'DAILYPeakWindSpeed', 'PeakWindDirection',
        'DAILYSustainedWindSpeed', 'DAILYSustainedWindDirection']
    raw_data = pd.read_csv('../DataBases/rawDataBasebeforeBasicFiltering.csv', sep=',', low_memory=False)
    raw_data = raw_data.dropna(subset=['DAILYSnowfall'], axis=0)
    raw_data = raw_data[important_features]
    raw_data.to_csv(path_or_buf='../DataBases/FilteredrawDate.csv', index=False)


if __name__ == "__main__":
    data_preparator()

