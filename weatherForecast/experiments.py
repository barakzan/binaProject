# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import datetime
import AIWeatherForecast


def experiment():
    results = []
    for i in range(1, 15):
        results[i] = AIWeatherForecast.forecast(i)


if __name__ == "__main__":
    experiment()

