# Precipitation Timeseries Analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"> <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /> </a> <img src="https://img.shields.io/badge/Python-3.8%2B-blue" /> <img src="https://img.shields.io/badge/Library-Pandas%20%7C%20Matplotlib%20%7C%20Statsmodels-orange" /> <img src="https://img.shields.io/badge/Analysis-Time%20Series-brightgreen" />
</a>

A comprehensive time series analysis of monthly precipitation data for Santiago de Cuba, Cuba. This project provides statistical methods and visualization tools for hydrological data analysis.

📊 Features

    Trend Analysis: Mann-Kendall test for trend detection

    Seasonality Analysis: Monthly patterns and visualizations

    Stationarity Testing: Augmented Dickey-Fuller test

    Homogeneity Analysis: SNHT test for change point detection

    Non-linearity Testing: BDS test for nonlinear patterns

    Autocorrelation Analysis: ACF plots and analysis

    Periodicity Verification: Time interval consistency checks

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Main analysis functions
    │
    └── timeseries-analysis.py  
```

--------
