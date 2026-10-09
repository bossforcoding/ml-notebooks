# Time series

From describing a series to forecasting it: decomposition, stationarity, autoregressive and ARMA models, and spectral analysis, applied to economic, physical and environmental data.

| Notebook | Topics | Datasets |
|---|---|---|
| [01 · Trend and seasonality](01_trend_and_seasonality.ipynb) | log growth rates, moving averages, seasonal boxplots, classical and STL decomposition, Box-Cox transformation, ACF of the remainder | electric cars in Swiss cantons, Australian beer and electricity production, global temperature |
| [02 · Stationarity and AR models](02_stationarity_and_ar_models.ipynb) | weak stationarity, ACF derived by hand, unit roots, random walks, ACF/PACF of AR processes, differencing and ARIMA with drift, out-of-sample forecasts | simulated processes, global temperature, force on a cylinder in a water tank |
| [03 · ARMA identification and forecasting](03_arma_identification_and_forecasting.ipynb) | identification from ACF/PACF, AIC vs BIC, Ljung-Box test, forecast evaluation against naive baselines | simulated AR/MA/ARMA processes, series of unknown origin, sunspot area |
| [04 · Spectral analysis](04_spectral_analysis.ipynb) | Fourier series and the Gibbs phenomenon, FFT amplitude spectrum, harmonics, filtering by thresholding | square wave, Swiss electricity consumption 2022 |

## Highlights

- **A stationary model forecasts the end of global warming** (notebook 02): an AR model on temperature levels reverts to the long-run mean, while modelling the yearly changes keeps the trend (about 1 degree per century since 1850).
- **Mixed models are more parsimonious** (notebook 03): an ARMA(2,1) fits better than an AR(4) or MA(3) with fewer parameters.
- **Common sense is a tough benchmark** (notebook 03): repeating the last 11-year solar cycle forecasts sunspots as well as a carefully selected AR(10).
- **The spectrum finds the working week, the residuals find the holidays** (notebook 04): a dozen Fourier coefficients explain 90% of Swiss electricity consumption, and the days they miss are public holidays.

All notebooks are saved with their outputs and can be read directly on GitHub. To run them, see the [setup instructions](../README.md#setup).
