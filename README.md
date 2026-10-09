# Machine learning & data analysis notebooks

A collection of Jupyter notebooks on statistics, machine learning and AI. Each one tackles a concrete question on real data, explains the method step by step and interprets the results.

| Folder | Content |
|---|---|
| [regression-analysis](regression-analysis/) | Linear regression: diagnostics, transformations, multiple regression, multicollinearity, model selection, ridge and lasso |
| [classification](classification/) | Logistic regression, class imbalance and ROC curves, decision trees and pruning, bootstrap, bagging and random forests |
| [time-series](time-series/) | Trend and seasonality, stationarity, AR/ARMA/ARIMA models and forecast evaluation, spectral analysis with the FFT |
| [deep-learning](deep-learning/) | Gradient descent, MLPs, CNNs on CIFAR-10, an LSTM language model, and a pointer Transformer for the travelling salesman problem |
| [probabilistic-ml](probabilistic-ml/) | Gaussian processes, Bayesian optimization of a real experiment, and a Bayesian network learned from survey data |
| [computer-vision](computer-vision/) | Image processing fundamentals, a document scanner that reads handwritten digits, and YOLOv8 license plate detection |

All notebooks are stored with their outputs, so they can be read directly on GitHub without running anything.

## Setup

```bash
git clone https://github.com/bossforcoding/ml-notebooks.git
cd ml-notebooks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Each folder is self-contained: open a notebook from its folder so the relative paths to `data/` and to the helper modules work.
