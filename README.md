# Machine learning & data analysis notebooks

A collection of Jupyter notebooks on statistics, machine learning and AI. Each one tackles a concrete question on real data, explains the method step by step and interprets the results.

| Folder | Content |
|---|---|
| [regression-analysis](regression-analysis/) | Linear regression: diagnostics, transformations, multiple regression, multicollinearity, model selection, ridge and lasso |

More folders will follow.

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
