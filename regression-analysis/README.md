# Regression analysis

Linear regression from first principles to model selection, worked through on small real-world datasets. Every notebook pairs the statistical technique with a concrete question and interprets the results in context.

| Notebook | Topics | Datasets |
|---|---|---|
| [01 · Simple linear regression](01_simple_linear_regression.ipynb) | least squares, transformations of the predictor, confidence vs prediction intervals, outliers, Monte Carlo check of standard errors | Anscombe's quartet, antique clocks, windmill power, Forbes' boiling point |
| [02 · Residual diagnostics](02_residual_diagnostics.ipynb) | the four diagnostic plots, simulated reference bands, choosing transformations (reciprocal, log-log, exponential decay) | simulated data, windmill, rainfall–runoff, bacteria under X-rays |
| [03 · Multiple regression](03_multiple_regression.ipynb) | correlated predictors, leverage vs influence, hidden groups of outliers and robust regression, dummy variables and interactions, partial F-tests | catheter length, savings rates across countries, synthetic data, lathe surface finish, farm revenue |
| [04 · Multicollinearity and model selection](04_multicollinearity_model_selection.ipynb) | variance inflation factors, dropping vs combining variables, stepwise selection with AIC/BIC | fitness and oxygen uptake, hospital length of stay, fund of hedge funds |
| [05 · Subset selection, ridge and lasso](05_subset_selection_ridge_lasso.ipynb) | exhaustive best subset search, ridge and lasso with cross-validation, when regularization pays off, permutation importance | US college applications |

## Highlights

- **A group of outliers that diagnostics miss** (notebook 03): the residual plots look fine, a robust fit with Tukey's biweight reveals that 8 observations reverse the sign of a coefficient.
- **Recovering a hidden portfolio** (notebook 04): stepwise selection on hedge fund strategy returns identifies the five strategies a fund of funds invests in, with weights and fees that make financial sense.
- **When regularization helps** (notebook 05): with the full dataset least squares, ridge and lasso perform alike; with 25–40 training samples the lasso cuts the test error by about 30%.
- **A predictor that makes the model useless** (notebook 05): the most important feature is only known after the event being predicted.

## Code

`diagnostics.py` contains the helpers shared by the notebooks:

- `diagnostic_plots(model)`: residuals vs fitted, normal Q-Q, scale-location and residuals vs leverage, with reference bands simulated from a correct model;
- `vif_table(X)`: variance inflation factors;
- `forward_selection` / `backward_selection`: stepwise selection on formula terms (factors are added or removed as a whole) using AIC or BIC.

All notebooks are saved with their outputs and can be read directly on GitHub. To run them, see the [setup instructions](../README.md#setup).
