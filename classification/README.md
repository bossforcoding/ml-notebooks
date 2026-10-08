# Classification

Logistic regression, decision trees and tree ensembles, each applied to a real question and evaluated honestly with cross-validation and held-out test sets.

| Notebook | Topics | Datasets |
|---|---|---|
| [01 · Logistic regression](01_logistic_regression.ipynb) | log-odds and odds ratios, confounding that reverses a coefficient, class imbalance and threshold choice, training vs CV vs test error, precision/recall/F1, ROC curve and AUC computed by hand | credit card default, fuel efficiency of cars |
| [02 · Decision trees](02_decision_trees.ipynb) | impurity measures, decision boundaries, overfitting, cost-complexity pruning with cross-validation, classification vs regression trees | transmission type of cars, orange juice purchases, fuel efficiency |
| [03 · Bootstrap, bagging and random forests](03_bootstrap_bagging_random_forests.ipynb) | the bootstrap validated against simulation, out-of-bag error, bagging vs random forests, model comparison, impurity vs permutation importance | simulated asset returns, breast tumour diagnosis |

## Highlights

- **Students look riskier, but are safer** (notebook 01): student status raises the default rate on its own and lowers it once the credit card balance is accounted for.
- **97% accuracy that misses 69% of the defaults** (notebook 01): with 3% positives, accuracy hides a weak classifier; lowering the threshold catches three quarters of them.
- **A 4-leaf tree beats a 173-leaf tree** (notebook 02): pruning by cross-validation gives a simpler, more accurate and actionable rule (loyalty first, then price).
- **The bootstrap recovers the true standard error from one sample** (notebook 03): 0.00844 vs 0.00846.
- **A random forest is not automatically better** (notebook 03): on breast tumour diagnosis, logistic regression performs just as well.

All notebooks are saved with their outputs and can be read directly on GitHub. To run them, see the [setup instructions](../README.md#setup).
