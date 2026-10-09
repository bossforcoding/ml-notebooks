# Probabilistic machine learning

Models that represent uncertainty explicitly: Gaussian processes, Bayesian optimization and Bayesian networks. The core algorithms are implemented from scratch in two small modules, so every step is visible.

| Notebook | Topics | Data |
|---|---|---|
| [01 · Gaussian processes](01_gaussian_processes.ipynb) | kernels as priors over functions, exact posterior, marginal likelihood and hyperparameter learning, kernel comparison with coverage and log predictive density | synthetic functions, a small nonlinear regression problem |
| [02 · Bayesian optimization](02_bayesian_optimization.ipynb) | expected improvement, UCB and Thompson sampling, regret vs random search, a real design experiment with measurement noise, constraints and confounded factors | Forrester test function, paper helicopter flight times |
| [03 · Bayesian network](03_bayesian_network.ipynb) | expert DAG, d-separation and explaining away, structure learning with BIC and hill climbing, Dirichlet parameter estimation, exact inference vs logic sampling | survey on eating habits, physical condition and obesity |

## Code

- `gp.py`: GP regression (RBF, Matern 5/2, periodic and linear kernels), hyperparameter fitting by marginal likelihood with optional known noise, posterior sampling, expected improvement and UCB.
- `bayesnet.py`: BIC scoring, hill-climbing structure learning, Bayesian parameter estimation, exact inference on the joint table and forward sampling for discrete networks.

## Highlights

- **The marginal likelihood picks the model complexity on its own** (notebook 01), and held-out metrics show where it can be misled by small data.
- **Bayesian optimization finds the optimum in 12 evaluations** in most runs, where random search rarely does (notebook 02).
- **Measured noise beats estimated noise**: fixing the noise from repeated flights keeps the GP from mistaking stopwatch error for structure, and the second round of experiments halves the model's prediction error (notebook 02).
- **A faithful model of a flawed dataset** (notebook 03): the network reproduces both a strong family-history effect and a spurious "snacking protects against obesity" pattern created by synthetic data.

## Data

The obesity dataset is the *Estimation of obesity levels based on eating habits and physical condition* dataset (Palechor & de la Hoz Manotas, 2019), available from the UCI Machine Learning Repository under CC BY 4.0. The paper helicopter flights were measured in a group experiment.
