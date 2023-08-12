# Versions
- Python 3.11.3
- Conda 23.5.0
- IPython 8.12.0
- Jupyter Notebook 6.5.4

# Project overview
This project contains a model for forecasting volatility of publicly traded financial instruments. The key component of the model is a fuzzy system.

## Fuzzy Systems
Fuzzy system is a list of fuzzy rules, each of the form 'IF antecedent THEN consequent'. Each rule corresponds to a fuzzy cluster, and a membership degree to such cluster is a real number from [0, 1], determined by some membership function.
