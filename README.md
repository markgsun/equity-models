# equity-models
Equity trading and investment models

## Statistical Arbitrage Trading Model
Contains full stat arb-based trading model, including:

* alpha_model which calculates alpha exposure of cross-section of stocks at certain time
* beta_model which calculates beta exposure of cross-section of stocks
* sigma_model which calculates covariance matrix after shrinkage
* optimize_port which optimizes portfolio based on alpha, sigma, beta
* trade_model which encapsulates portfolio construction

*Sources data from local FinancialData database*

Current features:
* Momentum, volume, and book to market premium alphas
* No transaction cost optimization

## Investment Model
Theoretical long-short investment model, featuring:

* Alpha exposures based on same model as statistical arbitrage model
* Quantile based binned portfolio backtests on highest and lowest alpha exposures

*Sources data from local FinancialData database*