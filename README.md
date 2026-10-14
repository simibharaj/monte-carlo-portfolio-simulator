# Monte Carlo Portfolio Simulator (educational)

Simulates how long a retirement portfolio lasts under random yearly returns, comparing full market exposure with a simplified "floor and cap" strategy. Both strategies face the exact same simulated markets, so the comparison is fair.

## Run it
```
pip install -r requirements.txt -r requirements-dev.txt
python simulator.py
pytest
```
Results are reproducible (fixed random seed, configurable). Set `annual_inflation` in `SimulationConfig` to increase withdrawals each year (for example, `0.02` for 2%); the default `0.0` keeps withdrawals fixed. Returns remain nominal, and inflation adjusts withdrawals only.

## Example output (default settings: $500k, $25k/yr withdrawals, 30 years, 7% mean, 16% volatility, 10,000 runs)
| Annual fee | Full market success | Floor 0% / cap 9.5% success |
|---|---|---|
| 0% | 74.9% | 98.5% |
| 1% | 65.0% | 86.6% |
| 2% | 54.5% | 52.7% |

## Read this before drawing conclusions
This is a **teaching model**, not a product illustration or advice.
- The floor strategy looks very strong with no fees and **loses its advantage as fees rise** (see the 2% row). A free floor does not exist in real products.
- Returns are drawn from a normal distribution. Real markets have fatter tails and different sequences.
- Inflation only adjusts withdrawals when `annual_inflation` is set; returns remain nominal. The model ignores taxes, surrender charges, participation rates and rider terms.
- Do not use these numbers to sell or recommend anything.
