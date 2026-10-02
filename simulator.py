"""
Monte Carlo retirement withdrawal simulator (educational).

Compares a portfolio fully exposed to market returns with a simplified
"capped and floored" strategy, over randomized yearly return sequences.
This is a teaching model. It is NOT a product illustration or advice.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SimulationConfig:
    starting_balance: float = 500_000.0
    annual_withdrawal: float = 25_000.0
    years: int = 30
    simulations: int = 10_000
    mean_return: float = 0.07
    volatility: float = 0.16
    annual_fee: float = 0.0   # deducted from the credited return each year
    seed: Optional[int] = 42  # fixed seed so results are reproducible


class RetirementSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        rng = np.random.default_rng(self.cfg.seed)
        # One shared set of return paths, so both strategies face identical markets.
        self.returns = rng.normal(self.cfg.mean_return, self.cfg.volatility,
                                  size=(self.cfg.simulations, self.cfg.years))

    def _run(self, credited: np.ndarray, label: str) -> Dict[str, float]:
        c = self.cfg
        balance = np.full(c.simulations, c.starting_balance, dtype=float)
        ruined = np.zeros(c.simulations, dtype=bool)
        for y in range(c.years):
            balance = (balance - c.annual_withdrawal) * (1.0 + credited[:, y] - c.annual_fee)
            ruined |= balance <= 0
            balance = np.where(ruined, 0.0, balance)
        return {
            "strategy": label,
            "success_rate_pct": round(float((~ruined).mean() * 100), 2),
            "median_terminal_wealth": round(float(np.median(balance)), 2),
            "p10_terminal_wealth": round(float(np.percentile(balance, 10)), 2),
        }

    def run_market(self) -> Dict[str, float]:
        return self._run(self.returns, "Full market exposure")

    def run_floor_cap(self, floor: float = 0.0, cap: float = 0.095) -> Dict[str, float]:
        credited = np.clip(self.returns, floor, cap)
        return self._run(credited, f"Floor {floor*100:.1f}% / cap {cap*100:.1f}%")


if __name__ == "__main__":
    sim = RetirementSimulator()
    for r in (sim.run_market(), sim.run_floor_cap()):
        print(r)
