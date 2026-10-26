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
    return_distribution: str = "normal"  # "normal" or "student_t"
    student_t_df: float = 5.0  # used only for student_t; must exceed 2
    annual_fee: float = 0.0   # deducted from the credited return each year
    annual_inflation: float = 0.0  # compounds withdrawals; 0 preserves fixed withdrawals
    seed: Optional[int] = 42  # fixed seed so results are reproducible

    def __post_init__(self):
        if self.return_distribution not in {"normal", "student_t"}:
            raise ValueError("return_distribution must be 'normal' or 'student_t'")
        if self.return_distribution == "student_t" and (
            not np.isfinite(self.student_t_df) or self.student_t_df <= 2
        ):
            raise ValueError("student_t_df must be finite and > 2 for finite variance")


class RetirementSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        rng = np.random.default_rng(self.cfg.seed)
        shape = (self.cfg.simulations, self.cfg.years)
        if self.cfg.return_distribution == "normal":
            sampled_returns = rng.normal(self.cfg.mean_return, self.cfg.volatility, size=shape)
        else:
            # Standardize Student-t draws to unit variance before applying the
            # configured mean and volatility. A simple return cannot fall below -100%.
            draws = rng.standard_t(df=self.cfg.student_t_df, size=shape)
            draws *= np.sqrt((self.cfg.student_t_df - 2) / self.cfg.student_t_df)
            sampled_returns = self.cfg.mean_return + self.cfg.volatility * draws
        # A return below -100% is an impossible loss of more than the investment;
        # model it as a total loss instead.
        self.returns = np.maximum(sampled_returns, -1.0)

    def _run(self, credited: np.ndarray, label: str) -> Dict[str, float]:
        c = self.cfg
        balance = np.full(c.simulations, c.starting_balance, dtype=float)
        ruined = np.zeros(c.simulations, dtype=bool)
        for y in range(c.years):
            withdrawal = c.annual_withdrawal * (1.0 + c.annual_inflation) ** y
            balance = (balance - withdrawal) * (1.0 + credited[:, y] - c.annual_fee)
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
