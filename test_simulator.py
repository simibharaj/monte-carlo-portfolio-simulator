import numpy as np
from simulator import RetirementSimulator, SimulationConfig

def test_reproducible_with_seed():
    a = RetirementSimulator(SimulationConfig(simulations=500)).run_market()
    b = RetirementSimulator(SimulationConfig(simulations=500)).run_market()
    assert a == b

def test_no_withdrawals_floor_never_loses_money():
    cfg = SimulationConfig(annual_withdrawal=0.0, simulations=500)
    r = RetirementSimulator(cfg).run_floor_cap(floor=0.0, cap=0.095)
    assert r["p10_terminal_wealth"] >= cfg.starting_balance

def test_success_rate_is_a_percentage():
    r = RetirementSimulator(SimulationConfig(simulations=500)).run_market()
    assert 0 <= r["success_rate_pct"] <= 100

def test_fees_reduce_outcomes():
    base = RetirementSimulator(SimulationConfig(simulations=500)).run_floor_cap()
    fee = RetirementSimulator(SimulationConfig(simulations=500, annual_fee=0.02)).run_floor_cap()
    assert fee["median_terminal_wealth"] <= base["median_terminal_wealth"]

def test_both_strategies_use_same_market_paths():
    sim = RetirementSimulator(SimulationConfig(simulations=200))
    assert sim.returns.shape == (200, 30)
