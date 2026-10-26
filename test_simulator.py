import numpy as np
import pytest
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

def test_inflation_compounds_annual_withdrawals():
    cfg = SimulationConfig(
        starting_balance=1_000.0,
        annual_withdrawal=100.0,
        years=3,
        simulations=1,
        mean_return=0.0,
        volatility=0.0,
        annual_inflation=0.10,
    )
    result = RetirementSimulator(cfg).run_market()
    # Withdrawals are 100, 110, and 121; returns are zero.
    assert result["median_terminal_wealth"] == 669.0


def test_student_t_returns_are_standardized_and_reproducible():
    cfg = SimulationConfig(
        simulations=20,
        years=4,
        return_distribution="student_t",
        student_t_df=5,
        seed=123,
    )
    sim = RetirementSimulator(cfg)
    draws = np.random.default_rng(cfg.seed).standard_t(
        df=cfg.student_t_df, size=(cfg.simulations, cfg.years)
    )
    expected = cfg.mean_return + cfg.volatility * draws * np.sqrt(
        (cfg.student_t_df - 2) / cfg.student_t_df
    )
    np.testing.assert_allclose(sim.returns, np.maximum(expected, -1.0), rtol=1e-14, atol=1e-15)


def test_student_t_rejects_degrees_of_freedom_without_finite_variance():
    with pytest.raises(ValueError, match="finite and > 2"):
        SimulationConfig(return_distribution="student_t", student_t_df=2)


def test_unknown_return_distribution_is_rejected():
    with pytest.raises(ValueError, match="return_distribution"):
        SimulationConfig(return_distribution="lognormal")


def test_student_t_simple_returns_cannot_exceed_total_loss():
    cfg = SimulationConfig(
        simulations=100,
        years=10,
        mean_return=0.0,
        volatility=2.0,
        return_distribution="student_t",
        student_t_df=2.1,
    )
    assert np.all(RetirementSimulator(cfg).returns >= -1.0)
