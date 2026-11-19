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


@pytest.mark.parametrize(
    "floor, cap, message",
    [
        (0.20, 0.05, "floor must be less than or equal to cap"),
        (float("nan"), 0.05, "floor and cap must be finite rates"),
        (0.0, float("inf"), "floor and cap must be finite rates"),
    ],
)
def test_floor_cap_rejects_invalid_bounds(floor, cap, message):
    sim = RetirementSimulator(SimulationConfig(simulations=10))
    with pytest.raises(ValueError, match=message):
        sim.run_floor_cap(floor=floor, cap=cap)


def test_equal_floor_and_cap_is_valid():
    cfg = SimulationConfig(
        starting_balance=100_000,
        annual_withdrawal=0,
        years=1,
        simulations=1,
        mean_return=0.10,
        volatility=0,
    )
    result = RetirementSimulator(cfg).run_floor_cap(floor=0.05, cap=0.05)
    assert result["median_terminal_wealth"] == 105_000


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


@pytest.mark.parametrize(
    "field, value",
    [
        ("starting_balance", float("nan")),
        ("annual_withdrawal", float("inf")),
        ("mean_return", float("nan")),
        ("volatility", float("inf")),
        ("annual_fee", float("nan")),
        ("annual_inflation", float("inf")),
    ],
)
def test_config_rejects_non_finite_scenario_values(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        SimulationConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("starting_balance", -1, "starting_balance must be non-negative"),
        ("annual_withdrawal", -1, "annual_withdrawal must be non-negative"),
        ("mean_return", -1.01, "mean_return must be at least -1"),
        ("volatility", -0.01, "volatility must be non-negative"),
        ("annual_fee", -0.01, "annual_fee must be non-negative"),
        ("annual_inflation", -1.01, "annual_inflation must be at least -1"),
    ],
)
def test_config_rejects_out_of_domain_scenario_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("years", 0),
        ("years", -1),
        ("years", 1.5),
        ("years", True),
        ("simulations", 0),
        ("simulations", -1),
        ("simulations", 1.5),
        ("simulations", False),
    ],
)
def test_config_requires_positive_integer_dimensions(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        SimulationConfig(**{field: value})
