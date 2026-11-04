import pytest

from plot_fee_sensitivity import plot_fee_sensitivity, run_fee_sensitivity
from simulator import RetirementSimulator, SimulationConfig


def test_fee_sweep_is_sorted_and_reproducible():
    first = run_fee_sensitivity([0.02, 0.0, 0.01], simulations=300, seed=17)
    second = run_fee_sensitivity([0.0, 0.01, 0.02], simulations=300, seed=17)

    assert first == second
    assert [row["annual_fee"] for row in first] == [0.0, 0.01, 0.02]


def test_fee_sweep_matches_simulator_for_same_seed():
    result = run_fee_sensitivity([0.01], simulations=300, seed=23)[0]
    simulator = RetirementSimulator(
        SimulationConfig(simulations=300, annual_fee=0.01, seed=23)
    )

    assert result["market_success_rate_pct"] == simulator.run_market()["success_rate_pct"]
    assert result["floor_cap_success_rate_pct"] == simulator.run_floor_cap()["success_rate_pct"]


def test_success_rates_do_not_increase_when_fees_rise():
    results = run_fee_sensitivity([0.0, 0.01, 0.02], simulations=500, seed=31)

    for key in ("market_success_rate_pct", "floor_cap_success_rate_pct"):
        rates = [row[key] for row in results]
        assert rates == sorted(rates, reverse=True)


@pytest.mark.parametrize(
    "fees",
    [[], [-0.01], [1.01], [float("nan")], [0.01, 0.01]],
)
def test_invalid_fee_grids_are_rejected(fees):
    with pytest.raises(ValueError):
        run_fee_sensitivity(fees, simulations=10)


def test_png_is_written_and_readable(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    results = run_fee_sensitivity([0.0, 0.01, 0.02], simulations=100, seed=42)
    output = tmp_path / "nested" / "fee-chart.png"

    saved = plot_fee_sensitivity(results, output, simulations=100, seed=42)

    assert saved == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.imread(output).shape[0] > 0
