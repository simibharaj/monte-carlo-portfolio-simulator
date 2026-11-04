"""Plot success rates across annual fees for the teaching simulator."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

from simulator import RetirementSimulator, SimulationConfig

DEFAULT_FEES = (0.0, 0.01, 0.02)


def run_fee_sensitivity(
    fees: Iterable[float] = DEFAULT_FEES,
    *,
    simulations: int = 10_000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Return market and floor-cap success rates for a reproducible fee sweep."""
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")

    try:
        fee_values = sorted(float(fee) for fee in fees)
    except (TypeError, ValueError) as error:
        raise ValueError("fees must be finite decimal rates between 0 and 1") from error

    if not fee_values:
        raise ValueError("provide at least one fee")
    if any(not math.isfinite(fee) or fee < 0 or fee > 1 for fee in fee_values):
        raise ValueError("fees must be finite decimal rates between 0 and 1")
    if len(fee_values) != len(set(fee_values)):
        raise ValueError("fees must not contain duplicates")

    results: list[dict[str, float]] = []
    for fee in fee_values:
        simulator = RetirementSimulator(
            SimulationConfig(simulations=simulations, annual_fee=fee, seed=seed)
        )
        results.append(
            {
                "annual_fee": fee,
                "market_success_rate_pct": simulator.run_market()["success_rate_pct"],
                "floor_cap_success_rate_pct": simulator.run_floor_cap()["success_rate_pct"],
            }
        )
    return results


def _format_fee(fee: float) -> str:
    return f"{fee * 100:.2f}".rstrip("0").rstrip(".") + "%"


def plot_fee_sensitivity(
    results: list[dict[str, float]],
    output_path: str | Path,
    *,
    simulations: int,
    seed: int,
) -> Path:
    """Render a PNG using an optional plotting dependency."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Install the optional chart dependency with "
            "`python -m pip install -r requirements-plot.txt`."
        ) from error

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    paper = "#F1EFE7"
    ink = "#26332F"
    forest = "#286449"
    slate = "#5D7075"
    grid = "#C8C9C2"

    fee_percent = [row["annual_fee"] * 100 for row in results]
    market_rates = [row["market_success_rate_pct"] for row in results]
    floor_cap_rates = [row["floor_cap_success_rate_pct"] for row in results]

    figure, axis = plt.subplots(figsize=(8.4, 5.2), facecolor=paper)
    try:
        axis.set_facecolor(paper)
        axis.plot(
            fee_percent,
            market_rates,
            color=slate,
            marker="s",
            linewidth=2.2,
            markersize=6,
            label="Full market exposure",
        )
        axis.plot(
            fee_percent,
            floor_cap_rates,
            color=forest,
            marker="o",
            linewidth=2.6,
            markersize=6,
            label="Floor 0% / cap 9.5%",
        )
        axis.set_title(
            "Annual fees change modeled survival",
            loc="left",
            color=ink,
            fontfamily="DejaVu Serif",
            fontsize=16,
            fontweight="bold",
            pad=16,
        )
        axis.text(
            0,
            1.02,
            f"30-year horizon, \\$500k start, \\$25k annual withdrawal, {simulations:,} paths per fee. "
            f"Seed {seed} aligns the market paths.",
            transform=axis.transAxes,
            color=slate,
            fontfamily="DejaVu Sans",
            fontsize=9,
            va="bottom",
        )
        axis.set_xlabel("Annual fee", color=ink, fontfamily="DejaVu Sans", labelpad=10)
        axis.set_ylabel("Success rate (%)", color=ink, fontfamily="DejaVu Sans", labelpad=10)
        axis.set_xticks(fee_percent, [_format_fee(row["annual_fee"]) for row in results])
        axis.set_ylim(0, 100)
        axis.set_yticks(range(0, 101, 20))
        axis.tick_params(colors=ink, labelsize=9)
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(grid)
        axis.spines["bottom"].set_color(grid)
        axis.legend(
            frameon=False,
            loc="upper right",
            prop={"family": "DejaVu Sans", "size": 9},
        )
        figure.text(
            0.08,
            0.025,
            "Illustrative teaching model. Not a forecast, product illustration, or advice.",
            color=slate,
            fontfamily="DejaVu Sans",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        figure.savefig(output, format="png", dpi=180, bbox_inches="tight", facecolor=paper)
    finally:
        plt.close(figure)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fee_sensitivity.png"),
        help="where to save the PNG (default: fee_sensitivity.png)",
    )
    parser.add_argument(
        "--fees",
        type=float,
        nargs="+",
        default=DEFAULT_FEES,
        help="annual fee rates as decimals (default: 0 0.01 0.02)",
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        results = run_fee_sensitivity(args.fees, simulations=args.simulations, seed=args.seed)
        output = plot_fee_sensitivity(
            results,
            args.output,
            simulations=args.simulations,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    for row in results:
        print(
            f"{_format_fee(row['annual_fee'])}: market "
            f"{row['market_success_rate_pct']:.2f}%, floor-cap "
            f"{row['floor_cap_success_rate_pct']:.2f}%"
        )
    print(f"Saved chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
