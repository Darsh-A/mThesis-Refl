"""Command-line entry points for the potluck generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ModelConfig, load_config, save_config
from .dataset import generate_dataset
from .diagnostics import (
    ceiling_report,
    diagnose,
    format_ledger,
    format_report,
    noise_sweep,
    plot_all,
    plot_reliability,
    plot_stratified,
    plot_transfer,
)
from .diagnostics.plots import plot_ceiling, plot_noise_sweep
from .io import load_dataset, save_dataset
from .validation import (
    eq13_check,
    run_ablation,
    transfer_report,
    validate_dataset,
)


def _config(path: str | None) -> ModelConfig:
    return load_config(path) if path else ModelConfig()


def _generate(args: argparse.Namespace) -> None:
    config = _config(args.config)
    arrays, meta = generate_dataset(
        config,
        n_samples=args.n,
        seed=args.seed,
        verbose=args.verbose,
    )
    output = Path(args.output or Path(config.runtime.output_dir) / "dataset.npz")
    save_dataset(output, arrays, meta)
    print(f"saved {meta['n_accepted']} samples to {output}")
    print(format_ledger(meta))


def _diagnose(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = diagnose(arrays, meta)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report))


def _validate(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    result = validate_dataset(arrays, meta)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


def _eq13(args: argparse.Namespace) -> None:
    config = _config(args.config)
    result = eq13_check(
        config,
        pisn_mass=args.mass,
        f_pisn=args.f_pisn,
        f_ratio=args.f_ratio,
        tpop2_myr=args.tpop2,
    )
    print(json.dumps(result, indent=2, default=str))


def _ablation(args: argparse.Namespace) -> None:
    config = _config(args.config)
    summary = run_ablation(
        config,
        n_samples=args.n,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, default=str))


def _plots(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = diagnose(arrays, meta)
    default_dir = Path(meta["config"]["runtime"]["output_dir"]) / "plots"
    outdir = Path(args.outdir) if args.outdir else default_dir
    paths = plot_all(arrays, meta, outdir)
    print(format_report(report))
    print(f"wrote {len(paths)} figures to {outdir}")


def _ceiling(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = ceiling_report(arrays, meta, cv=args.cv, seed=args.seed)
    summary = {
        key: report[key]
        for key in (
            "n",
            "n_splits",
            "y_rate",
            "auc",
            "nuisance_auc",
            "f_fep_r2",
            "f_fep_mae",
            "f_zp_r2",
            "f_zp_mae",
        )
        if key in report
    }
    print(json.dumps(summary, indent=2))
    if args.plot:
        plot_ceiling(report, args.plot)
        print(f"wrote {args.plot}")


def _stratified(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = ceiling_report(arrays, meta, cv=args.cv, seed=args.seed)
    summary = {
        key: report[key]
        for key in ("by_fzp", "by_pisn_mass", "by_feh")
        if key in report
    }
    print(json.dumps(summary, indent=2))
    if args.plot:
        plot_stratified(report, args.plot)
        print(f"wrote {args.plot}")


def _noise(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = noise_sweep(arrays, meta, cv=args.cv, seed=args.seed)
    print(json.dumps(report, indent=2))
    if args.plot:
        plot_noise_sweep(report, args.plot)
        print(f"wrote {args.plot}")


def _reliability(args: argparse.Namespace) -> None:
    arrays, meta = load_dataset(args.dataset)
    report = ceiling_report(arrays, meta, cv=args.cv, seed=args.seed)
    summary = {
        "chemistry_brier": report["reliability"]["brier"],
        "nuisance_brier": report["nuisance_reliability"]["brier"],
        "chemistry_auc": report["auc"],
        "nuisance_auc": report["nuisance_auc"],
    }
    print(json.dumps(summary, indent=2))
    if args.plot:
        plot_reliability(report, args.plot)
        print(f"wrote {args.plot}")


def _transfer(args: argparse.Namespace) -> None:
    base, _ = load_dataset(args.base)
    variant_dir = Path(args.variant_dir)
    variants = {}
    for path in sorted(variant_dir.glob("*.npz")):
        if path.resolve() == Path(args.base).resolve():
            continue
        arrays, _ = load_dataset(path)
        variants[path.stem] = arrays
    report = transfer_report(base, variants, seed=args.seed)
    print(json.dumps(report, indent=2))
    if args.plot:
        plot_transfer(report, args.plot)
        print(f"wrote {args.plot}")


def _write_config(args: argparse.Namespace) -> None:
    config = _config(args.config)
    save_config(args.output, config)
    print(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="potluck")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="draw and store a dataset")
    generate.add_argument("--config")
    generate.add_argument("--n", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--output")
    generate.add_argument("--verbose", action="store_true")
    generate.set_defaults(func=_generate)

    diagnose_parser = sub.add_parser("diagnose", help="summarise a stored dataset")
    diagnose_parser.add_argument("dataset")
    diagnose_parser.add_argument("--json", action="store_true")
    diagnose_parser.set_defaults(func=_diagnose)

    validate_parser = sub.add_parser("validate", help="run internal consistency checks")
    validate_parser.add_argument("dataset")
    validate_parser.set_defaults(func=_validate)

    eq13_parser = sub.add_parser("eq13", help="compare against Salvadori Eq. 13")
    eq13_parser.add_argument("--config")
    eq13_parser.add_argument("--mass", type=float, default=200.0)
    eq13_parser.add_argument("--f-pisn", type=float, default=0.5)
    eq13_parser.add_argument("--f-ratio", type=float, default=1.0e-3)
    eq13_parser.add_argument("--tpop2", type=float, default=10.0)
    eq13_parser.set_defaults(func=_eq13)

    ablation_parser = sub.add_parser("ablation", help="run the prior ablation ladder")
    ablation_parser.add_argument("--config")
    ablation_parser.add_argument("--n", type=int, default=1000)
    ablation_parser.add_argument("--output-dir")
    ablation_parser.add_argument("--verbose", action="store_true")
    ablation_parser.set_defaults(func=_ablation)

    plots_parser = sub.add_parser("plots", help="write diagnostic figures")
    plots_parser.add_argument("dataset")
    plots_parser.add_argument("--outdir")
    plots_parser.set_defaults(func=_plots)

    ceiling_parser = sub.add_parser(
        "ceiling", help="cross-validated noiseless chemistry ceiling"
    )
    ceiling_parser.add_argument("dataset")
    ceiling_parser.add_argument("--cv", type=int, default=5)
    ceiling_parser.add_argument("--seed", type=int, default=0)
    ceiling_parser.add_argument("--plot")
    ceiling_parser.set_defaults(func=_ceiling)

    stratified_parser = sub.add_parser(
        "stratified", help="AUC and R2 by fZP, PISN mass, and [Fe/H]"
    )
    stratified_parser.add_argument("dataset")
    stratified_parser.add_argument("--cv", type=int, default=5)
    stratified_parser.add_argument("--seed", type=int, default=0)
    stratified_parser.add_argument("--plot")
    stratified_parser.set_defaults(func=_stratified)

    noise_parser = sub.add_parser(
        "noise", help="chemistry ceiling against element-level noise"
    )
    noise_parser.add_argument("dataset")
    noise_parser.add_argument("--cv", type=int, default=5)
    noise_parser.add_argument("--seed", type=int, default=0)
    noise_parser.add_argument("--plot")
    noise_parser.set_defaults(func=_noise)

    reliability_parser = sub.add_parser(
        "reliability", help="calibration curve and Brier score"
    )
    reliability_parser.add_argument("dataset")
    reliability_parser.add_argument("--cv", type=int, default=5)
    reliability_parser.add_argument("--seed", type=int, default=0)
    reliability_parser.add_argument("--plot")
    reliability_parser.set_defaults(func=_reliability)

    transfer_parser = sub.add_parser(
        "transfer", help="train on base prior, test on ablation priors"
    )
    transfer_parser.add_argument("base")
    transfer_parser.add_argument("--variant-dir", required=True)
    transfer_parser.add_argument("--seed", type=int, default=0)
    transfer_parser.add_argument("--plot")
    transfer_parser.set_defaults(func=_transfer)

    config_parser = sub.add_parser("config", help="write a default configuration")
    config_parser.add_argument("output")
    config_parser.add_argument("--config")
    config_parser.set_defaults(func=_write_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the requested command."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
