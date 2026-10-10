"""Prepare the inputs, run one RGDR pass, and save its result."""

import argparse
from pathlib import Path
import sys

# Direct script execution makes the Code directory available to imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from RGDR import run_rgdr
from utils import load_integration, load_schedule


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration", default=config.DIR["integration"],
        help="Importable module:factory returning base, task, and VLM; base includes the flow model.",
    )
    parser.add_argument("--eta", type=float, default=None, help="One relative correction strength.")
    parser.add_argument("--epsilon", type=float, default=None, help="Direction normalization stabilizer.")
    parser.add_argument(
        "--schedule", default=None,
        help="Optional importable module:function implementing s_t(t, T).",
    )
    parser.add_argument(
        "--output", type=Path, default=config.DIR["output"],
        help="Output directory passed to the integration's save_result callback.",
    )
    args = parser.parse_args(argv)
    if args.output is not None:
        args.output = Path(args.output)
    if not args.integration:
        parser.error("Set DIR['integration'] in config.py or supply --integration module:factory.")
    materials = load_integration(args.integration)
    save_result = materials.get("save_result")
    if args.output is not None and not callable(save_result):
        raise ValueError("--output requires the integration's save_result callback.")
    schedule = load_schedule(args.schedule) if args.schedule is not None else None

    result = run_rgdr(
        base=materials["base"],
        task=materials["task"],
        vlm=materials["vlm"],
        eta=args.eta,
        s_t=schedule,
        epsilon=args.epsilon,
    )
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        save_result(result, args.output)
    print("RGDR pass completed: eta=" + str(result.settings.eta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
