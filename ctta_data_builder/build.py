#!/usr/bin/env python3
"""Command-line entry point for configurable CTTA dataset construction."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .core import build_dataset, load_build_config, validate_config, validate_raw_layout
from .datasets import DATASETS, WINDOW_SECONDS


def _comma_list(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("An override list cannot be empty")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build native-rate, non-overlapping 0.1-second MAT datasets for CTTA"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="JSON build configuration")
    parser.add_argument(
        "--catalog",
        choices=tuple(DATASETS),
        help="Print the available classes/domains for one dataset and exit",
    )
    parser.add_argument("--raw-root", type=Path, help="Override config raw_root")
    parser.add_argument("--output", type=Path, help="Override config output")
    parser.add_argument("--name", help="Override the generated FMCTTA dataset name")
    parser.add_argument(
        "--classes", help="Override class order with a comma-separated list"
    )
    parser.add_argument(
        "--domains",
        help=(
            "Override the domain-label mapping with raw domain names for labels "
            "0,1,... in a comma-separated list"
        ),
    )
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection", choices=("head", "uniform"))
    parser.add_argument(
        "--target-stream-order",
        choices=("class_block", "interleaved", "shuffled"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and required raw paths without reading signal values",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MAT file and manifest",
    )
    return parser


def _catalog(dataset: str) -> dict:
    adapter = DATASETS[dataset]
    return {
        "dataset": adapter.name,
        "sampling_frequency_hz": adapter.sampling_frequency,
        "window_seconds": WINDOW_SECONDS,
        "window_size_points": adapter.window_size,
        "channels": list(adapter.channel_names),
        "classes": list(adapter.class_names),
        "domains": list(adapter.domain_names),
    }


def _apply_overrides(config, args):
    changes = {}
    if args.raw_root is not None:
        changes["raw_root"] = args.raw_root.resolve()
    if args.output is not None:
        changes["output"] = args.output.resolve()
    if args.name is not None:
        changes["name"] = args.name
    classes = _comma_list(args.classes)
    domains = _comma_list(args.domains)
    if classes is not None:
        changes["classes"] = classes
    if domains is not None:
        changes["domains"] = domains
    if args.samples_per_class is not None:
        changes["samples_per_class"] = args.samples_per_class
    if args.selection is not None:
        changes["selection"] = args.selection
    if args.target_stream_order is not None:
        changes["target_stream_order"] = args.target_stream_order
    result = replace(config, **changes)
    validate_config(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps(_catalog(args.catalog), indent=2, ensure_ascii=False))
        return
    if args.config is None:
        parser.error("--config is required unless --catalog is used")

    config = _apply_overrides(load_build_config(args.config), args)
    if args.validate_only:
        paths = validate_raw_layout(config)
        details = {
            "status": "ok",
            "dataset": config.dataset,
            "name": config.dataset_name,
            "domain_labels": {
                str(label): name for label, name in enumerate(config.domains)
            },
            "classes": list(config.classes),
            "raw_recording_counts": paths,
            "note": (
                "Signal values were not read; available window counts are checked "
                "during the build."
            ),
        }
        print(json.dumps(details, indent=2, ensure_ascii=False))
        return

    result = build_dataset(config, overwrite=args.overwrite)
    print(f"Saved {result.data_shape} float32 signals to {result.mat_path}")
    print(f"Saved CTTA/FMCTTA manifest to {result.manifest_path}")


if __name__ == "__main__":
    main()
