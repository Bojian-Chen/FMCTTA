#!/usr/bin/env python3
"""Command-line entry point for configurable CTTA dataset construction."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .core import build_dataset, load_build_config, validate_config, validate_raw_layout
from .datasets import DATASETS, DEFAULT_WINDOW_SIZE_POINTS


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
            "Build native-rate, non-overlapping point-length MAT datasets for CTTA"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="JSON build configuration")
    parser.add_argument(
        "--catalog",
        choices=tuple(DATASETS),
        help="Print the available classes/domains for one dataset and exit",
    )
    parser.add_argument("--raw_root", type=Path, help="Override config raw_root")
    parser.add_argument("--output", type=Path, help="Override config output")
    parser.add_argument("--name", help="Override the generated FMCTTA dataset name")
    parser.add_argument(
        "--classes", help="Override class order with a comma-separated list"
    )
    parser.add_argument(
        "--channels",
        help="Select channels by catalog name with a comma-separated list",
    )
    parser.add_argument(
        "--domains",
        help="Override the domain-label mapping with raw domain names for labels 0,1,... in a comma-separated list",
    )
    parser.add_argument("--samples_per_class", type=int)
    parser.add_argument(
        "--all_windows",
        action="store_true",
        help=(
            "Preserve every complete non-overlapping window in each domain; "
            "writes streaming HDF5 and overrides samples_per_class"
        ),
    )
    parser.add_argument(
        "--window_size_points",
        type=int,
        help="Number of native-rate sampling points in each non-overlapping window",
    )
    parser.add_argument("--selection", choices=("head", "uniform"))
    parser.add_argument(
        "--target_stream_order",
        choices=("class_block", "interleaved", "shuffled"),
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate config and required raw paths without reading signal values",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MAT file and manifest",
    )
    return parser


def _normalize_legacy_dashed_options(
    argv: Optional[Sequence[str]], parser: argparse.ArgumentParser
) -> list[str]:
    """Accept old ``--kebab-case`` options while documenting ``--snake_case``."""

    normalized = list(sys.argv[1:] if argv is None else argv)
    known_options = parser._option_string_actions
    for index, item in enumerate(normalized):
        if item == "--":
            break
        if not item.startswith("--"):
            continue
        option, separator, value = item.partition("=")
        canonical_option = "--" + option[2:].replace("-", "_")
        if canonical_option in known_options:
            normalized[index] = canonical_option + (
                separator + value if separator else ""
            )
    return normalized


def _catalog(dataset: str, window_size_points: int) -> dict:
    if window_size_points < 1:
        raise ValueError("window_size_points must be positive")
    adapter = DATASETS[dataset]
    domain_catalog = [
        {
            "domain": domain_id,
            "name": domain.name,
            "role": domain.role,
            "recording_group": domain.recording_group or domain.raw_key,
            "segments_sec": [list(segment) for segment in domain.segments_sec],
        }
        for domain_id, domain in enumerate(adapter.domains)
    ]
    return {
        "dataset": adapter.name,
        "sampling_frequency_hz": adapter.sampling_frequency,
        "window_size_points": window_size_points,
        "window_seconds": window_size_points / adapter.sampling_frequency,
        "channels": list(adapter.channel_names),
        "classes": list(adapter.class_names),
        "domains": list(adapter.domain_names),
        "domain_catalog": domain_catalog,
        "source_candidate_domains": [
            item["domain"]
            for item in domain_catalog
            if item["role"] == "source_candidate"
        ],
        "target_stream_domains": [
            item["domain"]
            for item in domain_catalog
            if item["role"] == "target_stream"
        ],
        "recommended_source_domains": [
            domain_id
            for domain_id, domain in enumerate(adapter.domains)
            if domain.name in adapter.recommended_source_domains
        ],
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
    channels = _comma_list(args.channels)
    domains = _comma_list(args.domains)
    if classes is not None:
        changes["classes"] = classes
    if channels is not None:
        changes["channels"] = channels
    if domains is not None:
        changes["domains"] = domains
    if args.all_windows:
        changes["samples_per_class"] = None
    elif args.samples_per_class is not None:
        changes["samples_per_class"] = args.samples_per_class
    if args.window_size_points is not None:
        changes["window_size_points"] = args.window_size_points
    if args.selection is not None:
        changes["selection"] = args.selection
    if args.target_stream_order is not None:
        changes["target_stream_order"] = args.target_stream_order
    result = replace(config, **changes)
    validate_config(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_legacy_dashed_options(argv, parser))
    if args.catalog:
        points = args.window_size_points or DEFAULT_WINDOW_SIZE_POINTS
        print(json.dumps(_catalog(args.catalog, points), indent=2, ensure_ascii=False))
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
            "window_size_points": config.window_size_points,
            "samples_per_class": (
                "all" if config.samples_per_class is None
                else config.samples_per_class
            ),
            "domain_labels": {
                str(label): name for label, name in enumerate(config.domains)
            },
            "classes": list(config.classes),
            "channels": list(
                config.channels or DATASETS[config.dataset].channel_names
            ),
            "raw_recording_counts": paths,
            "note": "Signal values were not read; available window counts are checked during the build.",
        }
        print(json.dumps(details, indent=2, ensure_ascii=False))
        return

    result = build_dataset(config, overwrite=args.overwrite)
    print(f"Saved {result.data_shape} float32 signals to {result.mat_path}")
    print(f"Saved CTTA/FMCTTA manifest to {result.manifest_path}")


if __name__ == "__main__":
    main()
