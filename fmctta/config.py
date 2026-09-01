import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    num_classes: int
    num_domains: int
    samples_per_class: int
    default_sequence: Tuple[int, ...]
    description: str
    repair_labels: bool = False
    num_channels: int = 1
    default_stream_order: str = "shuffled"
    domain_names: Tuple[str, ...] = ()
    domain_groups: Tuple[str, ...] = ()
    samples_per_class_by_domain: Tuple[int, ...] = ()

    @property
    def samples_per_domain(self) -> int:
        return self.num_classes * self.samples_per_class

    @property
    def expected_samples(self) -> int:
        return sum(
            self.samples_for_domain(domain) for domain in range(self.num_domains)
        )

    def samples_per_class_for_domain(self, domain: int) -> int:
        if self.samples_per_class_by_domain:
            return self.samples_per_class_by_domain[domain]
        return self.samples_per_class

    def samples_for_domain(self, domain: int) -> int:
        return self.num_classes * self.samples_per_class_for_domain(domain)

    def domain_offset(self, domain: int) -> int:
        return sum(self.samples_for_domain(item) for item in range(domain))

    def path(self, data_root: Path) -> Path:
        return data_root / self.filename

    def domain_name(self, domain: int) -> str:
        if self.domain_names:
            return self.domain_names[domain]
        return str(domain)


DATASETS: Dict[str, DatasetSpec] = {
    "PU": DatasetSpec(
        "PU",
        "PU_0p1s_custom.mat",
        14,
        4,
        200,
        (0, 1, 2, 3),
        "Custom native-rate 0.1-second PU CTTA dataset",
        default_stream_order="interleaved",
        domain_names=(
            "N15_M07_F10",
            "N09_M07_F10",
            "N15_M01_F10",
            "N15_M07_F04",
        ),
    ),
    "HUST": DatasetSpec(
        "HUST",
        "HUST_Bearings_10domains_9classes.mat",
        9,
        10,
        100,
        (5, 6, 7, 8),
        "HUST bearing dataset",
    ),
    "HUST_gear": DatasetSpec(
        "HUST_gear",
        "HUST_Gearbox_25domains_3classes.mat",
        3,
        25,
        100,
        (0, 6, 12, 18, 24),
        "HUST gearbox dataset",
    ),
    "MCC5_THU_Gearbox": DatasetSpec(
        name="MCC5_THU_Gearbox",
        filename="MCC5_THU_Gearbox_online_20classes_12domains.mat",
        num_classes=20,
        num_domains=12,
        samples_per_class=100,
        default_sequence=tuple(range(12)),
        description=(
            "MCC5-THU gearbox: one steady labeled source followed by "
            "eleven time-varying speed/load target streams"
        ),
        num_channels=3,
        default_stream_order="interleaved",
        domain_names=(
            "source_steady_1000rpm_10Nm",
            "torque_circulation_2000rpm_10Nm",
            "torque_circulation_3000rpm_10Nm",
            "torque_circulation_3000rpm_20Nm",
            "torque_circulation_2000rpm_20Nm",
            "torque_circulation_1000rpm_20Nm",
            "speed_circulation_20Nm-1000rpm",
            "speed_circulation_20Nm-2000rpm",
            "speed_circulation_20Nm-3000rpm",
            "speed_circulation_10Nm-3000rpm",
            "speed_circulation_10Nm-2000rpm",
            "speed_circulation_10Nm-1000rpm",
        ),
    ),
    "MCC5_THU_Motor": DatasetSpec(
        name="MCC5_THU_Motor",
        filename="MCC5_THU_Motor_online_24classes_11domains.mat",
        num_classes=24,
        num_domains=11,
        samples_per_class=100,
        default_sequence=tuple(range(11)),
        description=(
            "MCC5-THU motor: one steady labeled source followed by "
            "ten time-varying speed/load target streams"
        ),
        num_channels=3,
        default_stream_order="interleaved",
        domain_names=(
            "source_steady_1000rpm_20Nm",
            "torque_circulation_2000rpm_20Nm",
            "torque_circulation_3000rpm_20Nm",
            "torque_circulation_3000rpm_40Nm",
            "torque_circulation_2000rpm_40Nm",
            "torque_circulation_1000rpm_40Nm",
            "speed_circulation_40Nm_1000rpm",
            "speed_circulation_40Nm_2000rpm",
            "speed_circulation_40Nm_3000rpm",
            "speed_circulation_20Nm_2000rpm",
            "speed_circulation_20Nm_1000rpm",
        ),
    ),
    "WT": DatasetSpec(
        "WT",
        "WT_0p1s_custom.mat",
        5,
        8,
        200,
        tuple(range(8)),
        "Custom native-rate 0.1-second WT CTTA dataset",
        num_channels=2,
        default_stream_order="interleaved",
        domain_names=("20", "25", "30", "35", "40", "45", "50", "55"),
    ),
}


def parse_domains(value: str) -> Tuple[int, ...]:
    try:
        domains = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid domain list: {value!r}") from exc
    if not domains:
        raise ValueError("At least one domain must be specified")
    return domains


def resolve_sequence(spec: DatasetSpec, value: str = "") -> Tuple[int, ...]:
    sequence = parse_domains(value) if value else spec.default_sequence
    invalid = [domain for domain in sequence if not 0 <= domain < spec.num_domains]
    if invalid:
        raise ValueError(
            f"{spec.name} has domains 0..{spec.num_domains - 1}; invalid: {invalid}"
        )
    if len(sequence) < 2:
        raise ValueError(
            "CTTA requires one source domain and at least one target domain"
        )
    return sequence


def resolve_target_domains(spec: DatasetSpec, value: str) -> Tuple[int, ...]:
    """Resolve an explicit online target stream for multi-source training."""

    domains = parse_domains(value)
    invalid = [domain for domain in domains if not 0 <= domain < spec.num_domains]
    if invalid:
        raise ValueError(
            f"{spec.name} has domains 0..{spec.num_domains - 1}; invalid: {invalid}"
        )
    return domains


def _parse_positive_counts(value: str, context: str) -> Tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid {context}: {value!r}") from exc
    if not counts or any(count < 1 for count in counts):
        raise ValueError(f"{context} must contain positive integers: {value!r}")
    return counts


def parse_source_samples_per_class(
    value: str, num_classes: int
) -> Optional[Tuple[int, ...]]:
    """Parse one count for all source classes or one count per class."""
    if not value.strip():
        return None
    counts = _parse_positive_counts(value, "source-samples-per-class")
    if len(counts) == 1:
        return counts * num_classes
    if len(counts) != num_classes:
        raise ValueError(
            "source-samples-per-class must contain one count or exactly "
            f"{num_classes} counts"
        )
    return counts


def parse_target_samples_per_class(
    value: str,
    num_classes: int,
    target_count: int,
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Parse per-target-occurrence class counts from the CLI representation."""
    if not value.strip():
        return None
    entries = tuple(item.strip() for item in value.split(";"))
    if any(not item for item in entries):
        raise ValueError("target-samples-per-class cannot contain empty entries")
    if len(entries) == 1:
        entries = entries * target_count
    elif len(entries) != target_count:
        raise ValueError(
            "target-samples-per-class must contain one entry or exactly "
            f"{target_count} semicolon-separated target entries"
        )

    parsed = []
    for position, entry in enumerate(entries, start=1):
        counts = _parse_positive_counts(
            entry, f"target-samples-per-class entry {position}"
        )
        if len(counts) == 1:
            counts = counts * num_classes
        elif len(counts) != num_classes:
            raise ValueError(
                "Each target-samples-per-class entry must contain one count or "
                f"exactly {num_classes} counts"
            )
        parsed.append(counts)
    return tuple(parsed)


def dataset_spec_from_manifest(path: Path) -> DatasetSpec:
    """Load a generated custom dataset without editing the static registry."""

    path = Path(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    values = content.get("fmctta_spec", content.get("octta_spec"))
    if not isinstance(values, dict):
        raise ValueError(
            f"{path} does not contain an 'fmctta_spec' or 'octta_spec' object"
        )
    required = {
        "name",
        "filename",
        "num_classes",
        "num_domains",
        "samples_per_class",
        "default_sequence",
        "description",
        "num_channels",
        "default_stream_order",
        "domain_names",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"{path} is missing fmctta_spec fields: {missing}")
    spec = DatasetSpec(
        name=str(values["name"]),
        filename=str(values["filename"]),
        num_classes=int(values["num_classes"]),
        num_domains=int(values["num_domains"]),
        samples_per_class=int(values["samples_per_class"]),
        default_sequence=tuple(int(item) for item in values["default_sequence"]),
        description=str(values["description"]),
        num_channels=int(values["num_channels"]),
        default_stream_order=str(values["default_stream_order"]),
        domain_names=tuple(str(item) for item in values["domain_names"]),
        domain_groups=tuple(str(item) for item in values.get("domain_groups", ())),
        samples_per_class_by_domain=tuple(
            int(item) for item in values.get("samples_per_class_by_domain", ())
        ),
    )
    if len(spec.domain_names) != spec.num_domains:
        raise ValueError(
            f"{path}: num_domains={spec.num_domains}, but domain_names has "
            f"{len(spec.domain_names)} entries"
        )
    if spec.domain_groups and len(spec.domain_groups) != spec.num_domains:
        raise ValueError(
            f"{path}: num_domains={spec.num_domains}, but domain_groups has "
            f"{len(spec.domain_groups)} entries"
        )
    if (
        spec.samples_per_class_by_domain
        and len(spec.samples_per_class_by_domain) != spec.num_domains
    ):
        raise ValueError(
            f"{path}: samples_per_class_by_domain must contain one count per domain"
        )
    if any(count < 1 for count in spec.samples_per_class_by_domain):
        raise ValueError(f"{path}: samples_per_class_by_domain must be positive")
    resolve_sequence(spec)
    return spec
