import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


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

    @property
    def samples_per_domain(self) -> int:
        return self.num_classes * self.samples_per_class

    @property
    def expected_samples(self) -> int:
        return self.num_domains * self.samples_per_domain

    def path(self, data_root: Path) -> Path:
        return data_root / self.filename

    def domain_name(self, domain: int) -> str:
        if self.domain_names:
            return self.domain_names[domain]
        return str(domain)


DATASETS: Dict[str, DatasetSpec] = {
    "SK": DatasetSpec(
        "SK",
        "SK_all_10classes.mat",
        10,
        24,
        100,
        (6, 1, 8, 15, 22, 17),
        "Southeast University bearing dataset",
    ),
    "SK_new": DatasetSpec(
        "SK_new",
        "SK_new_all_10classes.mat",
        10,
        24,
        100,
        (6, 1, 8, 15, 22, 17),
        "Alternative preprocessing of the SK bearing dataset",
    ),
    "HIT": DatasetSpec(
        "HIT",
        "HIT_23doamins_3classes.mat",
        3,
        23,
        100,
        (0, 4, 8, 12, 16, 20, 22),
        "HIT bearing dataset",
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
    "iFlytek": DatasetSpec(
        "iFlytek",
        "iFlytek_all_5classes.mat",
        5,
        16,
        100,
        (2, 3, 4, 5, 7),
        "iFlytek acoustic fault dataset",
    ),
    "WT": DatasetSpec(
        "WT",
        "WT_all_5classes.mat",
        5,
        8,
        100,
        (0, 1, 2, 3, 4),
        "Wind-turbine fault dataset",
    ),
    "PU_Real": DatasetSpec(
        "PU_Real",
        "PU_Real_4doamins_5classes.mat",
        5,
        4,
        100,
        (3, 2, 0, 1),
        "Paderborn real bearing faults",
    ),
    "PU_Real_200": DatasetSpec(
        "PU_Real_200",
        "PU_Real_4doamins_5classes_200.mat",
        5,
        4,
        200,
        (3, 2, 0, 1),
        "Paderborn real bearing faults, 200 samples per class",
    ),
    "PU_Art": DatasetSpec(
        "PU_Art",
        "PU_Art_4doamins_8classes.mat",
        8,
        4,
        100,
        (2, 0, 1, 3),
        "Paderborn artificial bearing faults",
    ),
    "Robot": DatasetSpec(
        "Robot",
        "Robot_4classes_4domains.mat",
        4,
        4,
        20,
        (0, 1, 2, 3),
        "Robot fault dataset (labels in the MAT file require reconstruction)",
        repair_labels=True,
    ),
}


def parse_domains(value: str) -> Tuple[int, ...]:
    try:
        domains = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid domain list: {value!r}") from exc
    if not domains:
        raise ValueError("At least one domain must be specified")
    if len(set(domains)) != len(domains):
        raise ValueError(
            f"Domain sequence contains duplicates: {domains}"
        )
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


def dataset_spec_from_manifest(path: Path) -> DatasetSpec:
    """Load a generated custom dataset without editing the static registry."""

    path = Path(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    values = content.get("fmctta_spec")
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not contain an 'fmctta_spec' object")
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
    )
    if len(spec.domain_names) != spec.num_domains:
        raise ValueError(
            f"{path}: num_domains={spec.num_domains}, but domain_names has "
            f"{len(spec.domain_names)} entries"
        )
    resolve_sequence(spec)
    return spec
