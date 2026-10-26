import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_INDICES_FILE = PROJECT_ROOT / "config" / "default_region_indices.txt"
FALLBACK_DEFAULT_REGION_INDICES_TEXT = "1,2,3,4,5,13,14,15,16,17,18,21,22"


def _coerce_exact_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, found {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, found {value!r}") from error

    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be a finite integer, found {value!r}")

    return int(number)


def _parse_csv_ints(value):
    if value is None:
        return []

    try:
        if math.isnan(value):
            return []
    except TypeError:
        pass

    if isinstance(value, (list, tuple, set)):
        return [_coerce_exact_int(item, "region index") for item in value]

    values = []

    for item in str(value).split(","):
        item = item.strip().strip('"').strip("'")

        if item:
            values.append(_coerce_exact_int(item, "region index"))

    return values


def _coerce_region_id(value):
    if value is None:
        return None

    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass

    return _coerce_exact_int(value, "electrode region ID")


def default_region_indices_text(path=None):
    path = Path(path) if path is not None else DEFAULT_REGION_INDICES_FILE

    if path.exists():
        text = path.read_text(encoding="utf-8").strip()

        if text:
            return ",".join(str(i) for i in _parse_csv_ints(text))

    return FALLBACK_DEFAULT_REGION_INDICES_TEXT


DEFAULT_REGION_INDICES_TEXT = default_region_indices_text()
DEFAULT_REGION_INDICES = _parse_csv_ints(DEFAULT_REGION_INDICES_TEXT)


def parse_region_indices(value):
    indices = _parse_csv_ints(value)

    if indices:
        return indices

    return list(DEFAULT_REGION_INDICES)


def region_indices_to_text(region_indices=None):
    indices = DEFAULT_REGION_INDICES if region_indices is None else region_indices
    return ",".join(str(int(i)) for i in indices)


def load_region_metadata(config_dir="./config"):
    config_dir = Path(config_dir)

    with open(config_dir / "sub_info.json", encoding="utf-8") as file:
        sub_info = json.load(file)

    with open(config_dir / "sub_bad_chl.json", encoding="utf-8") as file:
        sub_bad_chl = json.load(file)

    return sub_info, sub_bad_chl


def select_channel_indices(
    elec_region_idx,
    region_indices,
    bad_channel_indices=(),
    include_bad_channels=False,
):
    selected_regions = set(parse_region_indices(region_indices))
    bad_channels = {int(i) for i in bad_channel_indices}

    selected_indices = []

    for i, value in enumerate(elec_region_idx):
        region_id = _coerce_region_id(value)

        if region_id is None or region_id not in selected_regions:
            continue

        if not include_bad_channels and i in bad_channels:
            continue

        selected_indices.append(i)

    return selected_indices


def validate_channel_selection(
    channel_indices,
    n_channels,
    expected_n_channels=None,
    source="channel tensor",
):
    if n_channels <= 0:
        raise ValueError(f"{source}: channel tensor is empty")

    if expected_n_channels is not None and n_channels != expected_n_channels:
        raise ValueError(
            f"{source}: expected {expected_n_channels} channels, found {n_channels}"
        )

    indices = [int(i) for i in channel_indices]

    invalid = sorted(
        {i for i in indices if i < 0 or i >= n_channels}
    )

    if invalid:
        raise ValueError(
            f"{source}: channel indices outside 0..{n_channels - 1}: {invalid}"
        )

    duplicates = sorted(
        {i for i in indices if indices.count(i) > 1}
    )

    if duplicates:
        raise ValueError(
            f"{source}: duplicate channel indices: {duplicates}"
        )

    return indices


def resolve_selected_channels(
    patient_id,
    region_indices=None,
    config_dir="./config",
    include_bad_channels=False,
    allow_empty=False,
):
    sub_info, sub_bad_chl = load_region_metadata(config_dir)

    if patient_id not in sub_info:
        raise KeyError(
            f"{patient_id} not found in "
            f"{Path(config_dir) / 'sub_info.json'}"
        )

    elec_region_idx = sub_info[patient_id]["elec_region_idx"]
    selected_regions = parse_region_indices(region_indices)

    selected_indices = select_channel_indices(
        elec_region_idx,
        selected_regions,
        sub_bad_chl.get(patient_id, []),
        include_bad_channels=include_bad_channels,
    )

    if not selected_indices and not allow_empty:
        raise RuntimeError(
            f"No usable channels found for {patient_id}; "
            f"region_indices={sorted(set(selected_regions))}"
        )

    used_regions = sorted(
        {
            _coerce_region_id(elec_region_idx[i])
            for i in selected_indices
        }
    )

    return selected_indices, used_regions, elec_region_idx