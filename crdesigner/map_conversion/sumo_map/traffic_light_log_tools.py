import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CLASSIFICATION_RECORD_MARKER = "Traffic-light classification record:"
RETAINED_CATEGORIES = frozenset({"normal", "remapped"})
SKIPPED_CATEGORIES = frozenset(
    {"skipped_ambiguous", "skipped_removed_no_unique_upstream"}
)


def _normalize_logged_identifier(value: Any) -> int | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _sort_identifier_key(value: int | str):
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def load_traffic_light_classification_records_from_log(log_file: str | Path) -> list[dict[str, Any]]:
    log_path = Path(log_file)
    records: list[dict[str, Any]] = []

    with log_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            marker_index = line.find(CLASSIFICATION_RECORD_MARKER)
            if marker_index < 0:
                continue

            payload = line[marker_index + len(CLASSIFICATION_RECORD_MARKER) :].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse traffic-light classification record in {log_path}:{line_number}."
                ) from exc

            records.append(record)

    if not records:
        raise ValueError(
            "The classification log does not contain detailed 'Traffic-light classification record' "
            "entries. Re-run CR->SUMO conversion with the detailed traffic-light classification logging "
            "enabled and use that log file."
        )

    return records


def summarize_traffic_light_classification_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    categories_by_traffic_light_id: dict[int | str, set[str]] = defaultdict(set)
    lanelet_ids_by_traffic_light_id: dict[int | str, dict[str, set[int | str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    record_count = 0

    for record in records:
        record_count += 1
        category = str(record.get("category"))
        lanelet_id = _normalize_logged_identifier(record.get("lanelet_id"))
        for traffic_light_id in record.get("traffic_light_ids", []):
            normalized_traffic_light_id = _normalize_logged_identifier(traffic_light_id)
            categories_by_traffic_light_id[normalized_traffic_light_id].add(category)
            lanelet_ids_by_traffic_light_id[normalized_traffic_light_id][category].add(lanelet_id)

    disappeared_only_ids = []
    partially_lost_ids = []
    retained_only_ids = []

    for traffic_light_id, categories in categories_by_traffic_light_id.items():
        has_retained = bool(categories & RETAINED_CATEGORIES)
        has_skipped = bool(categories & SKIPPED_CATEGORIES)
        if has_skipped and not has_retained:
            disappeared_only_ids.append(traffic_light_id)
        elif has_skipped and has_retained:
            partially_lost_ids.append(traffic_light_id)
        elif has_retained:
            retained_only_ids.append(traffic_light_id)

    disappeared_only_ids.sort(key=_sort_identifier_key)
    partially_lost_ids.sort(key=_sort_identifier_key)
    retained_only_ids.sort(key=_sort_identifier_key)

    categories_by_traffic_light_id_json = {
        str(traffic_light_id): sorted(categories)
        for traffic_light_id, categories in sorted(
            categories_by_traffic_light_id.items(), key=lambda item: _sort_identifier_key(item[0])
        )
    }
    lanelet_ids_by_traffic_light_id_json = {
        str(traffic_light_id): {
            category: sorted(lanelet_ids, key=_sort_identifier_key)
            for category, lanelet_ids in sorted(category_map.items())
        }
        for traffic_light_id, category_map in sorted(
            lanelet_ids_by_traffic_light_id.items(), key=lambda item: _sort_identifier_key(item[0])
        )
    }

    return {
        "record_count": record_count,
        "traffic_light_counts": {
            "disappeared_only": len(disappeared_only_ids),
            "partially_lost": len(partially_lost_ids),
            "retained_only": len(retained_only_ids),
            "total_classified": len(categories_by_traffic_light_id),
        },
        "traffic_light_ids": {
            "disappeared_only": disappeared_only_ids,
            "partially_lost": partially_lost_ids,
            "retained_only": retained_only_ids,
        },
        "categories_by_traffic_light_id": categories_by_traffic_light_id_json,
        "lanelet_ids_by_traffic_light_id": lanelet_ids_by_traffic_light_id_json,
    }

