from __future__ import annotations

import inspect
import json
import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

_PATCH_FLAG = "_crdesigner_nd_patch_applied"
_PATCHED_ONCE = False
_TL_PATCH_FLAG = "_crdesigner_tl_patch_applied"
_TL_PATCHED_ONCE = False
_LANE_GROUP_PATCH_FLAG = "_crdesigner_lane_group_patch_applied"
_LANE_GROUP_PATCHED_ONCE = False


def _vertex_dimension(vertices: np.ndarray) -> int:
    if isinstance(vertices, np.ndarray) and vertices.ndim == 2 and vertices.shape[1] > 0:
        return int(vertices.shape[1])
    return 2


def _needs_patch(util_module) -> bool:
    """Return True if installed commonroad_sumo still hardcodes 2D reshapes."""
    try:
        source = inspect.getsource(util_module.resample_lanelet) + inspect.getsource(
            util_module.erode_lanelet
        )
    except (OSError, TypeError):
        # Source inspection can fail for optimized/packaged functions.
        return True
    return "reshape([-1, 2])" in source or "reshape([1, 2])" in source


def apply_commonroad_sumo_nd_patch() -> bool:
    """
    Patch commonroad_sumo CR->SUMO utility functions to support lanelet vertex dimensions > 2.

    Returns:
        True if patching was applied during this call, False otherwise.
    """
    global _PATCHED_ONCE

    if _PATCHED_ONCE:
        return False

    try:
        from commonroad_sumo.cr2sumo.map_converter import util as cr2sumo_util
    except ImportError:
        _LOGGER.debug("commonroad_sumo is not available; skipping CR->SUMO ND compatibility patch.")
        return False

    if getattr(cr2sumo_util, _PATCH_FLAG, False):
        _PATCHED_ONCE = True
        return False

    if not _needs_patch(cr2sumo_util):
        setattr(cr2sumo_util, _PATCH_FLAG, True)
        _PATCHED_ONCE = True
        _LOGGER.debug("commonroad_sumo already supports ND lanelet vertices; no patch needed.")
        return False

    original_resample_lanelet = cr2sumo_util.resample_lanelet
    original_erode_lanelet = cr2sumo_util.erode_lanelet

    def resample_lanelet_nd(lanelet, step=3.0):
        """
        Resample lanelet center/left/right vertices while preserving their coordinate dimension.
        """
        polyline = lanelet.center_vertices
        if len(polyline) < 2:
            return np.array(polyline)

        dim = _vertex_dimension(polyline)

        polyline_new_c = [polyline[0]]
        polyline_new_r = [lanelet.right_vertices[0]]
        polyline_new_l = [lanelet.left_vertices[0]]

        current_idx = 0
        current_position = step
        current_distance = np.linalg.norm(polyline[0] - polyline[1])

        while current_idx < len(polyline) - 1:
            if current_position <= current_distance:
                ratio = current_position / current_distance
                polyline_new_c.append(
                    (1 - ratio) * polyline[current_idx] + ratio * polyline[current_idx + 1]
                )
                polyline_new_r.append(
                    (1 - ratio) * lanelet.right_vertices[current_idx]
                    + ratio * lanelet.right_vertices[current_idx + 1]
                )
                polyline_new_l.append(
                    (1 - ratio) * lanelet.left_vertices[current_idx]
                    + ratio * lanelet.left_vertices[current_idx + 1]
                )
                current_position += step
            else:
                current_idx += 1
                if current_idx >= len(polyline) - 1:
                    break
                current_position -= current_distance
                current_distance = np.linalg.norm(polyline[current_idx + 1] - polyline[current_idx])

        polyline_new_c.append(polyline[-1])
        polyline_new_r.append(lanelet.right_vertices[-1])
        polyline_new_l.append(lanelet.left_vertices[-1])

        lanelet._center_vertices = np.array(polyline_new_c).reshape([-1, dim])
        lanelet._right_vertices = np.array(polyline_new_r).reshape([-1, dim])
        lanelet._left_vertices = np.array(polyline_new_l).reshape([-1, dim])
        lanelet._distance = lanelet._compute_polyline_cumsum_dist([lanelet.center_vertices])

    def erode_lanelet_nd(lanelet, radius: float):
        """
        Erode lanelet while preserving coordinate dimensionality (2D/3D).
        """

        def shorten_lanelet(lanelet_obj, radius_value: float):
            resample_lanelet_nd(lanelet_obj)
            dim = _vertex_dimension(lanelet_obj.center_vertices)

            def reshape_vertices(vertices: tuple):
                vertices = list(vertices)
                for i in range(3):
                    vertices[i] = np.asarray(vertices[i]).reshape([1, dim])
                return vertices

            cut_vertices_start = reshape_vertices(lanelet_obj.interpolate_position(radius_value))
            cut_vertices_end = reshape_vertices(
                lanelet_obj.interpolate_position(lanelet_obj.distance[-1] - radius_value)
            )

            lanelet_obj._center_vertices = np.insert(
                lanelet_obj._center_vertices[cut_vertices_start[3] + 1 :, :],
                0,
                cut_vertices_start[0],
                axis=0,
            )
            lanelet_obj._right_vertices = np.insert(
                lanelet_obj._right_vertices[cut_vertices_start[3] + 1 :, :],
                0,
                cut_vertices_start[1],
                axis=0,
            )
            lanelet_obj._left_vertices = np.insert(
                lanelet_obj._left_vertices[cut_vertices_start[3] + 1 :, :],
                0,
                cut_vertices_start[2],
                axis=0,
            )

            lanelet_obj._center_vertices = np.append(
                lanelet_obj._center_vertices[: cut_vertices_end[3] + 1, :],
                cut_vertices_end[0],
                axis=0,
            )
            lanelet_obj._right_vertices = np.append(
                lanelet_obj._right_vertices[: cut_vertices_end[3] + 1, :],
                cut_vertices_end[1],
                axis=0,
            )
            lanelet_obj._left_vertices = np.append(
                lanelet_obj._left_vertices[: cut_vertices_end[3] + 1, :],
                cut_vertices_end[2],
                axis=0,
            )
            lanelet_obj._distance = lanelet_obj._compute_polyline_cumsum_dist(
                [lanelet_obj.center_vertices]
            )

        # Require enough room on both sides before width erosion:
        # left/right boundaries are each moved by `radius`, so at least `2 * radius`
        # lane width is needed to avoid collapsing thin lanelets.
        if np.min(np.linalg.norm(lanelet.left_vertices - lanelet.right_vertices, axis=1)) > (
            2.0 * radius
        ):
            left = lanelet.center_vertices - lanelet.left_vertices
            lanelet._left_vertices += left / np.linalg.norm(left, axis=1)[np.newaxis].T * radius
            right = lanelet.center_vertices - lanelet.right_vertices
            lanelet._right_vertices += right / np.linalg.norm(right, axis=1)[np.newaxis].T * radius

        lanelet._distance = lanelet._compute_polyline_cumsum_dist([lanelet.center_vertices])

        if lanelet.distance[-1] > 2.1 * radius:
            shorten_lanelet(lanelet, radius)

        if lanelet._polygon:
            lanelet._polygon = cr2sumo_util.Polygon(
                np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, axis=0)))
            )
        return lanelet

    cr2sumo_util.resample_lanelet = resample_lanelet_nd
    cr2sumo_util.erode_lanelet = erode_lanelet_nd
    cr2sumo_util._crdesigner_original_resample_lanelet = original_resample_lanelet
    cr2sumo_util._crdesigner_original_erode_lanelet = original_erode_lanelet
    setattr(cr2sumo_util, _PATCH_FLAG, True)

    _PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO ND compatibility patch.")
    return True


def _needs_traffic_light_patch() -> bool:
    """Return True if installed commonroad_sumo still raises on ambiguous traffic-light lanelets."""
    try:
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter

        source = inspect.getsource(CR2SumoMapConverter._create_traffic_lights)
    except (ImportError, OSError, TypeError):
        # Source inspection can fail for optimized/packaged functions.
        return True

    # Upstream implementation raises ValueError inside calc_direction_2_connections.
    return "calc_direction_2_connections" in source and "raise ValueError" in source


def _empty_container_like(values):
    if isinstance(values, list):
        return []
    if isinstance(values, tuple):
        return tuple()
    if isinstance(values, set):
        return set()
    return set()


def _normalize_log_identifier(value):
    if hasattr(value, "traffic_light_id"):
        value = getattr(value, "traffic_light_id")
    elif hasattr(value, "lanelet_id"):
        value = getattr(value, "lanelet_id")
    elif hasattr(value, "id"):
        value = getattr(value, "id")
    elif hasattr(value, "value"):
        value = getattr(value, "value")

    if isinstance(value, np.generic):
        value = value.item()

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _log_identifier_sort_key(value):
    normalized = _normalize_log_identifier(value)
    if isinstance(normalized, int):
        return (0, normalized)
    return (1, str(normalized))


def _normalize_identifier_list(values) -> list[int | str]:
    if not values:
        return []
    return sorted(
        (_normalize_log_identifier(value) for value in values),
        key=_log_identifier_sort_key,
    )


def _build_traffic_light_classification_record(
    category: str,
    lanelet,
    *,
    traffic_light_ids=None,
    edge_id=None,
    replacement_edge_id=None,
    has_intersection_mapping: bool | None = None,
    note: str | None = None,
) -> dict:
    if traffic_light_ids is None:
        traffic_light_ids = getattr(lanelet, "traffic_lights", [])

    record = {
        "category": category,
        "lanelet_id": _normalize_log_identifier(lanelet.lanelet_id),
        "traffic_light_ids": _normalize_identifier_list(traffic_light_ids),
        "successor_ids": _normalize_identifier_list(getattr(lanelet, "successor", [])),
    }

    if edge_id is not None:
        record["edge_id"] = _normalize_log_identifier(edge_id)
    if replacement_edge_id is not None:
        record["replacement_edge_id"] = _normalize_log_identifier(replacement_edge_id)
    if has_intersection_mapping is not None:
        record["has_intersection_mapping"] = bool(has_intersection_mapping)
    if note:
        record["note"] = note

    return record


def _summarize_traffic_light_classification_records(records_by_category: dict[str, list[dict]]) -> dict:
    summary = {}
    for category, records in records_by_category.items():
        unique_traffic_light_ids = sorted(
            {
                traffic_light_id
                for record in records
                for traffic_light_id in record.get("traffic_light_ids", [])
            },
            key=_log_identifier_sort_key,
        )
        summary[category] = {
            "lanelet_count": len(records),
            "unique_traffic_light_count": len(unique_traffic_light_ids),
        }
    return summary


def _log_traffic_light_classification_records(records_by_category: dict[str, list[dict]]):
    if not any(records_by_category.values()):
        return

    _LOGGER.info(
        "Traffic-light classification is lanelet-based; one traffic-light ID may appear in multiple categories."
    )
    _LOGGER.info(
        "Traffic-light classification summary: %s",
        json.dumps(_summarize_traffic_light_classification_records(records_by_category), sort_keys=True),
    )

    for category, records in records_by_category.items():
        if not records:
            continue
        level = logging.WARNING if category.startswith("skipped_") else logging.INFO
        for record in records:
            _LOGGER.log(
                level,
                "Traffic-light classification record: %s",
                json.dumps(record, sort_keys=True),
            )


def _normalized_member_values(values) -> tuple[str, ...]:
    if not values:
        return tuple()
    return tuple(sorted(str(getattr(value, "value", value)) for value in values))


def _lanelet_compatibility_signature(lanelet) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Build a stable compatibility signature for grouping lanelets into SUMO edges.

    The signature intentionally includes lanelet type and user permissions so lanelets with
    incompatible semantics (e.g. urban + bicycleLane) are not merged into one SUMO edge.
    """
    lanelet_types = _normalized_member_values(getattr(lanelet, "lanelet_type", tuple()))
    user_one_way = _normalized_member_values(getattr(lanelet, "user_one_way", tuple()))
    user_bidirectional = _normalized_member_values(
        getattr(lanelet, "user_bidirectional", tuple())
    )
    return lanelet_types, user_one_way, user_bidirectional


def _split_lanelet_ids_by_compatibility(lanelet_network, lanelet_ids: list[int]) -> list[list[int]]:
    """
    Split an ordered lanelet-id list into contiguous segments with equal compatibility signatures.
    """
    if not lanelet_ids:
        return []

    segments: list[list[int]] = []
    current_segment: list[int] = []
    current_signature = None

    for lanelet_id in lanelet_ids:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet is None:
            raise RuntimeError(
                f"Cannot split lanelet compatibility groups: lanelet {lanelet_id} does not exist."
            )
        signature = _lanelet_compatibility_signature(lanelet)

        if current_signature is None or signature == current_signature:
            current_segment.append(lanelet_id)
            current_signature = signature
            continue

        segments.append(current_segment)
        current_segment = [lanelet_id]
        current_signature = signature

    if current_segment:
        segments.append(current_segment)

    return segments


def _needs_lane_grouping_patch(lanelets_module) -> bool:
    """Return True if installed commonroad_sumo still groups mixed lanelet semantics into one edge."""
    try:
        source = inspect.getsource(lanelets_module.partition_lanelet_network_into_edges_and_lanes)
    except (OSError, TypeError):
        return True

    if _LANE_GROUP_PATCH_FLAG in source:
        return False

    # Upstream fixed implementations should include semantics-aware splitting logic.
    expected_tokens = ("lanelet_type", "user_one_way", "user_bidirectional")
    return not all(token in source for token in expected_tokens)


def apply_commonroad_sumo_lane_grouping_patch() -> bool:
    """
    Patch commonroad_sumo lanelet grouping so mixed lane semantics are split into separate SUMO edges.

    Returns:
        True if patching was applied during this call, False otherwise.
    """
    global _LANE_GROUP_PATCHED_ONCE

    if _LANE_GROUP_PATCHED_ONCE:
        return False

    try:
        from commonroad_sumo.cr2sumo.map_converter import lanelets as cr2sumo_lanelets
        from commonroad_sumo.cr2sumo import map_converter as cr2sumo_map_converter_package
        from commonroad_sumo.cr2sumo.map_converter import (
            map_converter as cr2sumo_map_converter_module,
        )
    except ImportError:
        _LOGGER.debug(
            "commonroad_sumo is not available; skipping CR->SUMO lane-group compatibility patch."
        )
        return False

    if getattr(cr2sumo_lanelets, _LANE_GROUP_PATCH_FLAG, False):
        _LANE_GROUP_PATCHED_ONCE = True
        return False

    if not _needs_lane_grouping_patch(cr2sumo_lanelets):
        setattr(cr2sumo_lanelets, _LANE_GROUP_PATCH_FLAG, True)
        _LANE_GROUP_PATCHED_ONCE = True
        _LOGGER.debug(
            "commonroad_sumo lane grouping already appears compatibility-aware; no patch needed."
        )
        return False

    original_partition = cr2sumo_lanelets.partition_lanelet_network_into_edges_and_lanes

    def partition_lanelet_network_into_edges_and_lanes_compatible(lanelet_network):
        lanelet_ids_by_edge_ids = original_partition(lanelet_network)
        split_lanelet_ids_by_edge_ids = {}
        num_splits = 0

        for _edge_id, lanelet_ids in lanelet_ids_by_edge_ids.items():
            segments = _split_lanelet_ids_by_compatibility(lanelet_network, lanelet_ids)
            if len(segments) > 1:
                num_splits += len(segments) - 1

            for segment in segments:
                # Keep right-most lanelet id as edge-id key for each contiguous segment.
                split_lanelet_ids_by_edge_ids[segment[0]] = segment

        if num_splits:
            _LOGGER.info(
                "Split %d mixed lanelet groups into compatibility-homogeneous SUMO edges.",
                num_splits,
            )

        return split_lanelet_ids_by_edge_ids

    cr2sumo_lanelets.partition_lanelet_network_into_edges_and_lanes = (
        partition_lanelet_network_into_edges_and_lanes_compatible
    )
    cr2sumo_map_converter_package.partition_lanelet_network_into_edges_and_lanes = (
        partition_lanelet_network_into_edges_and_lanes_compatible
    )
    cr2sumo_map_converter_module.partition_lanelet_network_into_edges_and_lanes = (
        partition_lanelet_network_into_edges_and_lanes_compatible
    )
    cr2sumo_lanelets._crdesigner_original_partition_lanelet_network_into_edges_and_lanes = (
        original_partition
    )
    setattr(cr2sumo_lanelets, _LANE_GROUP_PATCH_FLAG, True)
    _LANE_GROUP_PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO lane grouping compatibility patch.")
    return True


def _collect_reachable_new_edge_ids(start_edges, new_edge_ids, next_attr: str) -> set[int]:
    """Traverse edge graph and collect reachable edge IDs that survived in new_edges."""
    queue = list(start_edges)
    visited = set()
    reachable_new_edge_ids = set()

    while queue:
        current = queue.pop()
        current_id = getattr(current, "id", None)
        if current_id is None or current_id in visited:
            continue
        visited.add(current_id)

        if current_id in new_edge_ids:
            reachable_new_edge_ids.add(current_id)
            continue

        queue.extend(getattr(current, next_attr, []))

    return reachable_new_edge_ids


def _find_unique_upstream_replacement_edge_id(
    converter, removed_edge_id: int, new_edge_ids: set[int] | None = None
) -> int | None:
    """
    Find a unique surviving edge upstream of a removed edge.

    Returns:
        Surviving edge ID if exactly one upstream candidate exists, otherwise None.
    """
    removed_edge = converter.edges.get(removed_edge_id)
    if removed_edge is None:
        return None

    if new_edge_ids is None:
        new_edge_ids = set(converter.new_edges.keys())

    reachable_upstream_new_edges = _collect_reachable_new_edge_ids(
        start_edges=getattr(removed_edge, "incoming", []),
        new_edge_ids=new_edge_ids,
        next_attr="incoming",
    )

    if len(reachable_upstream_new_edges) != 1:
        return None

    return next(iter(reachable_upstream_new_edges))


def apply_commonroad_sumo_traffic_light_patch() -> bool:
    """
    Patch commonroad_sumo CR->SUMO traffic-light conversion to tolerate ambiguous lanelet successors.

    The upstream converter can raise ValueError when a traffic-light lanelet is not mapped to an
    intersection and does not have exactly one successor. We skip such lanelets during traffic-light
    encoding instead of aborting the whole map conversion.

    Returns:
        True if patching was applied during this call, False otherwise.
    """
    global _TL_PATCHED_ONCE

    if _TL_PATCHED_ONCE:
        return False

    try:
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
    except ImportError:
        _LOGGER.debug(
            "commonroad_sumo is not available; skipping CR->SUMO traffic-light compatibility patch."
        )
        return False

    if getattr(cr2sumo_map_converter, _TL_PATCH_FLAG, False):
        _TL_PATCHED_ONCE = True
        return False

    if not _needs_traffic_light_patch():
        setattr(cr2sumo_map_converter, _TL_PATCH_FLAG, True)
        _TL_PATCHED_ONCE = True
        _LOGGER.debug(
            "commonroad_sumo traffic-light conversion already handles ambiguous successors; no patch needed."
        )
        return False

    original_create_traffic_lights = CR2SumoMapConverter._create_traffic_lights

    def create_traffic_lights_safe(self):
        incoming_lanelet_2_intersection = self._lanelet_network.map_inc_lanelets_to_intersections
        new_edge_ids = set(self.new_edges.keys())
        temporarily_disabled = []
        temporarily_remapped_lanelet_edges = []
        records_by_category = {
            "normal": [],
            "remapped": [],
            "skipped_ambiguous": [],
            "skipped_removed_no_unique_upstream": [],
        }
        ambiguous_lanelet_ids = []
        removed_lanelet_ids_no_unique_upstream = []
        skipped_ambiguous_successor = 0
        remapped_removed_edges = 0
        skipped_removed_edge_no_unique_upstream = 0

        for lanelet in self._lanelet_network.lanelets:
            if not lanelet.traffic_lights:
                continue

            has_intersection_mapping = lanelet.lanelet_id in incoming_lanelet_2_intersection

            # Keep existing safeguard: skip lanelets where direction->connection mapping is undefined.
            if not has_intersection_mapping and len(lanelet.successor) != 1:
                original_lights = lanelet.traffic_lights
                temporarily_disabled.append((lanelet, original_lights))
                lanelet.traffic_lights = _empty_container_like(original_lights)
                skipped_ambiguous_successor += 1
                ambiguous_lanelet_ids.append(str(lanelet.lanelet_id))
                records_by_category["skipped_ambiguous"].append(
                    _build_traffic_light_classification_record(
                        "skipped_ambiguous",
                        lanelet,
                        traffic_light_ids=original_lights,
                        edge_id=self.lanelet_id2edge_id.get(lanelet.lanelet_id),
                        has_intersection_mapping=has_intersection_mapping,
                        note="no intersection mapping and successor count != 1",
                    )
                )
                continue

            lanelet_edge_id = self.lanelet_id2edge_id.get(lanelet.lanelet_id)
            if lanelet_edge_id is None:
                original_lights = lanelet.traffic_lights
                temporarily_disabled.append((lanelet, original_lights))
                lanelet.traffic_lights = _empty_container_like(original_lights)
                skipped_removed_edge_no_unique_upstream += 1
                removed_lanelet_ids_no_unique_upstream.append(str(lanelet.lanelet_id))
                records_by_category["skipped_removed_no_unique_upstream"].append(
                    _build_traffic_light_classification_record(
                        "skipped_removed_no_unique_upstream",
                        lanelet,
                        traffic_light_ids=original_lights,
                        edge_id=lanelet_edge_id,
                        has_intersection_mapping=has_intersection_mapping,
                        note="lanelet had no edge mapping",
                    )
                )
                continue

            if lanelet_edge_id not in self.new_edges:
                replacement_edge_id = _find_unique_upstream_replacement_edge_id(
                    self, lanelet_edge_id, new_edge_ids
                )
                if replacement_edge_id is None:
                    original_lights = lanelet.traffic_lights
                    temporarily_disabled.append((lanelet, original_lights))
                    lanelet.traffic_lights = _empty_container_like(original_lights)
                    skipped_removed_edge_no_unique_upstream += 1
                    removed_lanelet_ids_no_unique_upstream.append(str(lanelet.lanelet_id))
                    records_by_category["skipped_removed_no_unique_upstream"].append(
                        _build_traffic_light_classification_record(
                            "skipped_removed_no_unique_upstream",
                            lanelet,
                            traffic_light_ids=original_lights,
                            edge_id=lanelet_edge_id,
                            has_intersection_mapping=has_intersection_mapping,
                            note="removed edge had no unique upstream surviving replacement",
                        )
                    )
                    continue

                temporarily_remapped_lanelet_edges.append((lanelet.lanelet_id, lanelet_edge_id))
                self.lanelet_id2edge_id[lanelet.lanelet_id] = replacement_edge_id
                remapped_removed_edges += 1
                records_by_category["remapped"].append(
                    _build_traffic_light_classification_record(
                        "remapped",
                        lanelet,
                        edge_id=lanelet_edge_id,
                        replacement_edge_id=replacement_edge_id,
                        has_intersection_mapping=has_intersection_mapping,
                    )
                )
                continue

            records_by_category["normal"].append(
                _build_traffic_light_classification_record(
                    "normal",
                    lanelet,
                    edge_id=lanelet_edge_id,
                    has_intersection_mapping=has_intersection_mapping,
                )
            )

        _log_traffic_light_classification_records(records_by_category)

        if skipped_ambiguous_successor:
            _LOGGER.warning(
                "Skipping traffic-light encoding on %d lanelets with ambiguous successors "
                "(no intersection mapping and successor count != 1). Example lanelet ids: %s",
                skipped_ambiguous_successor,
                ", ".join(ambiguous_lanelet_ids[:10]),
            )

        if remapped_removed_edges:
            _LOGGER.info(
                "Remapped traffic-light lanelets from removed edges to unique upstream surviving edges: %d",
                remapped_removed_edges,
            )

        if skipped_removed_edge_no_unique_upstream:
            _LOGGER.warning(
                "Skipped traffic-light encoding on %d lanelets whose edge was removed and had no "
                "unique upstream surviving replacement. Example lanelet ids: %s",
                skipped_removed_edge_no_unique_upstream,
                ", ".join(removed_lanelet_ids_no_unique_upstream[:10]),
            )

        try:
            return original_create_traffic_lights(self)
        finally:
            for lanelet, original_lights in temporarily_disabled:
                lanelet.traffic_lights = original_lights
            for lanelet_id, original_edge_id in temporarily_remapped_lanelet_edges:
                self.lanelet_id2edge_id[lanelet_id] = original_edge_id

    CR2SumoMapConverter._create_traffic_lights = create_traffic_lights_safe
    cr2sumo_map_converter._crdesigner_original_create_traffic_lights = original_create_traffic_lights
    setattr(cr2sumo_map_converter, _TL_PATCH_FLAG, True)
    _TL_PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO traffic-light compatibility patch.")
    return True
