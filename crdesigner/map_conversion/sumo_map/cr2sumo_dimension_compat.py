from __future__ import annotations

from collections import defaultdict
import inspect
import logging
import os

import numpy as np

_LOGGER = logging.getLogger(__name__)

_PATCH_FLAG = "_crdesigner_nd_patch_applied"
_PATCHED_ONCE = False
_TL_PATCH_FLAG = "_crdesigner_tl_patch_applied"
_TL_PATCHED_ONCE = False
_LANE_GROUP_PATCH_FLAG = "_crdesigner_lane_group_patch_applied"
_LANE_GROUP_PATCHED_ONCE = False
_NETCONVERT_TLS_PATCH_FLAG = "_crdesigner_netconvert_tls_patch_applied"
_NETCONVERT_TLS_PATCHED_ONCE = False

_JP_DEFAULT_GREEN_DURATION = 50.0
_JP_DEFAULT_YELLOW_DURATION = 10.0
_JP_DEFAULT_ALL_RED_DURATION = 5.0


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


def _needs_netconvert_tls_patch(map_converter_module) -> bool:
    """Return True if installed commonroad_sumo still lets netconvert guess/group signals."""
    try:
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter

        source = inspect.getsource(CR2SumoMapConverter.merge_intermediate_files)
        convert_function = getattr(
            map_converter_module, "convert_intermediate_sumo_project_with_netconvert", None
        )
        if convert_function is not None:
            source += inspect.getsource(convert_function)
    except (ImportError, OSError, TypeError):
        return True

    if "--tls.guess-signals=true" in source or "--tls.group-signals=true" in source:
        return True
    return "--tllogic-files=" not in source


def _log_netconvert_output(netconvert_result: str) -> None:
    for line in netconvert_result.splitlines():
        if line.startswith("Warning"):
            warning_message = line.lstrip("Warning: ")
            _LOGGER.debug(
                "netconvert produced a warning while converting network: %s",
                warning_message,
            )
        else:
            _LOGGER.debug("netconvert output: %s", line)


def apply_commonroad_sumo_netconvert_tls_patch() -> bool:
    """
    Patch commonroad_sumo netconvert handoff so explicit TLLOGIC files win over guessed signals.

    Returns:
        True if patching was applied during this call, False otherwise.
    """
    global _NETCONVERT_TLS_PATCHED_ONCE

    if _NETCONVERT_TLS_PATCHED_ONCE:
        return False

    try:
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
    except ImportError:
        _LOGGER.debug(
            "commonroad_sumo is not available; skipping CR->SUMO netconvert TLLOGIC patch."
        )
        return False

    if getattr(cr2sumo_map_converter, _NETCONVERT_TLS_PATCH_FLAG, False):
        _NETCONVERT_TLS_PATCHED_ONCE = True
        return False

    if not _needs_netconvert_tls_patch(cr2sumo_map_converter):
        setattr(cr2sumo_map_converter, _NETCONVERT_TLS_PATCH_FLAG, True)
        _NETCONVERT_TLS_PATCHED_ONCE = True
        _LOGGER.debug(
            "commonroad_sumo netconvert handoff already appears TLLOGIC-aware; no patch needed."
        )
        return False

    original_merge_intermediate_files = CR2SumoMapConverter.merge_intermediate_files
    original_convert_intermediate = getattr(
        cr2sumo_map_converter, "convert_intermediate_sumo_project_with_netconvert", None
    )

    def _common_tls_arguments(sumo_intermediate_project, sumo_project):
        args = [
            "--tls.guess-signals=false",
            "--tls.group-signals=false",
            "--offset.disable-normalization=true",
            f"--node-files={sumo_intermediate_project.get_file_path(cr2sumo_map_converter.SumoIntermediateFileType.NODES)}",
            f"--edge-files={sumo_intermediate_project.get_file_path(cr2sumo_map_converter.SumoIntermediateFileType.EDGES)}",
            f"--connection-files={sumo_intermediate_project.get_file_path(cr2sumo_map_converter.SumoIntermediateFileType.CONNECTIONS)}",
            f"--tllogic-files={sumo_intermediate_project.get_file_path(cr2sumo_map_converter.SumoIntermediateFileType.TLLOGICS)}",
            f"--output-file={sumo_project.get_file_path(cr2sumo_map_converter.SumoFileType.NET)}",
        ]

        try:
            args.insert(
                -1,
                f"--type-files={sumo_intermediate_project.get_file_path(cr2sumo_map_converter.SumoIntermediateFileType.TYPES)}",
            )
        except Exception:
            pass

        return args

    def merge_intermediate_files_with_explicit_tllogic(
        self,
        sumo_intermediate_project,
        cleanup: bool,
    ):
        sumo_project = cr2sumo_map_converter.SumoProject.from_intermediate_sumo_project(
            sumo_intermediate_project
        )
        args = [
            "--no-turnarounds=true",
            "--junctions.internal-link-detail=20",
            "--geometry.avoid-overlap=true",
            "--geometry.remove.keep-edges.explicit=true",
            "--geometry.remove.min-length=0.0",
            *_common_tls_arguments(sumo_intermediate_project, sumo_project),
        ]
        random_seed = getattr(getattr(self, "_conf", None), "random_seed", None)
        if random_seed is not None:
            args.append(f"--seed={random_seed}")

        netconvert_result = cr2sumo_map_converter.execute_sumo_application(
            cr2sumo_map_converter.SumoApplication.NETCONVERT,
            args,
        )
        if netconvert_result is None:
            cr2sumo_map_converter._LOGGER.error(
                "Failed to merge intermediate files: netconvert failed with an unknown error!"
            )
            return None

        _log_netconvert_output(netconvert_result)

        if cleanup:
            sumo_intermediate_project.cleanup()

        return sumo_project

    def convert_intermediate_sumo_project_with_explicit_tllogic(
        sumo_intermediate_project,
        cleanup: bool,
    ):
        sumo_project = cr2sumo_map_converter.SumoProject.from_intermediate_sumo_project(
            sumo_intermediate_project
        )
        args = [
            "--no-turnarounds=true",
            "--junctions.join=true",
            "--junctions.join-dist=20",
            "--junctions.join-same=true",
            "--junctions.join-turns=true",
            "--junctions.scurve-stretch=5.0",
            "--junctions.internal-link-detail=20",
            "--junctions.corner-detail=20",
            "--junctions.endpoint-shape=true",
            "--edges.join=true",
            "--ramps.guess=true",
            "--plain.extend-edge-shape=true",
            "--geometry.avoid-overlap=true",
            "--geometry.remove.min-length=5.0",
            "--fringe.guess=true",
            *_common_tls_arguments(sumo_intermediate_project, sumo_project),
        ]

        netconvert_result = cr2sumo_map_converter.execute_sumo_application(
            cr2sumo_map_converter.SumoApplication.NETCONVERT,
            args,
        )
        if netconvert_result is None:
            cr2sumo_map_converter._LOGGER.error(
                "Failed to merge intermediate files: netconvert failed with an unknown error!"
            )
            return None

        _log_netconvert_output(netconvert_result)

        if cleanup:
            sumo_intermediate_project.cleanup()

        return sumo_project

    CR2SumoMapConverter.merge_intermediate_files = merge_intermediate_files_with_explicit_tllogic
    cr2sumo_map_converter._crdesigner_original_merge_intermediate_files = (
        original_merge_intermediate_files
    )
    if original_convert_intermediate is not None:
        cr2sumo_map_converter.convert_intermediate_sumo_project_with_netconvert = (
            convert_intermediate_sumo_project_with_explicit_tllogic
        )
        cr2sumo_map_converter._crdesigner_original_convert_intermediate_sumo_project = (
            original_convert_intermediate
        )
    setattr(cr2sumo_map_converter, _NETCONVERT_TLS_PATCH_FLAG, True)
    _NETCONVERT_TLS_PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO netconvert TLLOGIC compatibility patch.")
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


def _connection_sort_key(connection) -> tuple:
    from_edge_id = getattr(getattr(connection, "from_edge", None), "id", -1)
    to_edge_id = getattr(getattr(connection, "to_edge", None), "id", -1)
    via_identifier = getattr(connection, "via", None) or getattr(connection, "via_id", None) or ""
    shape = getattr(connection, "shape", None)
    shape_key = ()
    if isinstance(shape, np.ndarray) and shape.ndim == 2 and len(shape) >= 1:
        shape_key = tuple(shape[0].tolist()) + tuple(shape[-1].tolist())
    return from_edge_id, to_edge_id, str(via_identifier), shape_key


def _incoming_direction_vector(lanelet_network, incoming_element) -> np.ndarray:
    incoming_lanelet_ids = sorted(getattr(incoming_element, "incoming_lanelets", []) or [])
    if not incoming_lanelet_ids:
        return np.array([1.0, 0.0])

    lanelet = lanelet_network.find_lanelet_by_id(incoming_lanelet_ids[0])
    if lanelet is None or len(getattr(lanelet, "center_vertices", [])) < 2:
        return np.array([1.0, 0.0])

    end_vertex = lanelet.center_vertices[-1][:2]
    start_index = -3 if len(lanelet.center_vertices) >= 3 else -2
    start_vertex = lanelet.center_vertices[start_index][:2]
    direction = end_vertex - start_vertex
    if np.linalg.norm(direction) == 0:
        return np.array([1.0, 0.0])
    return direction


def _incoming_angle_degrees(lanelet_network, incoming_element) -> float:
    direction = _incoming_direction_vector(lanelet_network, incoming_element)
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _circular_angle_difference_degrees(first_angle: float, second_angle: float) -> float:
    difference = abs(first_angle - second_angle) % 360.0
    return difference if difference <= 180.0 else 360.0 - difference


def _build_japanese_default_phase_groups(lanelet_network, incoming_elements_by_id) -> list[list[int]]:
    if not incoming_elements_by_id:
        return []

    angles = {
        incoming_id: _incoming_angle_degrees(lanelet_network, incoming_element)
        for incoming_id, incoming_element in incoming_elements_by_id.items()
    }
    unused_incoming_ids = set(incoming_elements_by_id.keys())
    phase_groups: list[list[int]] = []

    while unused_incoming_ids:
        base_incoming_id = min(unused_incoming_ids, key=lambda incoming_id: angles[incoming_id])
        unused_incoming_ids.remove(base_incoming_id)

        opposite_incoming_id = None
        opposite_score = None
        for candidate_incoming_id in sorted(unused_incoming_ids):
            angle_difference = _circular_angle_difference_degrees(
                angles[base_incoming_id], angles[candidate_incoming_id]
            )
            score = abs(180.0 - angle_difference)
            if score > 45.0:
                continue
            if opposite_score is None or score < opposite_score:
                opposite_incoming_id = candidate_incoming_id
                opposite_score = score

        phase_group = [base_incoming_id]
        if opposite_incoming_id is not None:
            unused_incoming_ids.remove(opposite_incoming_id)
            phase_group.append(opposite_incoming_id)

        phase_group.sort(key=lambda incoming_id: (angles[incoming_id], incoming_id))
        phase_groups.append(phase_group)

    phase_groups.sort(key=lambda group: min(angles[incoming_id] for incoming_id in group))
    return phase_groups


def _shape_based_conflict(connection_a, connection_b, lines_intersect, curvature_fn) -> tuple[bool, bool]:
    shape_a = getattr(connection_a, "shape", None)
    shape_b = getattr(connection_b, "shape", None)
    if not (
        isinstance(shape_a, np.ndarray)
        and shape_a.ndim == 2
        and len(shape_a) >= 2
        and isinstance(shape_b, np.ndarray)
        and shape_b.ndim == 2
        and len(shape_b) >= 2
    ):
        return False, False

    if not lines_intersect(shape_a, shape_b):
        return False, False

    direction_dot = np.dot(shape_a[-1] - shape_a[0], shape_b[-1] - shape_b[0])
    if direction_dot >= 0:
        return False, False

    return True, curvature_fn(shape_a) <= curvature_fn(shape_b)


def _build_japanese_default_green_state(
    ordered_connections,
    active_indices: list[int],
    signal_state_enum,
    lines_intersect,
    curvature_fn,
):
    state = [signal_state_enum.RED] * len(ordered_connections)
    for active_index in active_indices:
        state[active_index] = signal_state_enum.GREEN

    for left_pos, left_index in enumerate(active_indices):
        for right_index in active_indices[left_pos + 1 :]:
            conflicts, left_has_priority = _shape_based_conflict(
                ordered_connections[left_index],
                ordered_connections[right_index],
                lines_intersect,
                curvature_fn,
            )
            if not conflicts:
                continue
            if left_has_priority:
                state[left_index] = signal_state_enum.GREEN_PRIORITY
                state[right_index] = signal_state_enum.GREEN
            else:
                state[left_index] = signal_state_enum.GREEN
                state[right_index] = signal_state_enum.GREEN_PRIORITY

    return state


def _build_uniform_state(length: int, active_indices: list[int], active_state, inactive_state):
    state = [inactive_state] * length
    for active_index in active_indices:
        state[active_index] = active_state
    return state


def _build_japanese_default_tls_program(
    lanelet_network,
    node,
    incoming_connections_by_id,
    incoming_elements_by_id,
    tls_program_cls,
    phase_cls,
    signal_state_enum,
    node_type_enum,
    lines_intersect,
    curvature_fn,
):
    ordered_connections = sorted(
        {
            connection
            for connections in incoming_connections_by_id.values()
            for connection in connections
        },
        key=_connection_sort_key,
    )
    if not ordered_connections:
        return None, []

    node.type = node_type_enum.TRAFFIC_LIGHT
    tls_program = tls_program_cls(str(node.id), offset=0, program_id=f"jp_default_{node.id}")
    connection_indices = {
        connection: index for index, connection in enumerate(ordered_connections)
    }

    for phase_group in _build_japanese_default_phase_groups(
        lanelet_network, incoming_elements_by_id
    ):
        active_indices = sorted(
            {
                connection_indices[connection]
                for incoming_id in phase_group
                for connection in incoming_connections_by_id.get(incoming_id, set())
                if connection in connection_indices
            }
        )
        if not active_indices:
            continue

        tls_program.add_phase(
            phase_cls(
                _JP_DEFAULT_GREEN_DURATION,
                _build_japanese_default_green_state(
                    ordered_connections,
                    active_indices,
                    signal_state_enum,
                    lines_intersect,
                    curvature_fn,
                ),
            )
        )
        tls_program.add_phase(
            phase_cls(
                _JP_DEFAULT_YELLOW_DURATION,
                _build_uniform_state(
                    len(ordered_connections),
                    active_indices,
                    signal_state_enum.YELLOW,
                    signal_state_enum.RED,
                ),
            )
        )
        tls_program.add_phase(
            phase_cls(
                _JP_DEFAULT_ALL_RED_DURATION,
                [signal_state_enum.RED] * len(ordered_connections),
            )
        )

    if not tls_program.phases:
        return None, []

    for index, connection in enumerate(ordered_connections):
        connection.tls = tls_program
        connection.tl_link = index

    return tls_program, ordered_connections


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
        from commonroad_sumo.cr2sumo.map_converter.traffic_light import (
            Phase as TrafficLightPhase,
            SignalState as TrafficLightSignalState,
            TLSProgram as TrafficLightTLSProgram,
            compute_max_curvature_from_polyline,
            lines_intersect,
        )
        from commonroad_sumo.sumolib.net import NodeType
        from commonroad.scenario.traffic_light import TrafficLightDirection
    except ImportError:
        _LOGGER.debug(
            "commonroad_sumo is not available; skipping CR->SUMO traffic-light compatibility patch."
        )
        return False

    if getattr(cr2sumo_map_converter, _TL_PATCH_FLAG, False):
        _TL_PATCHED_ONCE = True
        return False

    original_create_traffic_lights = CR2SumoMapConverter._create_traffic_lights

    def create_traffic_lights_safe(self):
        cr_traffic_lights = self._lanelet_network._traffic_lights
        debug_tl_nodes = {
            node_id.strip()
            for node_id in os.environ.get("CRDESIGNER_DEBUG_TL_NODES", "").split(",")
            if node_id.strip()
        }

        def required_direction_keys(direction) -> set:
            if direction in (None, TrafficLightDirection.ALL):
                return set()

            required = set()
            if direction in (
                TrafficLightDirection.RIGHT,
                TrafficLightDirection.LEFT_RIGHT,
                TrafficLightDirection.STRAIGHT_RIGHT,
            ):
                required.add(TrafficLightDirection.RIGHT)
            if direction in (
                TrafficLightDirection.LEFT,
                TrafficLightDirection.LEFT_RIGHT,
                TrafficLightDirection.LEFT_STRAIGHT,
            ):
                required.add(TrafficLightDirection.LEFT)
            if direction in (
                TrafficLightDirection.STRAIGHT,
                TrafficLightDirection.STRAIGHT_RIGHT,
                TrafficLightDirection.LEFT_STRAIGHT,
            ):
                required.add(TrafficLightDirection.STRAIGHT)
            return required

        def available_direction_fallback(available_directions: set):
            available_directions = frozenset(available_directions)
            direction_map = {
                frozenset({TrafficLightDirection.LEFT}): TrafficLightDirection.LEFT,
                frozenset({TrafficLightDirection.RIGHT}): TrafficLightDirection.RIGHT,
                frozenset({TrafficLightDirection.STRAIGHT}): TrafficLightDirection.STRAIGHT,
                frozenset(
                    {TrafficLightDirection.LEFT, TrafficLightDirection.STRAIGHT}
                ): TrafficLightDirection.LEFT_STRAIGHT,
                frozenset(
                    {TrafficLightDirection.STRAIGHT, TrafficLightDirection.RIGHT}
                ): TrafficLightDirection.STRAIGHT_RIGHT,
                frozenset(
                    {TrafficLightDirection.LEFT, TrafficLightDirection.RIGHT}
                ): TrafficLightDirection.LEFT_RIGHT,
                frozenset(
                    {
                        TrafficLightDirection.LEFT,
                        TrafficLightDirection.STRAIGHT,
                        TrafficLightDirection.RIGHT,
                    }
                ): TrafficLightDirection.ALL,
            }
            return direction_map.get(available_directions)

        def select_connections_for_light(lanelet, light, direction_2_connections, available_directions):
            if (
                len(lanelet.successor) == 1
                or not light.direction
                or light.direction == TrafficLightDirection.ALL
            ):
                return set().union(*direction_2_connections.values()) if direction_2_connections else set()

            required_directions = required_direction_keys(light.direction)
            if not required_directions or required_directions.issubset(available_directions):
                selected_directions = required_directions or available_directions
            else:
                fallback_direction = available_direction_fallback(available_directions)
                if fallback_direction is None:
                    downgraded_direction_lanelet_ids.add(str(lanelet.lanelet_id))
                    return set()
                selected_directions = required_direction_keys(fallback_direction)
                downgraded_direction_lanelet_ids.add(str(lanelet.lanelet_id))

            connections = set()
            for direction in selected_directions:
                connections |= direction_2_connections.get(direction, set())
            return connections

        def normalize_group_state(states):
            state_set = set(states)
            if TrafficLightSignalState.GREEN_PRIORITY in state_set:
                return TrafficLightSignalState.GREEN_PRIORITY
            if TrafficLightSignalState.GREEN in state_set:
                return TrafficLightSignalState.GREEN
            if TrafficLightSignalState.GREEN_TURN_RIGHT in state_set:
                return TrafficLightSignalState.GREEN_TURN_RIGHT
            if TrafficLightSignalState.YELLOW in state_set:
                return TrafficLightSignalState.YELLOW
            if TrafficLightSignalState.RED_YELLOW in state_set:
                return TrafficLightSignalState.RED_YELLOW
            if TrafficLightSignalState.BLINKING in state_set:
                return TrafficLightSignalState.BLINKING
            if TrafficLightSignalState.NO_SIGNAL in state_set:
                return TrafficLightSignalState.NO_SIGNAL
            return TrafficLightSignalState.RED

        def direction_2_connections_for_lanelet(lanelet, edge, intersection):
            if intersection is not None:
                incoming_elem = intersection.map_incoming_lanelets[lanelet.lanelet_id]
                connections_init = {
                    TrafficLightDirection.STRAIGHT: incoming_elem.successors_straight,
                    TrafficLightDirection.LEFT: incoming_elem.successors_left,
                    TrafficLightDirection.RIGHT: incoming_elem.successors_right,
                }
            elif len(lanelet.successor) == 1:
                successor_set = set(lanelet.successor)
                connections_init = {
                    TrafficLightDirection.STRAIGHT: successor_set,
                    TrafficLightDirection.LEFT: successor_set,
                    TrafficLightDirection.RIGHT: successor_set,
                }
            else:
                return {}

            connections = {}
            for direction, init_queue in connections_init.items():
                queue = []
                for successor_lanelet_id in init_queue:
                    successor_edge_id = self.lanelet_id2edge_id.get(successor_lanelet_id)
                    successor_edge = self.edges.get(successor_edge_id)
                    if successor_edge is not None:
                        queue.append(successor_edge)

                visited = set()
                reachable_connections = set()
                while queue:
                    current = queue.pop()
                    current_id = getattr(current, "id", None)
                    if current_id is None or current_id in visited:
                        continue
                    visited.add(current_id)

                    if current_id in self.new_edges:
                        reachable_connections |= {
                            connection
                            for connection in self._new_connections
                            if connection.from_edge == edge and connection.to_edge == current
                        }
                        continue

                    queue.extend(getattr(current, "outgoing", []))

                if reachable_connections:
                    connections[direction] = reachable_connections

            return connections

        incoming_lanelet_2_intersection = self._lanelet_network.map_inc_lanelets_to_intersections
        new_edge_ids = set(self.new_edges.keys())
        node_key_2_node = {}
        node_2_traffic_light = defaultdict(set)
        light_2_connections = defaultdict(set)
        node_2_incoming_connections = defaultdict(lambda: defaultdict(set))
        node_2_incoming_elements = defaultdict(dict)
        ambiguous_lanelet_ids = []
        removed_lanelet_ids_no_unique_upstream = []
        downgraded_direction_lanelet_ids = set()
        skipped_ambiguous_successor = 0
        remapped_removed_edges = 0
        skipped_removed_edge_no_unique_upstream = 0

        for lanelet in self._lanelet_network.lanelets:
            if not lanelet.traffic_lights:
                continue

            # Keep existing safeguard: skip lanelets where direction->connection mapping is undefined.
            if (
                lanelet.lanelet_id not in incoming_lanelet_2_intersection
                and len(lanelet.successor) != 1
            ):
                skipped_ambiguous_successor += 1
                ambiguous_lanelet_ids.append(str(lanelet.lanelet_id))
                continue

            lanelet_edge_id = self.lanelet_id2edge_id.get(lanelet.lanelet_id)
            if lanelet_edge_id is None:
                skipped_removed_edge_no_unique_upstream += 1
                removed_lanelet_ids_no_unique_upstream.append(str(lanelet.lanelet_id))
                continue

            if lanelet_edge_id not in self.new_edges:
                replacement_edge_id = _find_unique_upstream_replacement_edge_id(
                    self, lanelet_edge_id, new_edge_ids
                )
                if replacement_edge_id is None:
                    skipped_removed_edge_no_unique_upstream += 1
                    removed_lanelet_ids_no_unique_upstream.append(str(lanelet.lanelet_id))
                    continue

                lanelet_edge_id = replacement_edge_id
                remapped_removed_edges += 1

            edge = self.new_edges.get(lanelet_edge_id)
            if edge is None:
                continue

            intersection = (
                incoming_lanelet_2_intersection[lanelet.lanelet_id]
                if lanelet.lanelet_id in incoming_lanelet_2_intersection
                else None
            )
            direction_2_connections = direction_2_connections_for_lanelet(lanelet, edge, intersection)
            available_directions = set(direction_2_connections.keys())
            if not available_directions:
                continue

            node = edge.to_node
            node_key = getattr(node, "id", None)
            if node_key is None:
                node_key = id(node)
            node_key_2_node[node_key] = node
            active_traffic_lights = [
                cr_traffic_lights[traffic_light_id]
                for traffic_light_id in set(lanelet.traffic_lights)
                if traffic_light_id in cr_traffic_lights
                and getattr(cr_traffic_lights[traffic_light_id], "active", True)
            ]
            if not active_traffic_lights:
                continue

            if intersection is not None:
                incoming_elem = intersection.map_incoming_lanelets.get(lanelet.lanelet_id)
                if incoming_elem is not None:
                    incoming_connections = (
                        set().union(*direction_2_connections.values())
                        if direction_2_connections
                        else set()
                    )
                    if incoming_connections:
                        node_2_incoming_connections[node_key][incoming_elem.incoming_id] |= (
                            incoming_connections
                        )
                        node_2_incoming_elements[node_key][incoming_elem.incoming_id] = incoming_elem
                        continue

            for traffic_light_id in set(lanelet.traffic_lights):
                traffic_light = cr_traffic_lights.get(traffic_light_id)
                if traffic_light is None or traffic_light not in active_traffic_lights:
                    continue

                connections = select_connections_for_light(
                    lanelet, traffic_light, direction_2_connections, available_directions
                )
                if not connections:
                    continue

                node_2_traffic_light[node_key].add(traffic_light)
                light_2_connections[traffic_light] |= connections

        _LOGGER.warning(
            "Skipping traffic-light encoding on %d lanelets with ambiguous successors "
            "(no intersection mapping and successor count != 1). Example lanelet ids: %s",
            skipped_ambiguous_successor,
            ", ".join(ambiguous_lanelet_ids[:10]) if ambiguous_lanelet_ids else "none",
        )

        _LOGGER.info(
            "Remapped traffic-light lanelets from removed edges to unique upstream surviving edges: %d",
            remapped_removed_edges,
        )

        _LOGGER.warning(
            "Skipped traffic-light encoding on %d lanelets whose edge was removed and had no "
            "unique upstream surviving replacement. Example lanelet ids: %s",
            skipped_removed_edge_no_unique_upstream,
            ", ".join(removed_lanelet_ids_no_unique_upstream[:10])
            if removed_lanelet_ids_no_unique_upstream
            else "none",
        )

        _LOGGER.warning(
            "Adjusted traffic-light directions on %d lanelets whose requested turn buckets were "
            "not reachable in the generated SUMO graph. Example lanelet ids: %s",
            len(downgraded_direction_lanelet_ids),
            ", ".join(sorted(downgraded_direction_lanelet_ids)[:10])
            if downgraded_direction_lanelet_ids
            else "none",
        )

        encoder = cr2sumo_map_converter.TrafficLightEncoder(self._scenario.dt)
        for node_key, incoming_connections_by_id in node_2_incoming_connections.items():
            to_node = node_key_2_node[node_key]
            program, connections = _build_japanese_default_tls_program(
                self._lanelet_network,
                to_node,
                incoming_connections_by_id,
                node_2_incoming_elements[node_key],
                TrafficLightTLSProgram,
                TrafficLightPhase,
                TrafficLightSignalState,
                NodeType,
                lines_intersect,
                compute_max_curvature_from_polyline,
            )
            if program is None:
                continue
            if str(getattr(to_node, "id", "")) in debug_tl_nodes:
                _LOGGER.warning(
                    "TL debug japanese-default node=%s incoming_groups=%s phases=%s",
                    getattr(to_node, "id", "unknown"),
                    {
                        incoming_id: len(connections_set)
                        for incoming_id, connections_set in incoming_connections_by_id.items()
                    },
                    ["".join(signal.value for signal in phase.state) for phase in program.phases],
                )
            self.traffic_light_signals.add_program(program)
            for connection in connections:
                self.traffic_light_signals.add_connection(connection)

        for node_key, lights in node_2_traffic_light.items():
            if node_key in node_2_incoming_connections:
                continue
            to_node = node_key_2_node[node_key]
            ordered_lights = sorted(
                (light for light in lights if light_2_connections.get(light)),
                key=lambda light: light.traffic_light_id,
            )
            if not ordered_lights:
                continue
            try:
                program, connections = encoder.encode(
                    to_node,
                    ordered_lights,
                    {light: light_2_connections[light] for light in ordered_lights},
                )
                if str(getattr(to_node, "id", "")) in debug_tl_nodes:
                    _LOGGER.warning(
                        "TL debug before normalize node=%s lights=%s connection_counts=%s phases=%s",
                        getattr(to_node, "id", "unknown"),
                        [light.traffic_light_id for light in ordered_lights],
                        {
                            light.traffic_light_id: len(light_2_connections[light])
                            for light in ordered_lights
                        },
                        ["".join(signal.value for signal in phase.state) for phase in program.phases],
                    )
                connection_indices = {connection: idx for idx, connection in enumerate(connections)}
                for light in ordered_lights:
                    indices = sorted(
                        connection_indices[connection]
                        for connection in light_2_connections[light]
                        if connection in connection_indices
                    )
                    if len(indices) <= 1:
                        continue
                    for phase in program.phases:
                        normalized_state = normalize_group_state(
                            [phase.state[idx] for idx in indices]
                        )
                        for idx in indices:
                            phase.state[idx] = normalized_state
                if str(getattr(to_node, "id", "")) in debug_tl_nodes:
                    _LOGGER.warning(
                        "TL debug after normalize node=%s phases=%s",
                        getattr(to_node, "id", "unknown"),
                        ["".join(signal.value for signal in phase.state) for phase in program.phases],
                    )
                self.traffic_light_signals.add_program(program)
                for connection in connections:
                    self.traffic_light_signals.add_connection(connection)
            except (RuntimeError, ValueError, TypeError):
                continue

    CR2SumoMapConverter._create_traffic_lights = create_traffic_lights_safe
    cr2sumo_map_converter._crdesigner_original_create_traffic_lights = original_create_traffic_lights
    setattr(cr2sumo_map_converter, _TL_PATCH_FLAG, True)
    _TL_PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO traffic-light compatibility patch.")
    return True
