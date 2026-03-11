from __future__ import annotations

import inspect
import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

_PATCH_FLAG = "_crdesigner_nd_patch_applied"
_PATCHED_ONCE = False
_TL_PATCH_FLAG = "_crdesigner_tl_patch_applied"
_TL_PATCHED_ONCE = False


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

        # Width erosion moves both boundaries inward by `radius`, so thin lanelets need
        # at least `2 * radius` total width to avoid self-intersection.
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
        temporarily_disabled = []

        for lanelet in self._lanelet_network.lanelets:
            if not lanelet.traffic_lights:
                continue
            if lanelet.lanelet_id in incoming_lanelet_2_intersection:
                continue
            if len(lanelet.successor) == 1:
                continue

            original_lights = lanelet.traffic_lights
            temporarily_disabled.append((lanelet, original_lights))
            lanelet.traffic_lights = _empty_container_like(original_lights)

        if temporarily_disabled:
            sample_ids = [str(la.lanelet_id) for la, _ in temporarily_disabled[:10]]
            _LOGGER.warning(
                "Skipping traffic-light encoding on %d lanelets with ambiguous successors "
                "(no intersection mapping and successor count != 1). Example lanelet ids: %s",
                len(temporarily_disabled),
                ", ".join(sample_ids),
            )

        try:
            return original_create_traffic_lights(self)
        finally:
            for lanelet, original_lights in temporarily_disabled:
                lanelet.traffic_lights = original_lights

    CR2SumoMapConverter._create_traffic_lights = create_traffic_lights_safe
    cr2sumo_map_converter._crdesigner_original_create_traffic_lights = original_create_traffic_lights
    setattr(cr2sumo_map_converter, _TL_PATCH_FLAG, True)
    _TL_PATCHED_ONCE = True
    _LOGGER.info("Applied commonroad_sumo CR->SUMO traffic-light compatibility patch.")
    return True
