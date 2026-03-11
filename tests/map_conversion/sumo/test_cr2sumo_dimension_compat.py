import importlib.util
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from crdesigner.map_conversion.sumo_map.cr2sumo_dimension_compat import (
    _find_unique_upstream_replacement_edge_id,
    _split_lanelet_ids_by_compatibility,
    apply_commonroad_sumo_lane_grouping_patch,
    apply_commonroad_sumo_nd_patch,
    apply_commonroad_sumo_traffic_light_patch,
)


@unittest.skipUnless(
    importlib.util.find_spec("commonroad_sumo") is not None
    and importlib.util.find_spec("commonroad") is not None,
    "commonroad/commonroad_sumo not installed",
)
class TestCR2SumoDimensionCompat(unittest.TestCase):
    class _FakeLanelet:
        def __init__(
            self,
            lanelet_id: int,
            lanelet_type=None,
            user_one_way=None,
            user_bidirectional=None,
        ):
            self.lanelet_id = lanelet_id
            self.lanelet_type = lanelet_type if lanelet_type is not None else set()
            self.user_one_way = user_one_way if user_one_way is not None else set()
            self.user_bidirectional = (
                user_bidirectional if user_bidirectional is not None else set()
            )

    class _FakeLaneletNetwork:
        def __init__(self, lanelets):
            self._lanelets = {lanelet.lanelet_id: lanelet for lanelet in lanelets}

        def find_lanelet_by_id(self, lanelet_id):
            return self._lanelets.get(lanelet_id)

    class _FakeEdge:
        def __init__(self, edge_id):
            self.id = edge_id
            self.incoming = []
            self.outgoing = []

    class _FakeConverter:
        def __init__(self, edges, new_edges):
            self.edges = edges
            self.new_edges = new_edges

    @classmethod
    def setUpClass(cls):
        apply_commonroad_sumo_nd_patch()
        from commonroad.scenario.lanelet import Lanelet
        from commonroad_sumo.cr2sumo.map_converter import util

        cls._lanelet_cls = Lanelet
        cls._util = util

    def _create_3d_lanelet(self):
        x_values = np.linspace(0.0, 20.0, 11)
        center = np.column_stack((x_values, np.zeros_like(x_values), np.linspace(1.0, 2.0, 11)))
        left = center + np.array([0.0, 1.8, 0.0])
        right = center + np.array([0.0, -1.8, 0.0])
        return self._lanelet_cls(left, center, right, lanelet_id=1)

    def test_resample_lanelet_supports_3d_vertices(self):
        lanelet = self._create_3d_lanelet()
        self._util.resample_lanelet(lanelet, step=0.75)

        self.assertEqual(3, lanelet.center_vertices.shape[1])
        self.assertEqual(3, lanelet.left_vertices.shape[1])
        self.assertEqual(3, lanelet.right_vertices.shape[1])

    def test_erode_lanelet_supports_3d_vertices(self):
        lanelet = self._create_3d_lanelet()
        self._util.erode_lanelet(lanelet, radius=0.4)

        self.assertEqual(3, lanelet.center_vertices.shape[1])
        self.assertEqual(3, lanelet.left_vertices.shape[1])
        self.assertEqual(3, lanelet.right_vertices.shape[1])

    def test_erode_lanelet_does_not_over_shrink_thin_lanelet(self):
        x_values = np.linspace(0.0, 20.0, 11)
        center = np.column_stack((x_values, np.zeros_like(x_values), np.linspace(1.0, 2.0, 11)))
        # Total width is 0.7 m (< 2 * 0.4), so width erosion must be skipped.
        left = center + np.array([0.0, 0.35, 0.0])
        right = center + np.array([0.0, -0.35, 0.0])
        lanelet = self._lanelet_cls(left, center, right, lanelet_id=2)

        self._util.erode_lanelet(lanelet, radius=0.4)
        min_width = np.min(np.linalg.norm(lanelet.left_vertices - lanelet.right_vertices, axis=1))
        self.assertGreater(min_width, 0.5)

    def test_apply_patch_is_idempotent(self):
        # Already applied in setUpClass.
        self.assertFalse(apply_commonroad_sumo_nd_patch())
        self.assertFalse(apply_commonroad_sumo_nd_patch())

    def test_a_traffic_light_patch_logs_zero_counts(self):
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
        from crdesigner.map_conversion.sumo_map import cr2sumo_dimension_compat as compat

        original_method = CR2SumoMapConverter._create_traffic_lights
        original_flag = getattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, None)

        try:
            compat._TL_PATCHED_ONCE = False
            if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)

            CR2SumoMapConverter._create_traffic_lights = lambda self: "ok"
            with patch.object(compat, "_needs_traffic_light_patch", return_value=True):
                self.assertTrue(apply_commonroad_sumo_traffic_light_patch())

            fake_converter = SimpleNamespace(
                _lanelet_network=SimpleNamespace(
                    lanelets=[],
                    map_inc_lanelets_to_intersections={},
                ),
                new_edges={},
                lanelet_id2edge_id={},
                edges={},
            )

            with self.assertLogs(
                "crdesigner.map_conversion.sumo_map.cr2sumo_dimension_compat", level="INFO"
            ) as logs:
                result = CR2SumoMapConverter._create_traffic_lights(fake_converter)

            self.assertEqual("ok", result)
            joined_logs = "\n".join(logs.output)
            self.assertIn(
                "Skipping traffic-light encoding on 0 lanelets with ambiguous successors",
                joined_logs,
            )
            self.assertIn(
                "Remapped traffic-light lanelets from removed edges to unique upstream surviving edges: 0",
                joined_logs,
            )
            self.assertIn(
                "Skipped traffic-light encoding on 0 lanelets whose edge was removed",
                joined_logs,
            )
        finally:
            CR2SumoMapConverter._create_traffic_lights = original_method
            compat._TL_PATCHED_ONCE = False
            if original_flag is None:
                if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                    delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)
            else:
                setattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, original_flag)

    def test_traffic_light_patch_adjusts_unreachable_direction(self):
        from commonroad.scenario.traffic_light import (
            TrafficLight,
            TrafficLightCycle,
            TrafficLightCycleElement,
            TrafficLightDirection,
            TrafficLightState,
        )
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
        from crdesigner.map_conversion.sumo_map import cr2sumo_dimension_compat as compat

        original_method = CR2SumoMapConverter._create_traffic_lights
        original_flag = getattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, None)

        Connection = namedtuple("Connection", ["from_edge", "to_edge"])

        class FakeIncomingElement:
            def __init__(self):
                self.successors_straight = {3}
                self.successors_left = set()
                self.successors_right = {2}

        class FakeIntersection:
            def __init__(self):
                self.map_incoming_lanelets = {1: FakeIncomingElement()}

        class FakeLanelet:
            def __init__(self):
                self.lanelet_id = 1
                self.traffic_lights = {100}
                self.successor = [2, 3]

        class FakeEdge:
            def __init__(self, edge_id):
                self.id = edge_id
                self.outgoing = []
                self.incoming = []

        lanelet = FakeLanelet()
        edge_in = FakeEdge(10)
        edge_right = FakeEdge(20)
        edge_removed_straight = FakeEdge(30)

        traffic_light = TrafficLight(
            100,
            np.array([0.0, 0.0]),
            TrafficLightCycle([TrafficLightCycleElement(TrafficLightState.GREEN, 5)], 1),
            active=True,
            direction=TrafficLightDirection.STRAIGHT,
        )

        captured = {}

        def fake_original_create_traffic_lights(self):
            current_ids = set(self._lanelet_network.lanelets[0].traffic_lights)
            captured["ids"] = current_ids
            captured["directions"] = {
                traffic_light_id: self._lanelet_network._traffic_lights[traffic_light_id].direction
                for traffic_light_id in current_ids
            }
            return "ok"

        try:
            compat._TL_PATCHED_ONCE = False
            if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)

            CR2SumoMapConverter._create_traffic_lights = fake_original_create_traffic_lights
            with patch.object(compat, "_needs_traffic_light_patch", return_value=True):
                self.assertTrue(apply_commonroad_sumo_traffic_light_patch())

            fake_converter = SimpleNamespace(
                _lanelet_network=SimpleNamespace(
                    lanelets=[lanelet],
                    map_inc_lanelets_to_intersections={1: FakeIntersection()},
                    _traffic_lights={100: traffic_light},
                ),
                new_edges={10: edge_in, 20: edge_right},
                edges={10: edge_in, 20: edge_right, 30: edge_removed_straight},
                lanelet_id2edge_id={1: 10, 2: 20, 3: 30},
                _new_connections={Connection(edge_in, edge_right)},
            )

            result = CR2SumoMapConverter._create_traffic_lights(fake_converter)
            self.assertEqual("ok", result)
            self.assertEqual({100}, lanelet.traffic_lights)
            self.assertEqual(TrafficLightDirection.STRAIGHT, traffic_light.direction)
            self.assertEqual(1, len(captured["ids"]))
            captured_light_id = next(iter(captured["ids"]))
            self.assertNotEqual(100, captured_light_id)
            self.assertEqual(
                TrafficLightDirection.RIGHT, captured["directions"][captured_light_id]
            )
        finally:
            CR2SumoMapConverter._create_traffic_lights = original_method
            compat._TL_PATCHED_ONCE = False
            if original_flag is None:
                if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                    delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)
            else:
                setattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, original_flag)

    def test_apply_traffic_light_patch_is_idempotent(self):
        apply_commonroad_sumo_traffic_light_patch()
        self.assertFalse(apply_commonroad_sumo_traffic_light_patch())

    def test_apply_lane_grouping_patch_is_idempotent(self):
        apply_commonroad_sumo_lane_grouping_patch()
        self.assertFalse(apply_commonroad_sumo_lane_grouping_patch())

    def test_split_lanelet_ids_by_compatibility_splits_mixed_types(self):
        lanelets = [
            self._FakeLanelet(659, lanelet_type={"urban"}, user_one_way={"bicycle", "car"}),
            self._FakeLanelet(661, lanelet_type={"urban"}, user_one_way={"bicycle", "car"}),
            self._FakeLanelet(725, lanelet_type={"urban"}, user_one_way={"bicycle", "car"}),
            self._FakeLanelet(723, lanelet_type={"urban"}, user_one_way={"bicycle", "car"}),
            self._FakeLanelet(975, lanelet_type={"bicycleLane"}, user_one_way={"bicycle"}),
        ]
        lanelet_network = self._FakeLaneletNetwork(lanelets)

        segments = _split_lanelet_ids_by_compatibility(
            lanelet_network, [659, 661, 725, 723, 975]
        )
        self.assertEqual([[659, 661, 725, 723], [975]], segments)

    def test_split_lanelet_ids_by_compatibility_keeps_same_signature(self):
        lanelets = [
            self._FakeLanelet(1, lanelet_type={"urban"}, user_one_way={"car"}),
            self._FakeLanelet(2, lanelet_type={"urban"}, user_one_way={"car"}),
            self._FakeLanelet(3, lanelet_type={"urban"}, user_one_way={"car"}),
        ]
        lanelet_network = self._FakeLaneletNetwork(lanelets)

        segments = _split_lanelet_ids_by_compatibility(lanelet_network, [1, 2, 3])
        self.assertEqual([[1, 2, 3]], segments)

    def test_split_lanelet_ids_by_compatibility_only_splits_contiguous_parts(self):
        lanelets = [
            self._FakeLanelet(1, lanelet_type={"urban"}, user_one_way={"car"}),
            self._FakeLanelet(2, lanelet_type={"bicycleLane"}, user_one_way={"bicycle"}),
            self._FakeLanelet(3, lanelet_type={"urban"}, user_one_way={"car"}),
        ]
        lanelet_network = self._FakeLaneletNetwork(lanelets)

        segments = _split_lanelet_ids_by_compatibility(lanelet_network, [1, 2, 3])
        self.assertEqual([[1], [2], [3]], segments)

    def test_find_unique_upstream_replacement_edge_id_unique(self):
        e1 = self._FakeEdge(1)
        e2 = self._FakeEdge(2)
        e2.incoming = [e1]

        converter = self._FakeConverter(edges={1: e1, 2: e2}, new_edges={1: e1})
        self.assertEqual(1, _find_unique_upstream_replacement_edge_id(converter, 2))

    def test_find_unique_upstream_replacement_edge_id_via_removed_chain(self):
        e1 = self._FakeEdge(1)
        e2 = self._FakeEdge(2)
        e3 = self._FakeEdge(3)
        e2.incoming = [e1]
        e3.incoming = [e2]

        converter = self._FakeConverter(edges={1: e1, 2: e2, 3: e3}, new_edges={1: e1})
        self.assertEqual(1, _find_unique_upstream_replacement_edge_id(converter, 3))

    def test_find_unique_upstream_replacement_edge_id_ambiguous(self):
        e1 = self._FakeEdge(1)
        e2 = self._FakeEdge(2)
        e3 = self._FakeEdge(3)
        e4 = self._FakeEdge(4)
        e4.incoming = [e1, e3]

        converter = self._FakeConverter(edges={1: e1, 2: e2, 3: e3, 4: e4}, new_edges={1: e1, 3: e3})
        self.assertIsNone(_find_unique_upstream_replacement_edge_id(converter, 4))

    def test_find_unique_upstream_replacement_edge_id_no_upstream(self):
        e2 = self._FakeEdge(2)
        converter = self._FakeConverter(edges={2: e2}, new_edges={})
        self.assertIsNone(_find_unique_upstream_replacement_edge_id(converter, 2))
