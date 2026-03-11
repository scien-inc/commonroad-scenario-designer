import importlib.util
import unittest

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
