import importlib.util
import unittest

import numpy as np

from crdesigner.map_conversion.sumo_map.cr2sumo_dimension_compat import (
    _find_unique_upstream_replacement_edge_id,
    apply_commonroad_sumo_nd_patch,
    apply_commonroad_sumo_traffic_light_patch,
)


@unittest.skipUnless(
    importlib.util.find_spec("commonroad_sumo") is not None
    and importlib.util.find_spec("commonroad") is not None,
    "commonroad/commonroad_sumo not installed",
)
class TestCR2SumoDimensionCompat(unittest.TestCase):
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
