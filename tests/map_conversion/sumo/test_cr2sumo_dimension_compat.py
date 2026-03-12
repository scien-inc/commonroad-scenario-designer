import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from crdesigner.map_conversion.sumo_map.cr2sumo_dimension_compat import (
    _find_unique_upstream_replacement_edge_id,
    _split_lanelet_ids_by_compatibility,
    apply_commonroad_sumo_lane_grouping_patch,
    apply_commonroad_sumo_netconvert_tls_patch,
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

            class FakeTrafficLightSignals:
                def add_program(self, program):
                    pass

                def add_connection(self, connection):
                    pass

            fake_converter = SimpleNamespace(
                _lanelet_network=SimpleNamespace(
                    lanelets=[],
                    map_inc_lanelets_to_intersections={},
                    _traffic_lights={},
                ),
                new_edges={},
                lanelet_id2edge_id={},
                edges={},
                _scenario=SimpleNamespace(dt=1.0),
                traffic_light_signals=FakeTrafficLightSignals(),
            )

            with self.assertLogs(
                "crdesigner.map_conversion.sumo_map.cr2sumo_dimension_compat", level="INFO"
            ) as logs:
                result = CR2SumoMapConverter._create_traffic_lights(fake_converter)

            self.assertIsNone(result)
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

    def test_traffic_light_patch_groups_same_incoming_connections_into_one_green_phase(self):
        from commonroad.scenario.traffic_light import (
            TrafficLight,
            TrafficLightCycle,
            TrafficLightCycleElement,
            TrafficLightDirection,
            TrafficLightState,
        )
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
        from commonroad_sumo.cr2sumo.map_converter.traffic_light import SignalState
        from crdesigner.map_conversion.sumo_map import cr2sumo_dimension_compat as compat

        original_method = CR2SumoMapConverter._create_traffic_lights
        original_flag = getattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, None)
        original_encoder = cr2sumo_map_converter.TrafficLightEncoder

        class FakeIncomingElement:
            def __init__(self, incoming_id, incoming_lanelets, straight=None, left=None, right=None):
                self.incoming_id = incoming_id
                self.incoming_lanelets = set(incoming_lanelets)
                self.successors_straight = set(straight or [])
                self.successors_left = set(left or [])
                self.successors_right = set(right or [])

        class FakeIntersection:
            def __init__(self, incoming_elements):
                self.map_incoming_lanelets = {}
                for incoming in incoming_elements:
                    for lanelet_id in incoming.incoming_lanelets:
                        self.map_incoming_lanelets[lanelet_id] = incoming

        class FakeLanelet:
            def __init__(self, lanelet_id, center_vertices, successor, traffic_lights):
                self.lanelet_id = lanelet_id
                self.center_vertices = np.array(center_vertices, dtype=float)
                self.traffic_lights = set(traffic_lights)
                self.successor = list(successor)

        class FakeEdge:
            def __init__(self, edge_id, to_node=None):
                self.id = edge_id
                self.outgoing = []
                self.incoming = []
                self.to_node = to_node

        class FakeConnection:
            def __init__(self, from_edge, to_edge, shape):
                self.from_edge = from_edge
                self.to_edge = to_edge
                self.shape = np.array(shape, dtype=float)
                self.tls = None
                self.tl_link = None

            def __hash__(self):
                return hash(
                    (
                        self.from_edge.id,
                        self.to_edge.id,
                        tuple(map(tuple, self.shape.tolist())),
                    )
                )

            def __eq__(self, other):
                return (
                    isinstance(other, FakeConnection)
                    and self.from_edge.id == other.from_edge.id
                    and self.to_edge.id == other.to_edge.id
                    and np.array_equal(self.shape, other.shape)
                )

        class FakeTrafficLightSignals:
            def __init__(self):
                self.programs = []
                self.connections = []

            def add_program(self, program):
                self.programs.append(program)

            def add_connection(self, connection):
                self.connections.append(connection)

        node = SimpleNamespace(id=5001, type=None)
        south_lanelet = FakeLanelet(1, [[0.0, -10.0], [0.0, 0.0]], [11, 12], {100})
        north_lanelet = FakeLanelet(2, [[0.0, 10.0], [0.0, 0.0]], [13], {101})
        successor_straight = FakeLanelet(11, [[0.0, 0.0], [0.0, 10.0]], [], set())
        successor_left = FakeLanelet(12, [[0.0, 0.0], [10.0, 0.0]], [], set())
        successor_opposite = FakeLanelet(13, [[0.0, 0.0], [0.0, -10.0]], [], set())

        south_incoming = FakeIncomingElement(601, {1}, straight={11}, left={12})
        north_incoming = FakeIncomingElement(602, {2}, straight={13})
        intersection = FakeIntersection([south_incoming, north_incoming])

        edge_south = FakeEdge(10, to_node=node)
        edge_north = FakeEdge(20, to_node=node)
        edge_straight = FakeEdge(30, to_node=node)
        edge_left = FakeEdge(31, to_node=node)
        edge_opposite = FakeEdge(32, to_node=node)

        straight_connection = FakeConnection(edge_south, edge_straight, [[0.0, -1.0], [0.0, 1.0]])
        left_connection = FakeConnection(edge_south, edge_left, [[0.0, -1.0], [0.0, 0.0], [1.0, 0.0]])
        opposite_connection = FakeConnection(edge_north, edge_opposite, [[0.0, 1.0], [0.0, -1.0]])

        traffic_light = TrafficLight(
            100,
            np.array([0.0, 0.0]),
            TrafficLightCycle([TrafficLightCycleElement(TrafficLightState.GREEN, 5)], 1),
            active=True,
            direction=TrafficLightDirection.ALL,
        )
        opposite_traffic_light = TrafficLight(
            101,
            np.array([0.0, 0.0]),
            TrafficLightCycle([TrafficLightCycleElement(TrafficLightState.GREEN, 5)], 1),
            active=True,
            direction=TrafficLightDirection.ALL,
        )

        try:
            compat._TL_PATCHED_ONCE = False
            if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)

            CR2SumoMapConverter._create_traffic_lights = lambda self: None
            self.assertTrue(apply_commonroad_sumo_traffic_light_patch())

            fake_converter = SimpleNamespace(
                _lanelet_network=SimpleNamespace(
                    lanelets=[south_lanelet, north_lanelet],
                    map_inc_lanelets_to_intersections={1: intersection, 2: intersection},
                    _traffic_lights={100: traffic_light, 101: opposite_traffic_light},
                    find_lanelet_by_id=lambda lanelet_id: {
                        1: south_lanelet,
                        2: north_lanelet,
                        11: successor_straight,
                        12: successor_left,
                        13: successor_opposite,
                    }.get(lanelet_id),
                ),
                new_edges={
                    10: edge_south,
                    20: edge_north,
                    30: edge_straight,
                    31: edge_left,
                    32: edge_opposite,
                },
                edges={
                    10: edge_south,
                    20: edge_north,
                    30: edge_straight,
                    31: edge_left,
                    32: edge_opposite,
                },
                lanelet_id2edge_id={1: 10, 2: 20, 11: 30, 12: 31, 13: 32},
                _new_connections={
                    straight_connection,
                    left_connection,
                    opposite_connection,
                },
                _scenario=SimpleNamespace(dt=1.0),
                traffic_light_signals=FakeTrafficLightSignals(),
            )

            CR2SumoMapConverter._create_traffic_lights(fake_converter)

            self.assertEqual(1, len(fake_converter.traffic_light_signals.programs))
            program = fake_converter.traffic_light_signals.programs[0]
            self.assertEqual(3, len(program.phases))

            first_phase = program.phases[0].state
            self.assertIn(first_phase[straight_connection.tl_link], {SignalState.GREEN, SignalState.GREEN_PRIORITY})
            self.assertIn(first_phase[left_connection.tl_link], {SignalState.GREEN, SignalState.GREEN_PRIORITY})
            self.assertIn(first_phase[opposite_connection.tl_link], {SignalState.GREEN, SignalState.GREEN_PRIORITY})
        finally:
            CR2SumoMapConverter._create_traffic_lights = original_method
            cr2sumo_map_converter.TrafficLightEncoder = original_encoder
            compat._TL_PATCHED_ONCE = False
            if original_flag is None:
                if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                    delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)
            else:
                setattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, original_flag)

    def test_traffic_light_patch_pairs_opposite_incomings_into_two_green_phases(self):
        from commonroad.scenario.traffic_light import (
            TrafficLight,
            TrafficLightCycle,
            TrafficLightCycleElement,
            TrafficLightDirection,
            TrafficLightState,
        )
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
        from commonroad_sumo.cr2sumo.map_converter.traffic_light import SignalState
        from crdesigner.map_conversion.sumo_map import cr2sumo_dimension_compat as compat

        original_method = CR2SumoMapConverter._create_traffic_lights
        original_flag = getattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG, None)

        class FakeIncomingElement:
            def __init__(self, incoming_id, lanelet_id, successor_id):
                self.incoming_id = incoming_id
                self.incoming_lanelets = {lanelet_id}
                self.successors_straight = {successor_id}
                self.successors_left = set()
                self.successors_right = set()

        class FakeIntersection:
            def __init__(self, incoming_elements):
                self.map_incoming_lanelets = {}
                for incoming in incoming_elements:
                    for lanelet_id in incoming.incoming_lanelets:
                        self.map_incoming_lanelets[lanelet_id] = incoming

        class FakeLanelet:
            def __init__(self, lanelet_id, center_vertices, successor):
                self.lanelet_id = lanelet_id
                self.center_vertices = np.array(center_vertices, dtype=float)
                self.traffic_lights = {100}
                self.successor = [successor]

        class FakeEdge:
            def __init__(self, edge_id, to_node=None):
                self.id = edge_id
                self.outgoing = []
                self.incoming = []
                self.to_node = to_node

        class FakeConnection:
            def __init__(self, from_edge, to_edge, shape):
                self.from_edge = from_edge
                self.to_edge = to_edge
                self.shape = np.array(shape, dtype=float)
                self.tls = None
                self.tl_link = None

            def __hash__(self):
                return hash(
                    (
                        self.from_edge.id,
                        self.to_edge.id,
                        tuple(map(tuple, self.shape.tolist())),
                    )
                )

            def __eq__(self, other):
                return (
                    isinstance(other, FakeConnection)
                    and self.from_edge.id == other.from_edge.id
                    and self.to_edge.id == other.to_edge.id
                    and np.array_equal(self.shape, other.shape)
                )

        class FakeTrafficLightSignals:
            def __init__(self):
                self.programs = []
                self.connections = []

            def add_program(self, program):
                self.programs.append(program)

            def add_connection(self, connection):
                self.connections.append(connection)

        node = SimpleNamespace(id=5002, type=None)
        south_lanelet = FakeLanelet(1, [[0.0, -10.0], [0.0, 0.0]], 11)
        north_lanelet = FakeLanelet(2, [[0.0, 10.0], [0.0, 0.0]], 12)
        west_lanelet = FakeLanelet(3, [[-10.0, 0.0], [0.0, 0.0]], 13)
        east_lanelet = FakeLanelet(4, [[10.0, 0.0], [0.0, 0.0]], 14)
        successors = {
            11: FakeLanelet(11, [[0.0, 0.0], [0.0, 10.0]], None),
            12: FakeLanelet(12, [[0.0, 0.0], [0.0, -10.0]], None),
            13: FakeLanelet(13, [[0.0, 0.0], [10.0, 0.0]], None),
            14: FakeLanelet(14, [[0.0, 0.0], [-10.0, 0.0]], None),
        }
        lanelets = {
            1: south_lanelet,
            2: north_lanelet,
            3: west_lanelet,
            4: east_lanelet,
            **successors,
        }

        incomings = [
            FakeIncomingElement(701, 1, 11),
            FakeIncomingElement(702, 2, 12),
            FakeIncomingElement(703, 3, 13),
            FakeIncomingElement(704, 4, 14),
        ]
        intersection = FakeIntersection(incomings)

        edges = {
            10: FakeEdge(10, to_node=node),
            20: FakeEdge(20, to_node=node),
            30: FakeEdge(30, to_node=node),
            40: FakeEdge(40, to_node=node),
            110: FakeEdge(110, to_node=node),
            120: FakeEdge(120, to_node=node),
            130: FakeEdge(130, to_node=node),
            140: FakeEdge(140, to_node=node),
        }
        connections = {
            FakeConnection(edges[10], edges[110], [[0.0, -1.0], [0.0, 1.0]]),
            FakeConnection(edges[20], edges[120], [[0.0, 1.0], [0.0, -1.0]]),
            FakeConnection(edges[30], edges[130], [[-1.0, 0.0], [1.0, 0.0]]),
            FakeConnection(edges[40], edges[140], [[1.0, 0.0], [-1.0, 0.0]]),
        }
        traffic_light = TrafficLight(
            100,
            np.array([0.0, 0.0]),
            TrafficLightCycle([TrafficLightCycleElement(TrafficLightState.GREEN, 5)], 1),
            active=True,
            direction=TrafficLightDirection.ALL,
        )

        try:
            compat._TL_PATCHED_ONCE = False
            if hasattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG):
                delattr(cr2sumo_map_converter, compat._TL_PATCH_FLAG)

            CR2SumoMapConverter._create_traffic_lights = lambda self: None
            self.assertTrue(apply_commonroad_sumo_traffic_light_patch())

            fake_converter = SimpleNamespace(
                _lanelet_network=SimpleNamespace(
                    lanelets=[south_lanelet, north_lanelet, west_lanelet, east_lanelet],
                    map_inc_lanelets_to_intersections={
                        1: intersection,
                        2: intersection,
                        3: intersection,
                        4: intersection,
                    },
                    _traffic_lights={100: traffic_light},
                    find_lanelet_by_id=lambda lanelet_id: lanelets.get(lanelet_id),
                ),
                new_edges=edges,
                edges=edges,
                lanelet_id2edge_id={
                    1: 10,
                    2: 20,
                    3: 30,
                    4: 40,
                    11: 110,
                    12: 120,
                    13: 130,
                    14: 140,
                },
                _new_connections=connections,
                _scenario=SimpleNamespace(dt=1.0),
                traffic_light_signals=FakeTrafficLightSignals(),
            )

            CR2SumoMapConverter._create_traffic_lights(fake_converter)

            self.assertEqual(1, len(fake_converter.traffic_light_signals.programs))
            program = fake_converter.traffic_light_signals.programs[0]
            self.assertEqual(6, len(program.phases))

            green_phase_states = [phase.state for phase in program.phases[::3]]
            green_index_sets = []
            for state in green_phase_states:
                green_index_sets.append(
                    {
                        index
                        for index, signal in enumerate(state)
                        if signal in {SignalState.GREEN, SignalState.GREEN_PRIORITY}
                    }
                )

            connection_index_by_edge = {
                (connection.from_edge.id, connection.to_edge.id): connection.tl_link
                for connection in fake_converter.traffic_light_signals.connections
            }
            north_south = {
                connection_index_by_edge[(10, 110)],
                connection_index_by_edge[(20, 120)],
            }
            east_west = {
                connection_index_by_edge[(30, 130)],
                connection_index_by_edge[(40, 140)],
            }
            self.assertIn(north_south, green_index_sets)
            self.assertIn(east_west, green_index_sets)
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

    def test_apply_netconvert_tls_patch_is_idempotent(self):
        apply_commonroad_sumo_netconvert_tls_patch()
        self.assertFalse(apply_commonroad_sumo_netconvert_tls_patch())

    def test_apply_lane_grouping_patch_is_idempotent(self):
        apply_commonroad_sumo_lane_grouping_patch()
        self.assertFalse(apply_commonroad_sumo_lane_grouping_patch())

    def test_netconvert_tls_patch_uses_explicit_tllogic_files(self):
        from commonroad_sumo.cr2sumo.map_converter import map_converter as cr2sumo_map_converter
        from commonroad_sumo.cr2sumo.map_converter.map_converter import CR2SumoMapConverter
        from crdesigner.map_conversion.sumo_map import cr2sumo_dimension_compat as compat

        original_method = CR2SumoMapConverter.merge_intermediate_files
        original_flag = getattr(cr2sumo_map_converter, compat._NETCONVERT_TLS_PATCH_FLAG, None)
        original_execute = cr2sumo_map_converter.execute_sumo_application
        original_sumo_project = cr2sumo_map_converter.SumoProject
        original_convert_intermediate = getattr(
            cr2sumo_map_converter, "convert_intermediate_sumo_project_with_netconvert", None
        )

        captured = {}

        class FakeProjectPaths:
            def __init__(self, output_path="/tmp/out.net.xml"):
                self.output_path = output_path

            @classmethod
            def from_intermediate_sumo_project(cls, project):
                return cls()

            def get_file_path(self, file_type):
                name = getattr(file_type, "name", str(file_type))
                mapping = {
                    "NODES": "/tmp/nodes.nod.xml",
                    "EDGES": "/tmp/edges.edg.xml",
                    "CONNECTIONS": "/tmp/connections.con.xml",
                    "TLLOGICS": "/tmp/tllogic.tll.xml",
                    "TYPES": "/tmp/types.typ.xml",
                    "NET": self.output_path,
                }
                return mapping[name]

        class FakeIntermediateProject:
            def __init__(self):
                self.cleaned = False

            def get_file_path(self, file_type):
                name = getattr(file_type, "name", str(file_type))
                mapping = {
                    "NODES": "/tmp/nodes.nod.xml",
                    "EDGES": "/tmp/edges.edg.xml",
                    "CONNECTIONS": "/tmp/connections.con.xml",
                    "TLLOGICS": "/tmp/tllogic.tll.xml",
                    "TYPES": "/tmp/types.typ.xml",
                }
                return mapping[name]

            def cleanup(self):
                self.cleaned = True

        def fake_execute(_app, args):
            captured["args"] = list(args)
            return ""

        try:
            compat._NETCONVERT_TLS_PATCHED_ONCE = False
            if hasattr(cr2sumo_map_converter, compat._NETCONVERT_TLS_PATCH_FLAG):
                delattr(cr2sumo_map_converter, compat._NETCONVERT_TLS_PATCH_FLAG)

            CR2SumoMapConverter.merge_intermediate_files = lambda self, project, cleanup: None
            with patch.object(compat, "_needs_netconvert_tls_patch", return_value=True):
                self.assertTrue(apply_commonroad_sumo_netconvert_tls_patch())

            cr2sumo_map_converter.execute_sumo_application = fake_execute
            cr2sumo_map_converter.SumoProject = FakeProjectPaths

            fake_converter = SimpleNamespace(
                _conf=SimpleNamespace(random_seed=7),
                _output_file="/tmp/out.net.xml",
            )
            fake_project = FakeIntermediateProject()

            result = CR2SumoMapConverter.merge_intermediate_files(
                fake_converter, fake_project, cleanup=False
            )

            self.assertIsNotNone(result)
            self.assertIn("--tls.guess-signals=false", captured["args"])
            self.assertIn("--tls.group-signals=false", captured["args"])
            self.assertIn("--tllogic-files=/tmp/tllogic.tll.xml", captured["args"])
            self.assertIn("--type-files=/tmp/types.typ.xml", captured["args"])
            self.assertIn("--seed=7", captured["args"])
            self.assertNotIn("--tls.guess-signals=true", captured["args"])
            self.assertNotIn("--tls.group-signals=true", captured["args"])
            self.assertFalse(
                any(arg.startswith("--tls.green.time=") for arg in captured["args"])
            )
            self.assertFalse(any(arg.startswith("--tls.red.time=") for arg in captured["args"]))
            self.assertFalse(
                any(arg.startswith("--tls.yellow.time=") for arg in captured["args"])
            )
            self.assertFalse(
                any(arg.startswith("--tls.allred.time=") for arg in captured["args"])
            )
            self.assertFalse(
                any(arg.startswith("--tls.left-green.time=") for arg in captured["args"])
            )
        finally:
            CR2SumoMapConverter.merge_intermediate_files = original_method
            cr2sumo_map_converter.execute_sumo_application = original_execute
            cr2sumo_map_converter.SumoProject = original_sumo_project
            if original_convert_intermediate is not None:
                cr2sumo_map_converter.convert_intermediate_sumo_project_with_netconvert = (
                    original_convert_intermediate
                )
            compat._NETCONVERT_TLS_PATCHED_ONCE = False
            if original_flag is None:
                if hasattr(cr2sumo_map_converter, compat._NETCONVERT_TLS_PATCH_FLAG):
                    delattr(cr2sumo_map_converter, compat._NETCONVERT_TLS_PATCH_FLAG)
            else:
                setattr(
                    cr2sumo_map_converter,
                    compat._NETCONVERT_TLS_PATCH_FLAG,
                    original_flag,
                )

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
