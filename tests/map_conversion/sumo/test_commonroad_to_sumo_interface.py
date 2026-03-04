import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crdesigner.map_conversion import map_conversion_interface


class TestCommonRoadToSumoInterface(unittest.TestCase):
    def test_preserve_mode_success(self):
        converter_instance = MagicMock()
        converter_cls = MagicMock()
        converter_cls.from_file.return_value = converter_instance

        with patch.object(
            map_conversion_interface, "apply_commonroad_sumo_nd_patch"
        ) as nd_patch_fn, patch.object(
            map_conversion_interface, "apply_commonroad_sumo_traffic_light_patch"
        ) as tl_patch_fn, patch.object(
            map_conversion_interface, "CR2SumoMapConverter", converter_cls
        ), patch.object(
            map_conversion_interface, "CRDesignerFileReader"
        ) as reader_cls:
            map_conversion_interface.commonroad_to_sumo(
                "in.xml",
                "/tmp/out/sumo/_placeholder.net.xml",
                z_mode="preserve",
                fallback_2d=True,
            )

        nd_patch_fn.assert_called_once()
        tl_patch_fn.assert_called_once()
        converter_cls.from_file.assert_called_once_with("in.xml")
        converter_instance.create_sumo_files.assert_called_once_with(Path("/tmp/out/sumo"))
        reader_cls.assert_not_called()

    def test_preserve_mode_falls_back_to_2d(self):
        converter_preserve = MagicMock()
        converter_preserve.create_sumo_files.side_effect = RuntimeError("boom")
        converter_fallback = MagicMock()

        converter_cls = MagicMock()
        converter_cls.from_file.return_value = converter_preserve
        converter_cls.return_value = converter_fallback

        scenario = MagicMock()
        scenario.scenario_id.map_name = "my_map"
        reader_instance = MagicMock()
        reader_instance.open.return_value = (scenario, None)

        with patch.object(
            map_conversion_interface, "apply_commonroad_sumo_nd_patch"
        ) as nd_patch_fn, patch.object(
            map_conversion_interface, "apply_commonroad_sumo_traffic_light_patch"
        ) as tl_patch_fn, patch.object(
            map_conversion_interface, "CR2SumoMapConverter", converter_cls
        ), patch.object(
            map_conversion_interface, "CRDesignerFileReader", return_value=reader_instance
        ):
            map_conversion_interface.commonroad_to_sumo(
                "in.xml",
                "/tmp/out/sumo/_placeholder.net.xml",
                z_mode="preserve",
                fallback_2d=True,
            )

        nd_patch_fn.assert_called_once()
        tl_patch_fn.assert_called_once()
        converter_cls.from_file.assert_called_once_with("in.xml")
        scenario.convert_to_2d.assert_called_once_with(map_name="my_map")
        converter_cls.assert_called_with(scenario)
        converter_fallback.create_sumo_files.assert_called_once_with(Path("/tmp/out/sumo"))

    def test_preserve_mode_raises_without_fallback(self):
        converter_preserve = MagicMock()
        converter_preserve.create_sumo_files.side_effect = RuntimeError("boom")
        converter_cls = MagicMock()
        converter_cls.from_file.return_value = converter_preserve

        with patch.object(map_conversion_interface, "apply_commonroad_sumo_nd_patch"), patch.object(
            map_conversion_interface, "apply_commonroad_sumo_traffic_light_patch"
        ), patch.object(map_conversion_interface, "CR2SumoMapConverter", converter_cls), patch.object(
            map_conversion_interface, "CRDesignerFileReader"
        ) as reader_cls:
            with self.assertRaises(RuntimeError):
                map_conversion_interface.commonroad_to_sumo(
                    "in.xml",
                    "/tmp/out/sumo/_placeholder.net.xml",
                    z_mode="preserve",
                    fallback_2d=False,
                )

        reader_cls.assert_not_called()

    def test_force_2d_mode(self):
        scenario = MagicMock()
        scenario.scenario_id.map_name = "my_map"
        reader_instance = MagicMock()
        reader_instance.open.return_value = (scenario, None)

        converter_instance = MagicMock()
        converter_cls = MagicMock(return_value=converter_instance)

        with patch.object(
            map_conversion_interface, "apply_commonroad_sumo_nd_patch"
        ) as nd_patch_fn, patch.object(
            map_conversion_interface, "apply_commonroad_sumo_traffic_light_patch"
        ) as tl_patch_fn, patch.object(
            map_conversion_interface, "CRDesignerFileReader", return_value=reader_instance
        ), patch.object(
            map_conversion_interface, "CR2SumoMapConverter", converter_cls
        ):
            map_conversion_interface.commonroad_to_sumo(
                "in.xml",
                "/tmp/out/sumo/_placeholder.net.xml",
                z_mode="force-2d",
                fallback_2d=True,
            )

        nd_patch_fn.assert_not_called()
        tl_patch_fn.assert_called_once()
        converter_cls.from_file.assert_not_called()
        scenario.convert_to_2d.assert_called_once_with(map_name="my_map")
        converter_instance.create_sumo_files.assert_called_once_with(Path("/tmp/out/sumo"))

    def test_invalid_z_mode(self):
        with self.assertRaises(ValueError):
            map_conversion_interface.commonroad_to_sumo(
                "in.xml",
                "/tmp/out/sumo/_placeholder.net.xml",
                z_mode="invalid",
                fallback_2d=True,
            )
