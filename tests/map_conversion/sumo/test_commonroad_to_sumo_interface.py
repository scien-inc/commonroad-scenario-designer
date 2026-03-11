import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
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

    def test_filter_commonroad_traffic_lights_from_crsumo_log_disappeared_only(self):
        log_text = "\n".join(
            [
                'INFO - Traffic-light classification record: {"category":"normal","lanelet_id":1,"traffic_light_ids":[10]}',
                'WARNING - Traffic-light classification record: {"category":"skipped_ambiguous","lanelet_id":2,"traffic_light_ids":[11]}',
                'WARNING - Traffic-light classification record: {"category":"skipped_removed_no_unique_upstream","lanelet_id":3,"traffic_light_ids":[12]}',
            ]
        )

        fake_lanelet_network = MagicMock()
        fake_lanelet_network.traffic_lights = [
            MagicMock(traffic_light_id=10),
            MagicMock(traffic_light_id=11),
            MagicMock(traffic_light_id=12),
            MagicMock(traffic_light_id=13),
        ]
        fake_scenario = MagicMock(lanelet_network=fake_lanelet_network)
        fake_pp = MagicMock()
        reader_instance = MagicMock()
        reader_instance.open.return_value = (fake_scenario, fake_pp)
        writer_instance = MagicMock()

        with TemporaryDirectory() as tmp_dir, patch.object(
            map_conversion_interface, "CRDesignerFileReader", return_value=reader_instance
        ), patch.object(
            map_conversion_interface, "CRDesignerFileWriter", return_value=writer_instance
        ):
            log_path = Path(tmp_dir) / "conversion.log"
            report_path = Path(tmp_dir) / "report.json"
            log_path.write_text(log_text, encoding="utf-8")

            map_conversion_interface.filter_commonroad_traffic_lights_from_crsumo_log(
                "in.xml",
                "/tmp/out/disappeared_only.xml",
                log_path,
                include_partially_lost=False,
                report_file=report_path,
            )

            self.assertTrue(report_path.exists())

        fake_lanelet_network.remove_traffic_light.assert_any_call(10)
        fake_lanelet_network.remove_traffic_light.assert_any_call(13)
        removed_ids = {call.args[0] for call in fake_lanelet_network.remove_traffic_light.call_args_list}
        self.assertNotIn(11, removed_ids)
        self.assertNotIn(12, removed_ids)
        fake_lanelet_network.cleanup_traffic_light_references.assert_called_once()
        writer_instance.write_to_file.assert_called_once()

    def test_filter_commonroad_traffic_lights_from_crsumo_log_can_include_partially_lost(self):
        log_text = "\n".join(
            [
                'INFO - Traffic-light classification record: {"category":"normal","lanelet_id":1,"traffic_light_ids":[10]}',
                'WARNING - Traffic-light classification record: {"category":"skipped_ambiguous","lanelet_id":2,"traffic_light_ids":[10,11]}',
            ]
        )

        fake_lanelet_network = MagicMock()
        fake_lanelet_network.traffic_lights = [
            MagicMock(traffic_light_id=10),
            MagicMock(traffic_light_id=11),
            MagicMock(traffic_light_id=12),
        ]
        fake_scenario = MagicMock(lanelet_network=fake_lanelet_network)
        reader_instance = MagicMock()
        reader_instance.open.return_value = (fake_scenario, MagicMock())
        writer_instance = MagicMock()

        with TemporaryDirectory() as tmp_dir, patch.object(
            map_conversion_interface, "CRDesignerFileReader", return_value=reader_instance
        ), patch.object(
            map_conversion_interface, "CRDesignerFileWriter", return_value=writer_instance
        ):
            log_path = Path(tmp_dir) / "conversion.log"
            log_path.write_text(log_text, encoding="utf-8")

            map_conversion_interface.filter_commonroad_traffic_lights_from_crsumo_log(
                "in.xml",
                "/tmp/out/problematic.xml",
                log_path,
                include_partially_lost=True,
            )

        fake_lanelet_network.remove_traffic_light.assert_called_once_with(12)
        fake_lanelet_network.cleanup_traffic_light_references.assert_called_once()
