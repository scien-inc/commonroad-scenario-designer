import tempfile
import unittest
from pathlib import Path

from crdesigner.map_conversion.sumo_map.traffic_light_log_tools import (
    load_traffic_light_classification_records_from_log,
    summarize_traffic_light_classification_records,
)


class TestTrafficLightLogTools(unittest.TestCase):
    def test_load_records_from_log(self):
        log_text = "\n".join(
            [
                "prefix",
                'INFO - Traffic-light classification record: {"category":"normal","lanelet_id":1,"traffic_light_ids":[10,11]}',
                'WARNING - Traffic-light classification record: {"category":"skipped_ambiguous","lanelet_id":2,"traffic_light_ids":[12]}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "conversion.log"
            log_path.write_text(log_text, encoding="utf-8")

            records = load_traffic_light_classification_records_from_log(log_path)

        self.assertEqual(2, len(records))
        self.assertEqual("normal", records[0]["category"])
        self.assertEqual([12], records[1]["traffic_light_ids"])

    def test_load_records_requires_detailed_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "conversion.log"
            log_path.write_text("no records here\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_traffic_light_classification_records_from_log(log_path)

    def test_summarize_records_separates_disappeared_partial_and_retained(self):
        records = [
            {"category": "normal", "lanelet_id": 1, "traffic_light_ids": [10, 11]},
            {"category": "remapped", "lanelet_id": 2, "traffic_light_ids": [11]},
            {"category": "skipped_ambiguous", "lanelet_id": 3, "traffic_light_ids": [12, 13]},
            {
                "category": "skipped_removed_no_unique_upstream",
                "lanelet_id": 4,
                "traffic_light_ids": [13],
            },
            {"category": "skipped_ambiguous", "lanelet_id": 5, "traffic_light_ids": [11]},
        ]

        summary = summarize_traffic_light_classification_records(records)

        self.assertEqual(5, summary["record_count"])
        self.assertEqual([12, 13], summary["traffic_light_ids"]["disappeared_only"])
        self.assertEqual([11], summary["traffic_light_ids"]["partially_lost"])
        self.assertEqual([10], summary["traffic_light_ids"]["retained_only"])
        self.assertEqual(
            {"normal", "remapped", "skipped_ambiguous"},
            set(summary["categories_by_traffic_light_id"]["11"]),
        )
        self.assertEqual([3], summary["lanelet_ids_by_traffic_light_id"]["13"]["skipped_ambiguous"])
        self.assertEqual(
            [4],
            summary["lanelet_ids_by_traffic_light_id"]["13"][
                "skipped_removed_no_unique_upstream"
            ],
        )
