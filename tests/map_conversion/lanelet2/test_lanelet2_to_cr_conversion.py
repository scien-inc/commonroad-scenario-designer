import os
import tempfile
import time
import unittest
from textwrap import dedent

from commonroad.planning.planning_problem import PlanningProblemSet  # type: ignore
from commonroad.scenario.scenario import Scenario, Tag  # type: ignore
from commonroad.scenario.traffic_light import TrafficLightDirection
from lxml import etree  # type: ignore

from crdesigner.common.config.general_config import general_config
from crdesigner.common.config.gui_config import utm_default
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.file_writer import CRDesignerFileWriter, OverwriteExistingFile
from crdesigner.map_conversion.common.utils import generate_unique_id
from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter
from crdesigner.map_conversion.lanelet2.lanelet2_parser import Lanelet2Parser
from crdesigner.map_conversion.lanelet2.lanelet2cr import Lanelet2CRConverter
from tests.map_conversion.utils import elements_equal


def get_tmp_dir():
    return os.path.dirname(os.path.abspath(__file__)) + "/.pytest_cache" + "/"


def _signalized_intersection_osm_xml() -> str:
    return dedent(
        """
        <osm version="0.6">
          <node id="1" lat="49.000009" lon="8.399720" />
          <node id="2" lat="49.000009" lon="8.400000" />
          <node id="3" lat="48.999991" lon="8.399720" />
          <node id="4" lat="48.999991" lon="8.400000" />
          <node id="5" lat="49.000009" lon="8.400280" />
          <node id="6" lat="48.999991" lon="8.400280" />
          <node id="7" lat="48.999720" lon="8.399991" />
          <node id="8" lat="49.000000" lon="8.399991" />
          <node id="9" lat="48.999720" lon="8.400009" />
          <node id="10" lat="49.000000" lon="8.400009" />
          <node id="11" lat="49.000280" lon="8.399991" />
          <node id="12" lat="49.000280" lon="8.400009" />

          <node id="21" lat="48.999995" lon="8.399985" />
          <node id="22" lat="49.000000" lon="8.399985" />
          <node id="23" lat="49.000005" lon="8.399985" />
          <node id="24" lat="49.000015" lon="8.399995" />
          <node id="25" lat="49.000015" lon="8.400000" />
          <node id="26" lat="49.000015" lon="8.400005" />

          <node id="31" lat="48.999991" lon="8.399980" />
          <node id="32" lat="49.000009" lon="8.399980" />
          <node id="33" lat="48.999985" lon="8.399991" />
          <node id="34" lat="48.999985" lon="8.400009" />

          <way id="1001"><nd ref="1"/><nd ref="2"/></way>
          <way id="1002"><nd ref="3"/><nd ref="4"/></way>
          <way id="1003"><nd ref="2"/><nd ref="5"/></way>
          <way id="1004"><nd ref="4"/><nd ref="6"/></way>
          <way id="1005"><nd ref="7"/><nd ref="8"/></way>
          <way id="1006"><nd ref="9"/><nd ref="10"/></way>
          <way id="1007"><nd ref="8"/><nd ref="11"/></way>
          <way id="1008"><nd ref="10"/><nd ref="12"/></way>

          <way id="4001">
            <nd ref="21"/><nd ref="22"/><nd ref="23"/>
            <tag k="type" v="traffic_light"/>
            <tag k="subtype" v="red_green"/>
          </way>
          <way id="4002">
            <nd ref="24"/><nd ref="25"/><nd ref="26"/>
            <tag k="type" v="traffic_light"/>
            <tag k="subtype" v="red_green"/>
          </way>
          <way id="5001"><nd ref="31"/><nd ref="32"/></way>
          <way id="5002"><nd ref="33"/><nd ref="34"/></way>

          <relation id="2001">
            <member type="way" role="left" ref="1001"/>
            <member type="way" role="right" ref="1002"/>
            <member type="relation" role="regulatory_element" ref="3001"/>
            <tag k="type" v="lanelet"/>
            <tag k="subtype" v="road"/>
            <tag k="location" v="urban"/>
            <tag k="one_way" v="yes"/>
            <tag k="turn_direction" v="straight"/>
          </relation>
          <relation id="2002">
            <member type="way" role="left" ref="1003"/>
            <member type="way" role="right" ref="1004"/>
            <tag k="type" v="lanelet"/>
            <tag k="subtype" v="road"/>
            <tag k="location" v="urban"/>
            <tag k="one_way" v="yes"/>
          </relation>
          <relation id="2003">
            <member type="way" role="left" ref="1005"/>
            <member type="way" role="right" ref="1006"/>
            <member type="relation" role="regulatory_element" ref="3002"/>
            <tag k="type" v="lanelet"/>
            <tag k="subtype" v="road"/>
            <tag k="location" v="urban"/>
            <tag k="one_way" v="yes"/>
            <tag k="turn_direction" v="straight"/>
          </relation>
          <relation id="2004">
            <member type="way" role="left" ref="1007"/>
            <member type="way" role="right" ref="1008"/>
            <tag k="type" v="lanelet"/>
            <tag k="subtype" v="road"/>
            <tag k="location" v="urban"/>
            <tag k="one_way" v="yes"/>
          </relation>

          <relation id="3001">
            <member type="way" role="refers" ref="4001"/>
            <member type="way" role="ref_line" ref="5001"/>
            <tag k="type" v="regulatory_element"/>
            <tag k="subtype" v="traffic_light"/>
          </relation>
          <relation id="3002">
            <member type="way" role="refers" ref="4002"/>
            <member type="way" role="ref_line" ref="5002"/>
            <tag k="type" v="regulatory_element"/>
            <tag k="subtype" v="traffic_light"/>
          </relation>
        </osm>
        """
    ).strip()


class TestLanelet2ToCommonRoadConversion(unittest.TestCase):
    """Tests the conversion from an osm file to a CommonRoad xml file."""

    @staticmethod
    def load_and_convert(
        osm_file_name: str, translate: bool = False, proj_string: str = None, file_path: str = None
    ) -> Scenario:
        """Loads and converts osm file to a scenario

        :param osm_file_name: name of the osm file
        :param translate: Boolean indicating whether the map should be moved to the origin
        :param proj_string: string defining projection method from geo-coordinates
        :return: Scenario that corresponds to that osm file
        """
        generate_unique_id(0)  # reset ID counter for next test case
        cwd_path = os.path.dirname(os.path.abspath(__file__))
        out_path = cwd_path + "/.pytest_cache"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        else:
            for dir_path, _, filenames in os.walk(out_path):
                for file in filenames:
                    if file.endswith(".xml"):
                        os.remove(os.path.join(dir_path, file))

        if file_path is None:
            file_path = (
                os.path.dirname(os.path.realpath(__file__))
                + f"/../test_maps/lanelet2/{osm_file_name}.osm"
            )

        with open(
            file_path,
            "r",
        ) as fh:
            osm = Lanelet2Parser(etree.parse(fh).getroot()).parse()

        if proj_string is not None:
            general_config.proj_string_cr = proj_string
        if translate is not None:
            lanelet2_config.translate = translate
        osm2l = Lanelet2CRConverter()
        return osm2l(osm)

    def compare_maps(self, file_name: str, translate: bool = False, file_path: str = None) -> bool:
        """
        Test if the scenario is equal to the loaded xml file.
        Disregard the different dates.
        """
        xml_output_name = file_name
        translated = "" if not translate else "_translated"

        cr_file_path = (
            os.path.dirname(os.path.realpath(__file__))
            + f"/../test_maps/lanelet2/{xml_output_name}{translated}.xml"
        )
        with open(
            cr_file_path,
            "r",
        ) as fh:
            parser = etree.XMLParser(remove_blank_text=True)
            tree_import = etree.parse(fh, parser=parser).getroot()
            writer = CRDesignerFileWriter(
                scenario=self.load_and_convert(file_name, translate=translate, file_path=file_path),
                planning_problem_set=PlanningProblemSet(),
                author="",
                affiliation="",
                source="CommonRoad Scenario Designer",
                tags={Tag.URBAN, Tag.HIGHWAY},
            )
            writer.write_to_file(
                get_tmp_dir() + xml_output_name + translated + ".xml", OverwriteExistingFile.ALWAYS
            )

            # set same date so this won't change the comparison
            date = time.strftime("%Y-%m-%d", time.localtime())
            tree_import.set("date", date)
            writer._file_writer.root_node.set("date", date)

            # compare both element trees
            return elements_equal(tree_import, writer._file_writer.root_node)

    def test_simple_map(self):
        """Simple test case file which includes successors and predecessors and adjacencies."""
        self.assertTrue(self.compare_maps("urban-1_lanelets_utm"))

    def test_simple_map_translated(self):
        """Simple test case file which includes successors and predecessors and adjacencies."""
        self.assertTrue(self.compare_maps("urban-1_lanelets_utm", translate=True))

    def test_merging_lanelets(self):
        """Basic test file including some splits and joins."""
        self.assertTrue(self.compare_maps("merging_lanelets_utm"))

    def test_map_with_priorities(self):
        """Basic test file including priorities."""
        self.assertTrue(self.compare_maps("traffic_priority_lanelets_utm"))

    def test_map_with_speed_limits(self):
        """Basic test file including speed limits."""
        self.assertTrue(self.compare_maps("traffic_speed_limit_utm"))

    def test_signalized_intersection_uses_incoming_lanelets(self):
        with tempfile.NamedTemporaryFile("w", suffix=".osm", delete=False) as tmp_file:
            tmp_file.write(_signalized_intersection_osm_xml())
            tmp_file_path = tmp_file.name

        try:
            scenario = self.load_and_convert("unused", file_path=tmp_file_path)
        finally:
            os.remove(tmp_file_path)

        self.assertEqual(1, len(scenario.lanelet_network.intersections))

        lanelets_by_description = {
            str(getattr(lanelet, "description", "")): lanelet for lanelet in scenario.lanelet_network.lanelets
        }
        self.assertTrue(lanelets_by_description["2001"].traffic_lights)
        self.assertTrue(lanelets_by_description["2003"].traffic_lights)
        self.assertFalse(lanelets_by_description["2002"].traffic_lights)
        self.assertFalse(lanelets_by_description["2004"].traffic_lights)

        directions = {traffic_light.direction for traffic_light in scenario.lanelet_network.traffic_lights}
        self.assertIn(TrafficLightDirection.STRAIGHT, directions)

    @unittest.skip("there are minor differences between the file at the end of the pipeline")
    def test_geodetic_transformation(self):
        """
        Convert lanelet2 to CR, then back to lanelet2 and again to CR in order to test
        whether the projection works correctly.
        More precisely: when converting from lanelet2 to CR, the method for projecting geodesic
        coordinates can be configured and is stored in the CR-scenario. When converting back
        to lanelet2, this stored projection method should be considered.
        """
        lanelet2_file_name = "urban-1_lanelets_utm"
        proj_string = utm_default

        cr_scenario = self.load_and_convert(lanelet2_file_name, proj_string=proj_string)
        l2osm = CR2LaneletConverter()
        lanelet2 = l2osm(cr_scenario)
        lanelet2_converted_file_name = f"{lanelet2_file_name}__converted"
        lanelet2_converted_file_path = f"{get_tmp_dir()}{lanelet2_converted_file_name}.osm"
        etree.ElementTree(lanelet2).write(lanelet2_converted_file_path, pretty_print=True)

        self.assertTrue(
            self.compare_maps(lanelet2_file_name, file_path=lanelet2_converted_file_path)
        )
