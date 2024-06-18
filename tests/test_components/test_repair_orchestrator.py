import os
from unittest import TestCase

from spec_repair.components.repair_orchestrator import RepairOrchestrator
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.old.specification_helper import read_file
from spec_repair.util.spec_util import format_spec


class TestRepairOrchestrator(TestCase):

    @classmethod
    def setUpClass(cls):
        # Change the working directory to the script's directory
        cls.original_working_directory = os.getcwd()
        test_components_dir = os.path.dirname(os.path.abspath(__file__))
        tests_dir = os.path.dirname(test_components_dir)
        os.chdir(tests_dir)

    @classmethod
    def tearDownClass(cls):
        # Restore the original working directory
        os.chdir(cls.original_working_directory)

    def test_repair_spec(self):
        spec: list[str] = format_spec(read_file(
            '../input-files/examples/Minepump/minepump_strong.spectra'))
        trace: list[str] = read_file(
            "./test_files/minepump_strong_auto_violation.txt")
        expected_spec: list[str] = format_spec(read_file(
            './test_files/minepump_aw_methane_gw_methane_fix.spectra'))

        repairer: RepairOrchestrator = RepairOrchestrator(SpecLearner(), SpecOracle())
        new_spec = repairer.repair_spec(spec, trace)
        self.assertEqual(expected_spec, new_spec)
