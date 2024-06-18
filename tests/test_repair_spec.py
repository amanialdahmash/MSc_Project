import importlib
import unittest
import subprocess

from spec_repair import config

test_files_path = "pipeline_test_files"


class PipelineTestCase(unittest.TestCase):

    def test_repair_spec_with_required_arguments(self):
        # Test the pipeline with required arguments only
        cmd = ['python', '../scripts/repair_spec.py',
               '-s', f'{test_files_path}/minepump.spectra',
               '-t', f'{test_files_path}/Minepump_log_0.txt']
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Assert that the script executed successfully (exit code 0)
        self.assertEqual(0, result.returncode)

        # Add additional assertions to verify the expected output or behavior

    def test_repair_spec_with_optional_argument(self):
        # Test the pipeline with optional argument
        importlib.reload(config)
        config.MANUAL = True
        cmd = ['python', '../scripts/repair_spec.py',
               '-s', f'{test_files_path}/minepump.spectra',
               '-t', f'{test_files_path}/Minepump_log_0.txt',
               '-o', f'{test_files_path}/minepump_tmp_to_delete.spectra']
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result)

        # Assert that the script executed successfully (exit code 0)
        self.assertEqual(0, result.returncode)

        # Add additional assertions to verify the expected output or behavior

    def test_repair_spec_missing_required_arguments(self):
        # Test the pipeline with missing required arguments
        importlib.reload(config)
        config.MANUAL = True
        cmd = ['python', '../scripts/repair_spec.py',
               '-s', f'{test_files_path}/minepump.spectra']
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Assert that the script exited with a non-zero code (indicating failure)
        self.assertNotEqual(0, result.returncode)

        # Add additional assertions to verify the error message or behavior

    # Add more test cases as needed
