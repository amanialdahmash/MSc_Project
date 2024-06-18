from unittest import TestCase

from spec_repair.config import PROJECT_PATH
from spec_repair.util.file_util import is_file_format


class TestUtil(TestCase):
    def test_is_file_format(self):
        real_file_path: str = f"{PROJECT_PATH}/minempump_fixed.spectra"
        self.assertTrue(
            is_file_format(real_file_path, ".spectra")
        )

        self.assertFalse(
            is_file_format(f"complete_jibberish^etc.txt", ".txt")
        )

        self.assertFalse(
            is_file_format(real_file_path, ".txt")
        )
