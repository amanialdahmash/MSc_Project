from unittest import TestCase

from spec_repair.util.file_util import is_file_extension


class TestFileUtil(TestCase):
    def test_degrade_spec_step(self):
        self.assertTrue(is_file_extension(".csv"))
        self.assertTrue(is_file_extension(".txt"))
        self.assertFalse(is_file_extension("data.csv"))
        self.assertFalse(is_file_extension("file_without_extension"))
        self.assertTrue(is_file_extension(".answer_set"))
