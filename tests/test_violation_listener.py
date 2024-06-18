from unittest import TestCase

from pipeline.components.violation_listener import is_valid_log_file_name


class TestPipeline(TestCase):
    def test_valid_file_names(self):
        self.assertTrue(is_valid_log_file_name("Minepump_log_12.09.2021_18_13_35.txt"))
        self.assertTrue(is_valid_log_file_name("Minepump_log_27.04.2023_21_18_21.txt"))
        self.assertTrue(is_valid_log_file_name("TowersOfHanoi_log_02.06.2023_19_42_20.txt"))

    def test_invalid_file_names(self):
        self.assertFalse(is_valid_log_file_name("app_log_data.txt"))
        self.assertFalse(is_valid_log_file_name("my_log_file.txt"))
        self.assertFalse(is_valid_log_file_name("log_file_2022.txt"))
        self.assertFalse(is_valid_log_file_name("log_data.txt"))

