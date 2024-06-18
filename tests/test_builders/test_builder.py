from unittest import TestCase

from spec_repair.builders.abstract_builder import AbstractBuilder


class TestBuilderCase(TestCase):
    def run_and_assert_returns_true_until_false(self, builder: AbstractBuilder, input_lines: str):
        prev_continues = True
        continues = False
        for line in input_lines.split('\n'):
            continues = builder.record(line)
            self.assertFalse(not prev_continues and continues)
            prev_continues = continues
        self.assertFalse(continues)
