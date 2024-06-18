from spec_repair.builders.summarybuilder import SummaryBuilder
from tests.test_builders.test_builder import TestBuilderCase


class TestSummaryBuilder(TestBuilderCase):
    def test_record_solution_status_1(self):
        input_lines = """
#########################
First run ended with [0, 0, 0, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
Moving to next run [0, 0, 0, 0, 0, 0, 0, 1]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual(builder.solution_status,
                         "Environment Assumptions Captured. Guarantees Different..")

    def test_record_solution_status_2(self):
        input_lines = """
#########################
Last run ended with [0, 0, 0, 0, 0, 0, 0, 1] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Success - Assumptions and Guarantees Captured..
Moving to next run [0, 0, 0, 0, 0, 0, 0, 2]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual(builder.solution_status,
                         "Success - Assumptions and Guarantees Captured..")

    def test_record_solution_status_3(self):
        input_lines = """
#########################
Last run ended with [2, 0, 0, 0, 1] and max_options=[2, 1, 1, 0, 2].
This run is Alternative Realizable Specification Produced..
Moving to next run [2, 0, 0, 0, 2]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual(builder.solution_status,
                         "Alternative Realizable Specification Produced..")

    def test_record_choices_current_run_1(self):
        input_lines = """
#########################
First run ended with [0, 0, 0, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
Moving to next run [0, 0, 0, 0, 0, 0, 0, 1]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], builder.choices)

    def test_record_choices_current_run_2(self):
        input_lines = """
#########################
Last run ended with [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2] and max_options=[2, 1, 0, 1, 1, 1, 0, 2, 1, 1, 0, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
Moving to next run [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2], builder.choices)

    def test_record_choices_current_run_3(self):
        input_lines = """
#########################
Last run ended with [2, 0, 0, 0, 1] and max_options=[2, 1, 1, 0, 2].
This run is Alternative Realizable Specification Produced..
Moving to next run [2, 0, 0, 0, 2]
#########################
        """
        builder = SummaryBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEqual([2, 0, 0, 0, 1], builder.choices)
