from spec_repair.builders.enums import Rule
from spec_repair.builders.rulebuilder import RuleBuilder
from tests.test_builders.test_builder import TestBuilderCase


class TestRuleBuilder(TestBuilderCase):
    def test__record_rule_builder_rules(self):
        input_lines = """
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

        """
        builder = RuleBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)

        self.assertIsNone(builder.recording)
        self.assertEquals(builder.rules[0].old, "G(highwater=false|methane=false);")
        self.assertEquals(builder.rules[0].new, "G(highwater=false-> highwater=false|methane=false);")

    def test__record_rule_builder_name_antecedent(self):
        input_lines = """
Rule:

    G(highwater=false|methane=false);

Hypothesis:

    antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

    G(highwater=false-> highwater=false|methane=false);

Unrealizable

            """
        builder = RuleBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)

        self.assertIsNone(builder.recording)
        self.assertEquals(builder.rules[0].name, "assumption2_1")

    def test__record_rule_builder_name_consequent(self):
        input_lines = """
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

Realizable: success.
        """
        builder = RuleBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)

        self.assertIsNone(builder.recording)
        self.assertEquals(builder.rules[0].name, "guarantee1_1")

    def test__record_rule_builder_assertion_error_bad_keyword(self):
        input_lines = """
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:
        """
        builder = RuleBuilder()
        with self.assertRaises(ValueError):
            for line in input_lines.split('\n'):
                builder.record(line)

    def test__record_rule_builder_two_rules(self):
        input_lines = """
Rule:

	G(methane=true->next(pump=false));

Hypothesis:

	consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(methane=true->highwater=true|next(pump=false));

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

Realizable: success.
"""
        builder = RuleBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertIsNone(builder.recording)
        self.assertEqualsRule(builder.rules[0],
                              Rule("guarantee2_1",
                                   "G(methane=true->next(pump=false));",
                                   "G(methane=true->highwater=true|next(pump=false));"))
        self.assertEqualsRule(builder.rules[1],
                              Rule("guarantee1_1",
                                   "G(highwater=true->next(pump=true));",
                                   "G(highwater=true->highwater=true|next(pump=true));"))

    def assertEqualsRule(self, rule: Rule, some_rule: Rule):
        self.assertEquals(rule.name, some_rule.name)
        self.assertEquals(rule.old, some_rule.old)
        self.assertEquals(rule.new, some_rule.new)
