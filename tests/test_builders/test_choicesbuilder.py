from spec_repair.builders.choicesbuilder import ChoicesBuilder
from spec_repair.builders.enums import ChoiceType
from tests.test_builders.test_builder import TestBuilderCase


class TestChoicesBuilder(TestBuilderCase):
    maxDiff = None
    def test_record_choice_taken_1(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice, 0)

    def test_record_choice_taken_2(self):
        input_lines = """


Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '2'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice, 2)

    def test_record_options_type_1(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice_type, ChoiceType.EXP_WEAKENING)

    def test_record_options_type_2(self):
        input_lines = """


Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '2'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice_type, ChoiceType.EXP_WEAKENING)

    def test_record_options_type_3(self):
        input_lines = """
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '1'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice_type, ChoiceType.CS_GENERATION)

    def test_record_options_type_4(self):
        input_lines = """


Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.choice_type, ChoiceType.CS_DEADLOCK_COMPLETION)

    def test_record_options_1(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'

        """
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.options[0],
                          "['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']")
        self.assertEquals(builder.options[1],
                          "['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']")
        self.assertEquals(builder.options[2],
                          "['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']")

    def test_record_options_2(self):
        input_lines = """
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '1'

"""
        builder = ChoicesBuilder()
        builder.start()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        self.assertEquals(builder.options[0],
                          "('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')")
        self.assertEquals(builder.options[1],
                          "('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')")

    def test_record_assertion_error_bad_keyword(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'

        """
        builder = ChoicesBuilder()
        with self.assertRaises(ValueError):
            for line in input_lines.split('\n'):
                builder.record(line)
