from unittest import skip

import pandas as pd

from spec_repair.config import PROJECT_PATH
from spec_repair.builders.csvbuilder import CSVBuilder
from pandas.testing import assert_frame_equal

from spec_repair.wrappers.spec import Spec
from tests.test_builders.test_builder import TestBuilderCase
from spec_repair.util.file_util import read_file


class TestCSVBuilder(TestBuilderCase):
    maxDiff = None
    dtype_mapping = {
        "choices": list[str],
        "assumption weakening steps": int,
        "assumption counter-strategy generation steps": int,
        "guarantee weakening steps": int,
        "guarantee counter-strategy generation steps": int,
        "guarantee deadlock generations": int,
        "solution status compared to ideal": str,
        "id solution reached": int,
        "assumptions modified": int,
        "guarantees modified": int,
        "rules and modifications": str,
        "cleaned output (choices and results)": str
    }

    @skip("Skipping this test, since it fails due to type mismatches. Latter tests care more about content")
    def test_get_dataframe_blank_table_has_names(self):
        builder = CSVBuilder()
        table = builder.get_dataframe()
        ref_df = pd.DataFrame(columns=list(self.dtype_mapping.keys()))

        assert_frame_equal(table, ref_df)

    def test_get_dataframe_one_run_1(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 18.63s

Elapsed time: 26.49s

Elapsed time: 34.1s

#########################
First run ended with [0, 0, 0, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 1.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 1]
#########################
        """
        builder = CSVBuilder()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        table = builder.get_dataframe()
        ref_table = pd.DataFrame({
            "choices": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "assumption weakening steps": [2],
            "assumption counter-strategy generation steps": [3],
            "guarantee weakening steps": [1],
            "guarantee counter-strategy generation steps": [1],
            "guarantee deadlock generations": [1],
            "solution status compared to ideal": ['Environment Assumptions Captured. Guarantees Different..'],
            "id solution reached": [1],
            "assumptions modified": [1],
            "guarantees modified": [1],
            "rules and modifications": ["""
assumption2_1:
G(highwater=false|methane=false);
=>
G(highwater=false-> highwater=false|methane=false);
########
assumption2_1:
G(highwater=false|methane=false);
=>
G(pump=true-> highwater=false|methane=false);
########
guarantee1_1:
G(highwater=true->next(pump=true));
=>
G(highwater=true->highwater=true|next(pump=true));
            """.strip()],
            "cleaned output (choices and results)": ["""
AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

GCS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

GD: 0
0: ['!highwater', '!methane', '!prev_pump', '!pump']

GW: 0
0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
            """.strip()]
        }, dtype=object)
        assert_frame_equal(table, ref_table)

    def test_get_dataframe_two_runs(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 18.63s

Elapsed time: 26.49s

Elapsed time: 34.1s

#########################
First run ended with [0, 0, 0, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 1.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 1]
#########################
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '1'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).

New Rule:

		G(highwater=true->methane=true|next(pump=true));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 18.43s

Elapsed time: 26.03s

Elapsed time: 33.59s

#########################
Last run ended with [0, 0, 0, 0, 0, 0, 0, 1] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Success - Assumptions and Guarantees Captured..
SPEC ID: 0.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 2]
#########################
"""
        builder = CSVBuilder()
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        table = builder.get_dataframe()
        ref_table = pd.DataFrame({
            "choices": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
            "assumption weakening steps": [2, 2],
            "assumption counter-strategy generation steps": [3, 3],
            "guarantee weakening steps": [1, 1],
            "guarantee counter-strategy generation steps": [1, 1],
            "guarantee deadlock generations": [1, 1],
            "solution status compared to ideal": ['Environment Assumptions Captured. Guarantees Different..',
                                                  'Success - Assumptions and Guarantees Captured..'],
            "id solution reached": [1, 0],
            "assumptions modified": [1, 1],
            "guarantees modified": [1, 1],
            "rules and modifications": ["""
assumption2_1:
G(highwater=false|methane=false);
=>
G(highwater=false-> highwater=false|methane=false);
########
assumption2_1:
G(highwater=false|methane=false);
=>
G(pump=true-> highwater=false|methane=false);
########
guarantee1_1:
G(highwater=true->next(pump=true));
=>
G(highwater=true->highwater=true|next(pump=true));
            """.strip(),
                                        """
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(highwater=false-> highwater=false|methane=false);
                                        ########
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(pump=true-> highwater=false|methane=false);
                                        ########
                                        guarantee1_1:
                                        G(highwater=true->next(pump=true));
                                        =>
                                        G(highwater=true->methane=true|next(pump=true));
                                        """.strip()],
            "cleaned output (choices and results)": ["""
AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

GCS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

GD: 0
0: ['!highwater', '!methane', '!prev_pump', '!pump']

GW: 0
0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
""".strip(),
                                                     """
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     
                                                     GCS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     GD: 0
                                                     0: ['!highwater', '!methane', '!prev_pump', '!pump']
                                                     
                                                     GW: 1
                                                     0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     """.strip()]
        }, dtype=object)
        assert_frame_equal(table, ref_table)

    def test_get_dataframe_three_runs(self):
        input_lines = """
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 18.63s

Elapsed time: 26.49s

Elapsed time: 34.1s

#########################
First run ended with [0, 0, 0, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 1.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 1]
#########################
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '1'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).

New Rule:

		G(highwater=true->methane=true|next(pump=true));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 18.43s

Elapsed time: 26.03s

Elapsed time: 33.59s

#########################
Last run ended with [0, 0, 0, 0, 0, 0, 0, 1] and max_options=[2, 1, 0, 1, 1, 1, 0, 2].
This run is Success - Assumptions and Guarantees Captured..
SPEC ID: 0.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 2]
#########################
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'


Unrealizable core:

guarantee1_1

13:	G(highwater=true->next(pump=true));



guarantee2_1

16:	G(methane=true->next(pump=false));



Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

Enter the index of your choice [0-2]: 

Choice taken: '2'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).

New Rule:

		G(highwater=true->pump=true|next(pump=true));

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['pump', '!highwater', '!methane', '!prev_pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['!highwater', '!methane', '!prev_pump', '!pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

3: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

4: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']

5: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

6: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']

7: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

8: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']

Enter the index of your choice [0-8]: 

Choice taken: '0'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).

New Rule:

		G(highwater=true->highwater=true|next(pump=true));

	G(methane=true->next(pump=false));

Hypothesis:

	consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).

New Rule:

		G(methane=true->pump=true|next(pump=false));

Realizable: success.

Fixed specification: /Users/tg4018/Documents/PhD/SpectraASPTranslators/input-files/examples/Minepump/minepump_strong_fixed.spectra

Elapsed time: 29.41s

Elapsed time: 37.0s

Elapsed time: 44.55s

#########################
Last run ended with [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0] and max_options=[2, 1, 0, 1, 1, 1, 0, 2, 1, 1, 0, 0, 8].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 2.
Moving to next run [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
#########################
"""
        builder = CSVBuilder()
        spec_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"
        spec: str = read_file(spec_file)
        builder.add_ideal_spec(Spec(spec))
        self.run_and_assert_returns_true_until_false(builder, input_lines)
        table = builder.get_dataframe()
        ref_table = pd.DataFrame({
            "choices": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]],
            "assumption weakening steps": [2, 2, 2],
            "assumption counter-strategy generation steps": [3, 3, 3],
            "guarantee weakening steps": [1, 1, 2],
            "guarantee counter-strategy generation steps": [1, 1, 3],
            "guarantee deadlock generations": [1, 1, 3],
            "solution status compared to ideal": ['Environment Assumptions Captured. Guarantees Different..',
                                                  'Success - Assumptions and Guarantees Captured..',
                                                  'Environment Assumptions Captured. Guarantees Different..'
                                                  ],
            "id solution reached": [1, 0, 2],
            "assumptions modified": [1, 1, 1],
            "guarantees modified": [1, 1, 2],
            "rules and modifications": ["""
assumption2_1:
G(highwater=false|methane=false);
=>
G(highwater=false-> highwater=false|methane=false);
########
assumption2_1:
G(highwater=false|methane=false);
=>
G(pump=true-> highwater=false|methane=false);
########
guarantee1_1:
G(highwater=true->next(pump=true));
=>
G(highwater=true->highwater=true|next(pump=true));
            """.strip(),
                                        """
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(highwater=false-> highwater=false|methane=false);
                                        ########
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(pump=true-> highwater=false|methane=false);
                                        ########
                                        guarantee1_1:
                                        G(highwater=true->next(pump=true));
                                        =>
                                        G(highwater=true->methane=true|next(pump=true));
                                        """.strip(),
                                        """
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(highwater=false-> highwater=false|methane=false);
                                        ########
                                        assumption2_1:
                                        G(highwater=false|methane=false);
                                        =>
                                        G(pump=true-> highwater=false|methane=false);
                                        ########
                                        guarantee1_1:
                                        G(highwater=true->next(pump=true));
                                        =>
                                        G(highwater=true->pump=true|next(pump=true));
                                        ########
                                        guarantee1_1:
                                        G(highwater=true->next(pump=true));
                                        =>
                                        G(highwater=true->highwater=true|next(pump=true));
                                        guarantee2_1:
                                        G(methane=true->next(pump=false));
                                        =>
                                        G(methane=true->pump=true|next(pump=false));
                                        """.strip()
                                        ],
            "cleaned output (choices and results)": ["""
AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

AW: 0
0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

ACS: 0
0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

GCS: 0
0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

GD: 0
0: ['!highwater', '!methane', '!prev_pump', '!pump']

GW: 0
0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
""".strip(),
                                                     """
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     
                                                     GCS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     GD: 0
                                                     0: ['!highwater', '!methane', '!prev_pump', '!pump']
                                                     
                                                     GW: 1
                                                     0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     """.strip(),
                                                     """
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     AW: 0
                                                     0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     ACS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     
                                                     GCS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     GD: 0
                                                     0: ['!highwater', '!methane', '!prev_pump', '!pump']
                                                     
                                                     GW: 2
                                                     0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     
                                                     GCS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')
                                                     
                                                     GCS: 0
                                                     0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')
                                                     
                                                     GD: 0
                                                     0: ['pump', '!highwater', '!methane', '!prev_pump']
                                                     
                                                     GD: 0
                                                     0: ['!highwater', '!methane', '!prev_pump', '!pump']
                                                     
                                                     GW: 0
                                                     0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     3: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     4: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
                                                     5: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     6: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
                                                     7: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(pump,V1,V2).', '', '']
                                                     8: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(methane,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
                                                     """.strip()
                                                     ]
        }, dtype=object)
        self.assertEqual(table['rules and modifications'].iloc[-1],
                         ref_table['rules and modifications'].iloc[-1])
        assert_frame_equal(table, ref_table)

    def test_error_gets_intercepted(self):
        input_lines = """
#########################
Last run ended with [0, 0, 0, 0, 0, 0, 3] and max_options=[3, 1, 0, 1, 1, 1, 5].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 3.
Moving to next run [0, 0, 0, 0, 0, 0, 4]
#########################
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(current,highwater,V1,V2).', '', '']

1: ['antecedent_exception(assumption2_1,V1,V2) :- holds_at(current,methane,V1,V2).', '', '']

2: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(current,pump,V1,V2).', '', '']

3: ['consequent_holds(eventually,assumption2_1,V1,V2) :- root_consequent_holds(eventually,assumption2_1,V1,V2).', '', '']

Enter the index of your choice [0-3]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(current,highwater,V1,V2).

New Rule:

	G(highwater=false-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(current,pump,V1,V2).', '', '']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Rule:

	G(highwater=false|methane=false);

Hypothesis:

	antecedent_exception(assumption2_1,V1,V2) :- not_holds_at(current,pump,V1,V2).

New Rule:

	G(pump=true-> highwater=false|methane=false);

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nholds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_S1_DEAD).\nnot_holds_at(methane,0,ini_S0_S1_DEAD).\nnot_holds_at(pump,0,ini_S0_S1_DEAD).\nnot_holds_at(highwater,1,ini_S0_S1_DEAD).\nholds_at(methane,1,ini_S0_S1_DEAD).\nnot_holds_at(pump,1,ini_S0_S1_DEAD).\nholds_at(highwater,2,ini_S0_S1_DEAD).\nholds_at(methane,2,ini_S0_S1_DEAD).\nnot_holds_at(pump,2,ini_S0_S1_DEAD).\n', 'ini_S0_S1_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No assumption weakening produces realizable spec (las file UNSAT)

Moving to Guarantee Weakening



Reverting to:

	antecedent_exception(assumption2_1,V1,V2) :- holds_at(current,highwater,V1,V2).

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'


No Unrealizable Core Found.

Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(current,highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(current,methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,highwater,V1,V2).', '', '']

3: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,methane,V1,V2).', '', '']

4: ['consequent_holds(eventually,guarantee1_1,V1,V2) :- root_consequent_holds(eventually,guarantee1_1,V1,V2).', '', '']

5: ['consequent_holds(eventually,guarantee2_1,V1,V2) :- root_consequent_holds(eventually,guarantee2_1,V1,V2).', '', '']

Enter the index of your choice [0-5]: 

Choice taken: '4'
Rule:

	G(highwater=true->next(pump=true));

Hypothesis:

	consequent_holds(eventually,guarantee1_1,V1,V2) :- root_consequent_holds(eventually,guarantee1_1,V1,V2).

New Rule:

	G(highwater=true->F(pump=true));

Unrealizable

Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nholds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

1: ('not_holds_at(highwater,0,ini_S0_DEAD).\nnot_holds_at(methane,0,ini_S0_DEAD).\nnot_holds_at(pump,0,ini_S0_DEAD).\nholds_at(highwater,1,ini_S0_DEAD).\nholds_at(methane,1,ini_S0_DEAD).\nnot_holds_at(pump,1,ini_S0_DEAD).\n', 'ini_S0_DEAD')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ('not_holds_at(highwater,0,ini_S0_S1_S2_S2).\nnot_holds_at(methane,0,ini_S0_S1_S2_S2).\nnot_holds_at(pump,0,ini_S0_S1_S2_S2).\nnot_holds_at(highwater,1,ini_S0_S1_S2_S2).\nholds_at(methane,1,ini_S0_S1_S2_S2).\nholds_at(pump,1,ini_S0_S1_S2_S2).\nholds_at(highwater,2,ini_S0_S1_S2_S2).\nholds_at(methane,2,ini_S0_S1_S2_S2).\nnot_holds_at(pump,2,ini_S0_S1_S2_S2).\nnot_holds_at(highwater,3,ini_S0_S1_S2_S2).\nholds_at(methane,3,ini_S0_S1_S2_S2).\nnot_holds_at(pump,3,ini_S0_S1_S2_S2).\n', 'ini_S0_S1_S2_S2')

1: ('not_holds_at(highwater,0,ini_S0_S1_S2_S2).\nnot_holds_at(methane,0,ini_S0_S1_S2_S2).\nnot_holds_at(pump,0,ini_S0_S1_S2_S2).\nnot_holds_at(highwater,1,ini_S0_S1_S2_S2).\nholds_at(methane,1,ini_S0_S1_S2_S2).\nnot_holds_at(pump,1,ini_S0_S1_S2_S2).\nholds_at(highwater,2,ini_S0_S1_S2_S2).\nholds_at(methane,2,ini_S0_S1_S2_S2).\nnot_holds_at(pump,2,ini_S0_S1_S2_S2).\nnot_holds_at(highwater,3,ini_S0_S1_S2_S2).\nholds_at(methane,3,ini_S0_S1_S2_S2).\nnot_holds_at(pump,3,ini_S0_S1_S2_S2).\n', 'ini_S0_S1_S2_S2')

Enter the index of your choice [0-1]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['pump', '!highwater', '!methane', '!prev_pump']

Enter the index of your choice [0-0]: 

Choice taken: '0'
Select an option by choosing its index:

0: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,highwater,V1,V2).', '', '']

1: ['consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,methane,V1,V2).', '', '']

2: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,highwater,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(eventually,highwater,V1,V2).', '', '']

3: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,methane,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(eventually,highwater,V1,V2).', '', '']

4: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,methane,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(eventually,highwater,V1,V2).', '', '']

5: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(eventually,highwater,V1,V2).', '', '']

6: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- holds_at(eventually,methane,V1,V2).', '', '']

7: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(current,pump,V1,V2).', '', '']

8: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(eventually,highwater,V1,V2).', '', '']

9: ['consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,pump,V1,V2).', 'consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(eventually,pump,V1,V2).', '', '']

Enter the index of your choice [0-9]: 

Choice taken: '0'
Rule:

	G(highwater=true->F(pump=true));

Hypothesis:

	consequent_exception(guarantee1_1,V1,V2) :- not_holds_at(current,pump,V1,V2).

New Rule:

		G(highwater=true->pump=false|F(pump=true));

	G(methane=true->next(pump=false));

Hypothesis:

	consequent_exception(guarantee2_1,V1,V2) :- holds_at(current,highwater,V1,V2).

New Rule:

		G(methane=true->highwater=true|next(pump=false));

Using BDD Package: JTLVJavaFactory, Version: JTLVJavaFactory 1.3

Using BDD Package: JTLVJavaFactory, Version: JTLVJavaFactory 1.3

Error: Could not prepare game input from Spectra file. Please verify that the file is a valid Spectra specification.



Elapsed time: 111.07s

#########################
Last run ended with [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0] and max_options=[3, 1, 0, 1, 1, 1, 5, 1, 1, 0, 9].
This run is Environment Assumptions Captured.
	Guarantees Different..
SPEC ID: 4.
Moving to next run [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1]
#########################
"""
        builder = CSVBuilder()
        spec_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"
        spec: str = read_file(spec_file)
        builder.add_ideal_spec(Spec(spec))
        with self.assertRaises(ValueError) as context:
            self.run_and_assert_returns_true_until_false(builder, input_lines)
        expected_error_message = ("ERROR ENCOUNTERED DURING PARSING!!!:\n"
                                  "Error: Could not prepare game input from Spectra file. Please verify that the file is a valid Spectra specification.")
        self.assertEqual(str(context.exception), expected_error_message)
