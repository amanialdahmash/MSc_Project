from unittest import TestCase

import pandas as pd

from spec_repair.components.spec_encoder import SpecEncoder
from spec_repair.components.spec_generator import SpecGenerator
from spec_repair.util.file_util import read_file
from spec_repair.util.spec_util import get_assumptions_and_guarantees_from


class TestSpecEncoder(TestCase):
    minepump_spec_file = '../../input-files/examples/Minepump/minepump_strong.spectra'
    minepump_clingo_file = '../test_files/minepump_strong_WA_no_cs.lp'
    maxDiff = None

    def test_encode_asp(self):
        expected_clingo_str: str = read_file(self.minepump_clingo_file)
        spec_df: pd.DataFrame = get_assumptions_and_guarantees_from(self.minepump_spec_file)
        trace_name = "trace_name_0"
        trace = [
            f'not_holds_at(highwater,0,{trace_name}).\n',
            f'not_holds_at(methane,0,{trace_name}).\n',
            f'not_holds_at(pump,0,{trace_name}).\n',
            '\n',
            f'holds_at(highwater,1,{trace_name}).\n',
            f'holds_at(methane,1,{trace_name}).\n',
            f'not_holds_at(pump,1,{trace_name}).\n',
            '\n'
        ]
        encoder: SpecEncoder = SpecEncoder(SpecGenerator())
        clingo_str: str = encoder.encode_ASP(spec_df, trace, set())
        clingo_str = clingo_str.replace('\n\n\n', '\n\n')

        self.assertEqual(expected_clingo_str, clingo_str)
