from typing import Set, List
from unittest import TestCase

from spec_repair.util.graph_util import merge_sets


def _freeze_sets_in_list(l: List[Set[any]]) -> Set[frozenset[any]]:
    return {frozenset(s) for s in l}


class TestSpec(TestCase):
    def test_merge_sets_no_merging(self):
        sets_to_merge = [{'11', '14'}, {'1', '12'}, {'13', '0'}]
        merged_sets = merge_sets(_freeze_sets_in_list(sets_to_merge))

        self.assertSetEqual(_freeze_sets_in_list(merged_sets),
                            _freeze_sets_in_list([{'11', '14'}, {'1', '12'}, {'13', '0'}]))

    def test_merge_sets_merging(self):
        sets_to_merge = [{'0', '4'}, {'3', '10'}, {'10', '2'}, {'9', '11'},
         {'6', '7'}, {'9', '8'}, {'2', '11'}, {'12', '13'},
         {'5', '7'}, {'13', '14'}, {'5', '4'}, {'6', '3'},
         {'12', '14'}, {'1', '8'}]
        merged_sets = merge_sets(_freeze_sets_in_list(sets_to_merge))

        self.assertSetEqual(_freeze_sets_in_list(merged_sets),
                            _freeze_sets_in_list([{'0', '6', '5', '4', '1', '2', '9', '8', '3', '10', '7', '11'}, {'12', '13', '14'}]))


