from typing import Optional

from spec_repair.enums import Learning
from spec_repair.heuristics import choose_one_with_heuristic, HeuristicType
from spec_repair.ltl import CounterStrategy
from spec_repair.util.spec_util import cs_to_named_cs_traces, trace_replace_name, trace_list_to_asp_form, \
    trace_list_to_ilasp_form


class CounterTrace:
    def __init__(self, cs: CounterStrategy, heuristic: HeuristicType, name: Optional[str] = None):
        trace_name_dict: dict[str, str] = cs_to_named_cs_traces(cs)
        self._raw_trace, self._path = choose_one_with_heuristic(list(trace_name_dict.items()), heuristic)
        if name is not None:
            self._name = name
            self._raw_trace = trace_replace_name(self._raw_trace, self._path, name)
        else:
            self._name = self._path
        self._is_deadlock = "DEAD" in self._path

    def get_asp_form(self):
        return trace_list_to_asp_form([self._raw_trace])

    def get_ilasp_form(self, learning: Learning, complete_deadlock: bool = False):
        return trace_replace_name(trace_list_to_ilasp_form(self.get_asp_form(), learning), self._path, self._name)
