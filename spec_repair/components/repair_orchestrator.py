from typing import Optional, List, Dict

from spec_repair.components.counter_trace import CounterTrace
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.enums import Learning
from spec_repair.exceptions import NoWeakeningException
from spec_repair.heuristics import (
    choose_one_with_heuristic,
    first_choice,
    manual_choice,
    random_choice,
)
from spec_repair.ltl import CounterStrategy
from spec_repair.special_types import StopHeuristicType
from spec_repair.util.file_util import read_file
from spec_repair.util.spec_util import CSTraces, extract_trace, format_spec


def counter_strat_to_trace(lines: Optional[List[str]] = None) -> Dict[str, str]:
    start = "INI"
    output = ""
    trace_name_dict: dict[str, str] = {}
    extract_trace(lines, output, start, 0, "ini", trace_name_dict)

    return trace_name_dict


class RepairOrchestrator:
    def __init__(self, learner: SpecLearner, oracle: SpecOracle):
        self._learner = learner
        self._oracle = oracle
        self._ct_cnt = 0

    # Reimplementation of the highest level abstraction code
    def repair_spec(
        self,
        spec: list[str],
        trace: list[str],
        stop_heuristic: StopHeuristicType = lambda a, g: True,
    ):
        self._ct_cnt = 0
        ct_asm, ct_gar = [], []
        weak_spec_history = []

        # Assumption Weakening for Consistency
        weak_spec: list[str] = self._learner.learn_weaker_spec(
            spec,
            trace,
            list(),
            learning_type=Learning.ASSUMPTION_WEAKENING,
            ##heuristic=manual_choice,
        )
        weak_spec_history.append(weak_spec)
        cs: Optional[CounterStrategy] = self._oracle.synthesise_and_check(weak_spec)

        # Assumption Weakening for Realisability
        try:
            while cs:  # not is_realisable
                ct_asm.append(self.ct_from_cs(cs))
                weaker_spec: list[str] = self._learner.learn_weaker_spec(
                    spec,
                    trace,
                    ct_asm,
                    learning_type=Learning.ASSUMPTION_WEAKENING,
                    ##heuristic=manual_choice,
                )
                if weaker_spec == spec and stop_heuristic(spec, ct_asm):
                    break
                weak_spec_history.append(weaker_spec)
                cs = self._oracle.synthesise_and_check(weaker_spec)
        except NoWeakeningException as e:
            print(str(e))

        if not cs:
            return weak_spec_history[-1]
        print("Moving to Guarantee Weakening")

        # Guarantee Weakening
        spec = weak_spec_history[0]
        ct_gar.append(ct_asm[0])
        while cs:
            spec: list[str] = self._learner.learn_weaker_spec(
                spec,
                trace,
                ct_gar,
                learning_type=Learning.GUARANTEE_WEAKENING,
                ##heuristic=manual_choice,
            )
            cs = self._oracle.synthesise_and_check(spec)
            if cs:
                ct_gar.append(self.ct_from_cs(cs))

        return spec

    def ct_from_cs(self, cs: list[str]) -> CounterTrace:
        ct_name = f"counter_strat_{self._ct_cnt}"
        self._ct_cnt += 1
        return CounterTrace(cs, heuristic=first_choice, name=ct_name)
