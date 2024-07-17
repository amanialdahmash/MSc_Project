import re
from typing import Set, Optional, List

import pandas as pd

from spec_repair.components.counter_trace import CounterTrace
from spec_repair.components.spec_encoder import SpecEncoder
from spec_repair.config import FASTLAS
from spec_repair.enums import Learning
from spec_repair.exceptions import NoViolationException, NoWeakeningException
from spec_repair.heuristics import (
    choose_one_with_heuristic,
    random_choice,
    HeuristicType,
    manual_choice,  # only?
)

from spec_repair.ltl import spectra_to_df
from spec_repair.components.spec_generator import SpecGenerator
from spec_repair.wrappers.asp_wrappers import get_violations, run_ILASP


class SpecLearner:
    def __init__(self, rl_agent):  ##rl
        self.file_generator = SpecGenerator()
        self.spec_encoder = SpecEncoder(self.file_generator)
        self.rl_agent = rl_agent

    ####
    def learn_weaker_spec(
        self,
        spec: list[str],
        trace: List[str],
        cs_traces: List[CounterTrace],
        learning_type: Learning,
        ##heuristic: HeuristicType = manual_choice,
    ) -> Optional[list[str]]:
        mode_dec = self.rl_agent.get_mode_dec()  ##
        spec_df: pd.DataFrame = spectra_to_df(spec)
        asp: str = self.spec_encoder.encode_ASP(spec_df, trace, cs_traces)
        violations = get_violations(asp, exp_type=learning_type.exp_type())
        if not violations:
            raise NoViolationException("Violation trace is not violating!")
        ilasp: str = self.spec_encoder.encode_ILASP(
            spec_df, trace, cs_traces, violations, learning_type
        )
        output: str = run_ILASP(ilasp)
        hypotheses = get_hypotheses(output)

        # print("ASP: ", asp)
        # print("ILASP: ", ilasp)
        # print("output: ", output)
        # print("hyp: ", hypotheses)

        if not hypotheses:
            for key in self.rl_agent.mode_dec.keys():  #
                self.rl_agent.failed_attempts[key] += 1  #
            self.rl_agent.update_mode_dec("counter_strategy_found")  #
            ##??why
            raise NoWeakeningException(
                f"No {learning_type.exp_type_str()} weakening produces realizable spec (las file UNSAT)"
            )
        # hypothesis = select_learning_hypothesis(hypotheses, heuristic)
        ##get best instead, auto
        hypothesis = self.select_best_hypothesis(hypotheses)
        new_spec = self.spec_encoder.integrate_learned_hypotheses(
            spec, hypothesis, learning_type
        )
        self.rl_agent.update_mode_dec("realisable")  #
        return new_spec

    def select_best_hypothesis(self, hypotheses):
        best_hyp = None
        best_score = float("inf")

        for hyp in hypotheses:  # search for best in all hypotheses
            score = self.extract(hyp)  ###
            if score > best_score:
                best_score = score
                best_hyp = hyp
        return best_hyp

    def extract(self, hyp):
        match = re.search(r"\(score(\d+)\)", hyp[0])
        if match:
            return int(match.group(1))
        return float("inf")


###i wont use this func: (for now,trying diff approach)
def select_learning_hypothesis(
    hypotheses: List[List[str]], heuristic: HeuristicType
) -> List[str]:
    # TODO: store amount of top_hyp learned
    # TODO: make sure no repeated hypotheses occur
    print(hypotheses)
    all_hyp = hypotheses[1:]
    scores = [
        int(re.search(r"score (\d*)", hyp[0]).group(1))
        for hyp in all_hyp
        if re.search(r"score (\d*)", hyp[0])
    ]
    top_hyp = [hyp[1:] for i, hyp in enumerate(all_hyp) if scores[i] == min(scores)]
    learning_hyp = choose_one_with_heuristic(top_hyp, heuristic)
    return learning_hyp


def get_hypotheses(output: str) -> Optional[List[List[str]]]:
    # if re.search("UNSATISFIABLE", "".join(output)):
    if "UNSATISFIABLE" in output:
        return None
    # if re.search(r"1 \(score 0\)", "".join(output)):
    if re.search(r"1 \(score 0\)", output):
        raise NoViolationException(
            "Learning problem is trivially solvable. "
            "If spec is not realisable, we have a learning error."
        )

    if FASTLAS:
        output = re.sub(r"time\(V\d*\)|trace\(V\d*\)", "", output)
        output = re.sub(r" ,", "", output)
        output = re.sub(r", \.", ".", output)
        output = re.sub(r"\), ", "); ", output)
        hypotheses = [re.sub(r"b'|'", "", output).split("\n")]
    else:
        hypotheses = [
            # part.split("\n") for part in "".join(output).split("%% Solution ")
            part.split("\n")
            for part in output.split("%% Solution ")
        ]
    ##
    hypotheses = [[line for line in hyp if line.strip()] for hyp in hypotheses]
    ##return hypotheses
    return [hyp for hyp in hypotheses if hyp]
