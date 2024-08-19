from typing import Optional, List, Dict

from spec_repair.components.counter_trace import CounterTrace
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.enums import Learning
from spec_repair.exceptions import NoViolationException, NoWeakeningException
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
        self._ct_cnt = 0  # cts
        ct_asm, ct_gar = [], []
        weak_spec_history = []
        cs = None  ##
        gen_specs = set()
        max_iter = 10  ##
        iterations = 0

        # cs = self._oracle.synthesise_and_check(spec)
        # if cs is None:
        #     print("REALIIIIISSZZZABBBLLEE OOH")

        # asmp wkn
        once = False
        weak_spec: list[str] = None
        while iterations < max_iter:
            print("ITER", iterations + 1)
            # Assumption Weakening for Consistency
            try:
                weak_spec: list[str] = self._learner.learn_weaker_spec(
                    spec,
                    trace,
                    list(),
                    learning_type=Learning.ASSUMPTION_WEAKENING,
                    ##heuristic=manual_choice,
                )
            except NoViolationException:
                if not once:
                    first = True
                else:
                    cs = self._oracle.synthesise_and_check(spec)
                    if not cs:
                        print("REALIIIIISSZZZABBBLLEE OOH")
                        return spec
                    else:
                        raise
            print("WEEAK", weak_spec)
            if weak_spec:  ##
                spec_str = "".join(weak_spec)  # \n
                if spec_str in gen_specs:
                    print("REPEATED SPEEC")
                    break
                gen_specs.add(spec_str)
                cs = self._oracle.synthesise_and_check(weak_spec)
                print("CS:", cs)
                if not cs:
                    print("REALISABLE")
                    # result = self._learner.rl_agent.update_mode_dec("realisable")
                    # if result == "realisable":
                    #     print("REALIABLE SPECC!!!")
                    return weak_spec  # weak_spec_history[-1] if weak_spec_history else spec
                else:
                    print("not yet1")
                    result = self._learner.rl_agent.update_mode_dec(
                        "counter_strategy_found"
                    )
                    if result == "max":
                        print("MAXXX IN ASSM!!!")
                        break
                weak_spec_history.append(weak_spec)
            else:
                # cs = self._oracle.synthesise_and_check(weak_spec)
                # if not cs:
                #     print("NO")
                #     # result = self._learner.rl_agent.update_mode_dec("realisable")
                #     # if result == "realisable":
                #     #     print("REALIABLE SPECC!!!")
                #     return spec  # weak_spec_history[-1] if weak_spec_history else spec
                # else:
                print("not yet2")
                result = self._learner.rl_agent.update_mode_dec(
                    "counter_strategy_found"
                )
                if result == "max":
                    print("MAXXX IN ASSM!!!")
                    break

            iterations += 1
            # else:
            # result=self._learner.rl_agent.update_mode_dec("counter_strategy_found")

            # it?

            # weak_spec_history.append(weak_spec)
            # cs: Optional[CounterStrategy] = self._oracle.synthesise_and_check(weak_spec)
            # ct_asm.append(self.ct_from_cs(cs))  ##
            # if cs:
            #    break

        # Assumption Weakening for Realisability
        # try:##
        ####?????
        # if cs:
        #     while cs:  # not is_realisable
        #         print("TWOTWOTWO")

        #         print(f"cs found {cs}")
        #         ct_asm.append(self.ct_from_cs(cs))

        #         weaker_spec: list[str] = self._learner.learn_weaker_spec(
        #             spec,
        #             trace,
        #             ct_asm,
        #             learning_type=Learning.ASSUMPTION_WEAKENING,
        #             ##heuristic=manual_choice,
        #         )
        #         # if weaker_spec is None:  ##
        #         #    continue  #
        #         if weak_spec is not None:  ##
        #             cs = self._oracle.synthesise_and_check(weak_spec)
        #             if not cs:
        #                 result = self._learner.rl_agent.update_mode_dec("realisable")
        #                 if result in ["max", "converged", "realisable"]:
        #                     return weak_spec_history[-1]
        #             else:
        #                 result = self._learner.rl_agent.update_mode_dec(
        #                     "counter_strategy_found"
        #                 )
        #                 if result in ["max", "converged"]:
        #                     return weak_spec_history[-1]
        #             weak_spec_history.append(weak_spec)
        # cs = self._oracle.synthesise_and_check(weak_spec)

        # if not cs:
        #     result = self._learner.rl_agent.update_mode_dec("realisable")
        #     if result in ["max", "converged", "realisable"]:
        #         if weak_spec_history[-1]:
        #             return weak_spec_history[-1]
        #         else:
        #             return spec
        # else:
        #     result = self._learner.rl_agent.update_mode_dec(
        #         "counter_strategy_found"
        #     )
        #     if result in ["max", "converged"]:
        #         if weak_spec_history[-1]:
        #             return weak_spec_history[-1]
        #         else:
        #             return spec
        # it?
        # if weaker_spec == spec and stop_heuristic(spec, ct_asm):
        # if weak_spec_history[-1]:
        #     return weak_spec_history[-1]
        # else:
        # return spec
        # weak_spec_history.append(weaker_spec)
        # cs = self._oracle.synthesise_and_check(weaker_spec)
        # except NoWeakeningException as e:##
        #    print(str(e))##

        if not weak_spec_history:  # cs
            print("Repair completed without moving to Guarantee Weakening")
            # if weak_spec_history:
            #     return weak_spec_history[-1]
            # else:
            return spec
        print("Moving to Guarantee Weakening")

        # Guarantee Weakening
        spec = weak_spec_history[-1]  # 0
        # ct_gar.append(ct_asm[0])
        iterations = 0
        while iterations < max_iter:
            print("GAUR WEAK")
            try:
                spec: list[str] = self._learner.learn_weaker_spec(
                    spec,
                    trace,
                    ct_gar,
                    learning_type=Learning.GUARANTEE_WEAKENING,
                    ##heuristic=manual_choice,
                )
            except NoViolationException:
                cs = self._oracle.synthesise_and_check(spec)
                if not cs:
                    print("REALIIIIISSZZZABBBLLEE OOH")
                    return spec
                else:
                    raise
            # if spec is None:  ##
            #    break  #
            # print("SPEC IS", spec)
            # if weak_spec is not None:  ##
            if spec:

                spec_str = "".join(spec)
                if spec_str in gen_specs:
                    print("REPEATED SPEEC")
                    break
                gen_specs.add(spec_str)
                cs = self._oracle.synthesise_and_check(spec)
                print("CS:", cs)
                if not cs:
                    print("REALISABLE")
                    # result = self._learner.rl_agent.update_mode_dec("realisable")
                    # if result == "realisable":
                    #     print("REALIABLE SPECC!!!")
                    # return weak_spec_history[-1] if weak_spec_history else spec
                    return spec
                else:
                    print("not yet")
                    result = self._learner.rl_agent.update_mode_dec(
                        "counter_strategy_found"
                    )
                    if result == "max":
                        print("MAXXX IN ASSM!!!")
                        break
                weak_spec_history.append(spec)
            else:

                result = self._learner.rl_agent.update_mode_dec(
                    "counter_strategy_found"
                )
                if result == "max":
                    print("MAXXX IN ASSM!!!")
                    break

            iterations += 1

        #         cs = self._oracle.synthesise_and_check(weak_spec)
        #         if not cs:
        #             result = self._learner.rl_agent.update_mode_dec("realisable")
        #             if result in ["max", "converged", "realisable"]:
        #                 if weak_spec_history:
        #                     return weak_spec_history[-1]
        #                 else:
        #                     return spec
        #         else:
        #             result = self._learner.rl_agent.update_mode_dec(
        #                 "counter_strategy_found"
        #             )
        #             if result in ["max", "converged"]:
        #                 if weak_spec_history:
        #                     return weak_spec_history[-1]
        #                 else:
        #                     return spec
        #             # it?

        #     # cs = self._oracle.synthesise_and_check(spec)
        #     if spec == spec and stop_heuristic(spec, ct_gar):
        #         # ct_gar.append(self.ct_from_cs(cs))
        #         if weak_spec_history:
        #             return weak_spec_history[-1]
        #         else:
        #             return spec
        # if not cs:
        #     print("Repair completed after moving to Guarantee Weakening")
        # else:
        print("Repair Failed :(")
        if weak_spec_history:
            return weak_spec_history[-1]
        else:
            return spec

    def ct_from_cs(self, cs: list[str]) -> CounterTrace:
        ct_name = f"counter_strat_{self._ct_cnt}"
        self._ct_cnt += 1
        return CounterTrace(cs, heuristic=first_choice, name=ct_name)
