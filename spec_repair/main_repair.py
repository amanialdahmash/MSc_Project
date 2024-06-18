from spec_repair.components.repair_orchestrator import RepairOrchestrator
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.config import PROJECT_PATH
from spec_repair.ltl import Trace
from spec_repair.util.file_util import generate_asp_trace_to_file
from spec_repair.util.spec_util import get_assumptions_and_guarantees_from

# TODO: use this file as replacement for Specification.py logic


if __name__ == '__main__':
    start_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"
    trace_file = generate_asp_trace_to_file(start_file, end_file)
    spec_df = get_assumptions_and_guarantees_from(start_file)
    trace = Trace(trace_file)
    spec_repairer = RepairOrchestrator(learner=SpecLearner(), oracle=SpecOracle())
    spec_repairer.repair_spec(spec_df, trace)
