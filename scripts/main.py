from spec_repair.components.repair_orchestrator import RepairOrchestrator
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.components.rl_agent import RLAgent  ##
from spec_repair.old.specification_helper import write_file, read_file
from spec_repair.util.spec_util import format_spec

####example from ILASP website: https://doc.ilasp.com/specification/mode_declarations.html
initial_mode_dec = {
    "modeha": "#modeha(r(var(t1), const(t2))).",
    "modeh_p": "#modeh(p).",
    "modeb_1": "#modeb(1, p).",
    "modeb_2": "#modeb(2, q(var(t1)))",
    "constant_c1": "#constant(t2, c1).",
    "constant_c2": "#constant(t2, c2).",
    "maxv": "#maxv(2).",
}
####

rl_agent = RLAgent(initial_mode_dec)  ##

spec: list[str] = format_spec(
    read_file("input-files/examples/Minepump/minepump_strong.spectra")
)
trace: list[str] = read_file("tests/test_files/minepump_strong_auto_violation.txt")
expected_spec: list[str] = format_spec(
    read_file("tests/test_files/minepump_aw_methane_gw_methane_fix.spectra")
)

repairer: RepairOrchestrator = RepairOrchestrator(
    SpecLearner(rl_agent), SpecOracle()
)  ##agent
new_spec = repairer.repair_spec(spec, trace)
write_file(new_spec, "tests/test_files/out/minepump_test_fix.spectra")
