from spec_repair.components.repair_orchestrator import RepairOrchestrator
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.components.rl_agent import RLAgent  ##
from spec_repair.old.specification_helper import write_file, read_file
from spec_repair.old.util_titus import generate_trace_asp
from spec_repair.util.spec_util import format_spec
from sklearn.model_selection import train_test_split  ##
import random
import re
import os
import tempfile


def main():
    ####example from ILASP website: https://doc.ilasp.com/specification/mode_declarations.html
    # initial_mode_dec = {
    #     "modeha": "#modeha(r(var(t1), const(t2))).",
    #     "modeh_p": "#modeh(p).",
    #     "modeb_1": "#modeb(1, p).",
    #     "modeb_2": "#modeb(2, q(var(t1)))",
    #     "constant_c1": "#constant(t2, c1).",
    #     "constant_c2": "#constant(t2, c2).",
    #     "maxv": "#maxv(2).",
    # }
    ####

    # rl_agent = RLAgent(initial_mode_dec)  ##
    # mutations each have spec and trace
    # spec: list[str] = format_spec(read_file("input-files/examples/Minepump/minepump_strong.spectra"))
    # trace: list[str] = read_file("tests/test_files/minepump_strong_auto_violation.txt")
    expected_spec: list[str] = format_spec(
        read_file("tests/test_files/minepump_aw_methane_gw_methane_fix.spectra")
    )
    # initial_mode_dec = derive_initial_mode_dec(expected_spec)
    # rl_agent = RLAgent(initial_mode_dec)  ##
    oracle = SpecOracle()

    print("ORIGINAL IDEAL", expected_spec)
    ##agent
    cs = oracle.synthesise_and_check(expected_spec)
    print("CHECKING THIS")
    num_mut = 10
    muts = generate_mutated_specs(expected_spec, num_mut, oracle)
    print("ALL 10 MUTS", muts)
    # check_unique_mutations(muts)

    mode_dec = derive_initial_mode_dec(expected_spec)
    print("MODEDEC!!")
    print(mode_dec)
    rl_agent = RLAgent(mode_dec)  ##
    repairer: RepairOrchestrator = RepairOrchestrator(
        SpecLearner(rl_agent), SpecOracle()
    )
    violation_traces = generate_violating_traces(expected_spec, muts)  #

    print("violation traces:", violation_traces)

    train_data, test_data = split(muts, violation_traces)

    print("TRAIN:", train_data)
    print("TEST:", test_data)

    ###now for trainiing
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_mut}")
        for spec, trace in train_data:
            mode_dec = derive_initial_mode_dec(spec)
            rl_agent.update_with_spec(mode_dec)  ##
            repairer: RepairOrchestrator = RepairOrchestrator(
                SpecLearner(rl_agent), oracle
            )
            new_spec = repairer.repair_spec(spec, trace)
            write_file(new_spec, "tests/test_files/out/minepump_test_fix.spectra")
        # # change write file

    # rl_agent.save()
    for spec, trace in test_data:
        mode_dec = derive_initial_mode_dec(spec)
        rl_agent.update_with_spec(mode_dec)  ##
        repairer: RepairOrchestrator = RepairOrchestrator(SpecLearner(rl_agent), oracle)
        new_spec = repairer.repair_spec(spec, trace)
        write_file(new_spec, "tests/test_files/out/minepump_test_fix.spectra")


def split(muts, trace, test_size=0.2):
    train_muts, test_muts, train_trace, test_trace = train_test_split(
        muts, trace, test_size=test_size
    )
    return list(zip(train_muts, train_trace)), list(zip(test_muts, test_trace))


def gen_mutation(spec, num):  #####!!
    muts = []
    for _ in range(num):
        mut = spec.copy()
        for i in range(len(mut)):
            if random.random() < 0.5:
                if "true" in mut[i]:
                    mut[i] = mut[i].replace("true", "false")
            elif "false" in mut[i]:
                mut[i] = mut[i].replace("false", "true")
        muts.append(mut)
    return muts


def gen_v_traces(ideal_spec, mut):
    traces = []
    for s in mut:
        trace = generate_violating_traces(ideal_spec, s)
        traces.append(trace)
    return traces


def create_mode_dec(spec):
    v, p, c = parse(spec)
    mode_dec = gen_mode_dec(v, p, c)
    return mode_dec


def gen_mode_dec(v, p, c):
    mode_dec = {}

    for i, pred in enumerate(p):
        mode_dec[f"modeh_{pred}"] = f"#modeh({pred}(var(time)))."
        mode_dec[f"modeb_{i*2}"] = f"#modeb(1,{pred}(var(variable),var(time)))."
        mode_dec[f"modeb_{i*2+1}"] = f"#modeb(1,not_{pred}(var(variable),var(time)))."

    for i, var in enumerate(v):
        mode_dec[f"constant_{i}"] = f"#constant(variable,{var})."

    for i, const in enumerate(c):
        mode_dec[f"constant_{i}"] = f"#constant(constant,{const})."

    mode_dec["maxv"] = "#maxv(2)."

    return mode_dec


def parse(spec):
    v = set()
    p = set()
    c = set()

    for line in spec:
        if line.startswith("env boolean") or line.startswith("sys boolean"):
            var = line.split()[2].strip(";")
            v.add(var)

    if "->" in line or "|" in line:
        parts = re.split(r"->|\||\(|\)|;|&", line)
        for part in parts:
            part = part.strip()
            if part and part not in v:
                if re.match(r"\w+\(.*\)", part):
                    p.add(part.split("(")[0])
                elif part in ["true", "false"]:
                    c.add(part)
                else:
                    v.add(part)
    return v, p, c


def generate_mutated_specs(spec, num_mutations, oracle):  # REWRITE
    def mutate_line(line):
        # Apply multiple types of mutations
        if random.random() < 0.3:
            line = (
                line.replace("true", "false")
                if "true" in line
                else line.replace("false", "true")
            )
        if random.random() < 0.3:
            line = (
                line.replace("next", "eventually")
                if "next" in line
                else line.replace("eventually", "next")
            )
        if random.random() < 0.3:
            line = line.replace("&", "|") if "&" in line else line.replace("|", "&")
        return line

    mutated_specs = []
    attempts = 0
    max_attempts = 100  # Limit the number of attempts
    while len(mutated_specs) < num_mutations and attempts < max_attempts:
        mut = spec.copy()
        for i in range(len(mut)):
            if random.random() < 0.5:  # Randomly decide to mutate each line or not
                mut[i] = mutate_line(mut[i])

        trace_file = tempfile.mktemp(suffix=".spectra")
        # open(trace_file, "a").close()
        try:
            print("NOW CHECKOTHEER?")
            cs = oracle.synthesise_and_check(mut)
            if not cs:
                # print("R")
                if check_unique_mutations(mut, mutated_specs):
                    # trace_file = tempfile.mktemp(suffix=".txt")  ##txt
                    trace_path, _ = generate_trace_asp(spec, mut, trace_file)
                    if trace_path:
                        mutated_specs.append(mut)
                        print("Valid mutation found")
                        print(mut)
        except Exception as e:
            print(f"Error during synthesise_and_check: {e}")
        attempts += 1
    return mutated_specs


def generate_violating_traces(ideal_spec_path, mutated_specs):  # REWRITE
    traces = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, spec in enumerate(mutated_specs):
            mutated_spec_path = os.path.join(temp_dir, f"mutated_spec_{i}.spectra")
            print("1", mutated_spec_path)
            write_file(spec, mutated_spec_path)
            trace_file = os.path.join(temp_dir, f"trace_{i}.spectra")
            print("2", trace_file)
            try:
                trace_path, _ = generate_trace_asp(
                    ideal_spec_path, mutated_spec_path, trace_file
                )
                print("3", trace_path)
                if trace_path:
                    traces.append(read_file(trace_path))
                    print("4")
                else:
                    traces.append([])  # Append an empty trace if generation fails
            except Exception as e:
                print(f"Error during generate_trace_asp: {e}")
                traces.append([])  # Append an empty trace if generation fails
    return traces


def derive_initial_mode_dec(spec):
    constants = set()
    predicates = set()
    for line in spec:
        matches = re.findall(r"\b\w+\b", line)
        for match in matches:
            if match.islower() and match not in {"true", "false"}:
                constants.add(match)
            elif match.islower() is False:
                predicates.add(match)

    # mode_dec = {
    #     "modeha": f"#modeha(r(var(t1),const(t2))).",
    #     "modeh_p": f"#modeh({random.choice(list(predicates))}).",
    #     "modeb_1": f"#modeb(1,{random.choice(list(predicates))}).",
    #     "modeb_2": f"#modeb(2,{random.choice(list(predicates))}(var(t1))).",
    #     "constant_c1": f"#constant(t2,{random.choice(list(constants))}).",
    #     "constant_c1": f"#constant(t2,{random.choice(list(constants))}).",
    #     "maxv": "#maxv(2).",##btwn pred?
    # }

    modeh = [f"#modeh({pred}(var(t1)))." for pred in predicates]
    modeb = [f"#modeb({pred}(var(t1)))." for pred in predicates]
    consts = [f"#constant(t2,{const})." for const in constants]
    maxv_val = max(len(predicates), 2)  ##or just 2?
    maxv = [f"#maxv({maxv_val})"]
    # return mode_dec
    return modeh + modeb + consts + maxv


def check_unique_mutations(new, mutated_specs):
    for e in mutated_specs:
        if new == e:
            return False
    return True


if __name__ == "__main__":
    main()
