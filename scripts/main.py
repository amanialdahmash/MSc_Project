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

import matplotlib.pyplot as plt


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
        # read_file("tests/test_files/minepump_aw_methane_gw_methane_fix.spectra")
        read_file("tests/test_files/forklift_test.spectra")
        # read_file("tests/test_files/CS2.spectra")
    )

    oracle = SpecOracle()

    print("ORIGINAL IDEAL", expected_spec)
    num_mut = 10
    muts, violation_traces = generate_mutated_specs(expected_spec, num_mut, oracle)
    print("ALL 10 MUTS", muts)
    # check_unique_mutations(muts)

    mode_dec = derive_initial_mode_dec(expected_spec)
    # print("MODEDEC!!")
    # print(mode_dec)
    rl_agent = RLAgent(mode_dec)  ##
    # repairer: RepairOrchestrator = RepairOrchestrator(SpecLearner(rl_agent), SpecOracle())
    # violation_traces = generate_violating_traces(expected_spec, muts)  #

    print("violation traces:", violation_traces)

    train_data, test_data = split(muts, violation_traces)

    print("TRAIN:", train_data)
    print("TEST:", test_data)

    training_rewards = []
    eval_rewards = []
    training_loss = []
    eval_loss = []
    ###now for trainiing
    num_epochs = 1000
    rl_agent.training = True
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_reward = []
        epoch_loss = []
        for spec, trace in train_data:
            # mode_dec = derive_initial_mode_dec(spec)
            rl_agent.update_with_spec(mode_dec)  ##resets mode_dec
            repairer: RepairOrchestrator = RepairOrchestrator(
                SpecLearner(rl_agent), oracle
            )
            new_spec = repairer.repair_spec(spec, trace, rl_agent.training)

            state = rl_agent.extract_features(spec, trace)  ##
            action = rl_agent.select_action(state)
            q_val = rl_agent.q_table.get(str(state), {}).get(action, 0)
            # print("QVAL", q_val)
            if new_spec != spec:
                print("SUCCESS")
                # print(spec)
            else:
                # self.q_table[state_str][action]
                # reward = rl_agent.get_reward("counter_strategy_found")
                print("FAIL")
            # loss = rl_agent.loss(state, action, new_spec, trace)
            # epoch_loss.append(loss)
            #     epoch_reward2.append(q_val)
            # eval_rewards.append(epoch_reward2)
            # avg_loss = sum(epoch_loss) / len(epoch_loss)
            # training_loss.append(avg_loss)
            epoch_reward.append(q_val)  ##reward?
        training_rewards.append(epoch_reward)
        ##
        # write_file(new_spec, "tests/test_files/out/minepump_test_fix.spectra")
        # # change write file

    # rl_agent.save()
    #########!!
    # print("TrainingDone")
    # rl_agent.training = False
    # rl_agent.epsilon = 0  ##
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch+1}/{num_epochs}")
    #     epoch_reward2 = []
    #     epoch_loss = []
    #     for spec, trace in test_data:
    #         # mode_dec = derive_initial_mode_dec(spec)
    #         #rl_agent.update_with_spec(mode_dec)  ##?
    #         repairer: RepairOrchestrator = RepairOrchestrator(
    #             SpecLearner(rl_agent), oracle
    #         )
    #         new_spec = repairer.repair_spec(spec, trace, rl_agent.training)

    #         state = rl_agent.extract_features(spec, trace)  ##!!
    #         action = rl_agent.select_action(state)
    #         q_val = rl_agent.q_table.get(str(state), {}).get(action, 0)
    #         # print("QVAL", q_val)
    #         if new_spec != spec:
    #             # reward = rl_agent.get_reward("realisable")
    #             print("SUCCESS")
    #             print(spec)
    #         else:
    #             # reward = rl_agent.get_reward("counter_strategy_found")
    #             print("FAIL")
    #         # loss = rl_agent.loss(state, action, new_spec, trace)
    #         # epoch_loss.append(loss)
    #         epoch_reward2.append(q_val)
    #     eval_rewards.append(epoch_reward2)
    # avg_loss = sum(epoch_loss) / len(epoch_loss)
    # eval_loss.append(avg_loss)

    print("TestingDone")
    print("TRAINGING REWARDS:", training_rewards)
    # print("TESTING REWARDS:", eval_rewards)
    # print("TRAININGLOSS", training_loss)
    # print("TESTNGLOSS", eval_loss)

    # write_file(new_spec, "tests/test_files/out/minepump_test_fix.spectra")

    plt.figure(figsize=(12, 6))
    plt.plot([sum(r) / len(r) for r in training_rewards], label="Training Rewards")
    # plt.plot(eval_rewards, label="Eval Rewards")
    # plt.plot([sum(r) / len(r) for r in eval_rewards], label="Testing Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.title("Rewards over Time")
    plt.legend()
    plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(training_loss, label="Training Loss")
    # # plt.plot(eval_rewards, label="Eval Rewards")
    # plt.plot(eval_loss, label="Testing Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Loss")
    # plt.title("Loss over Time")
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=(12, 6))
    # # plt.plot([sum(r) / len(r) for r in training_rewards], label="Training Rewards")
    # plt.plot(eval_rewards, label="Testing Rewards")
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Reward")
    # plt.title("Rewards over Time (Testing)")
    # plt.legend()
    # plt.show()


def split(muts, trace, test_size=0.2):
    print("muts:", muts)
    print("trace:", trace)
    train_muts, test_muts, train_trace, test_trace = train_test_split(
        muts, trace, test_size=test_size
    )
    return list(zip(train_muts, train_trace)), list(zip(test_muts, test_trace))


# def get vio(self,spec, mut, trace_file):
#     trace_path, violated_trace, _ = generate_trace_asp(
#                         spec, mut, trace_file
#                     )
#     if trace_path:
#                         mutated_specs.append(mut)
#                         violated_traces.append(violated_trace)


def generate_mutated_specs(spec, num_mutations, oracle):
    def mutate_line(line):
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
        if random.random() < 0.3:
            if "G(" in line or "F(" in line:
                line = line.replace("G(", "F(G(").replace("F(G(", "G(")
        return line

    mutated_specs = []
    violated_traces = []
    attempts = 0
    max_attempts = 100
    while len(mutated_specs) < num_mutations and attempts < max_attempts:
        mut = spec.copy()
        for i in range(len(mut)):
            if random.random() < 0.5:
                mut[i] = mutate_line(mut[i])
        trace_file = tempfile.mktemp(suffix=".spectra")
        try:
            cs = oracle.synthesise_and_check(mut)
            if not cs:
                if check_unique_mutations(mut, mutated_specs):
                    trace_path, violated_trace, _ = generate_trace_asp(
                        spec, mut, trace_file
                    )
                    if trace_path:
                        mutated_specs.append(mut)
                        violated_traces.append(violated_trace)
                        print("Valid mutation found")
        except Exception as e:
            print(f"Error during synthesise_and_check: {e}")
        attempts += 1
    return mutated_specs, violated_traces


# def generate_mutated_specs(spec, num_mutations, oracle):  # REWRITE
#     def mutate_line(line):
#         # Apply multiple types of mutations
#         if random.random() < 0.3:
#             line = (
#                 line.replace("true", "false")
#                 if "true" in line
#                 else line.replace("false", "true")
#             )
#         if random.random() < 0.3:
#             line = (
#                 line.replace("next", "eventually")
#                 if "next" in line
#                 else line.replace("eventually", "next")
#             )
#         if random.random() < 0.3:
#             line = line.replace("&", "|") if "&" in line else line.replace("|", "&")
#         ##adding
#         if random.random() < 0.2:
#             if "G(" in line or "F(" in line:
#                 line = line.replace("G(", "F(G(").replace("F(", "G(F(")
#         return line

#     mutated_specs = []
#     violation_traces = []
#     attempts = 0
#     max_attempts = 100  # Limit the number of attempts
#     while len(mutated_specs) < num_mutations and attempts < max_attempts:
#         mut = spec.copy()
#         for i in range(len(mut)):
#             if random.random() < 0.5:  # Randomly decide to mutate each line or not
#                 mut[i] = mutate_line(mut[i])

#         trace_file = tempfile.mktemp(suffix=".spectra")
#         # open(trace_file, "a").close()
#         try:
#             # print("NOW CHECKOTHEER?")
#             cs = oracle.synthesise_and_check(mut)
#             if not cs:
#                 # print("R")
#                 if check_unique_mutations(mut, mutated_specs):
#                     # trace_file = tempfile.mktemp(suffix=".txt")  ##txt
#                     trace_path, violation_trace, _ = generate_trace_asp(
#                         spec, mut, trace_file
#                     )
#                     print("trace path", trace_path)
#                     if trace_path:
#                         mutated_specs.append(mut)
#                         violation_traces.append(violation_trace)
#                         print("Valid mutation found")
#                         print(mut)
#                         print(violation_trace)
#         except Exception as e:
#             print(f"Error during synthesise_and_check: {e}")
#         attempts += 1
#     return mutated_specs, violation_traces


def derive_initial_mode_dec(spec):
    constants = set()
    predicates = set()
    for line in spec:
        matches = re.findall(r"\b\w+\b", line)
        for match in matches:
            if match.islower() and match not in {
                "true",
                "false",
                # "module",
                # "env",
                # "sys",
                # "boolean",
                # "next",
                # "G",
                # "PREV",
            }:
                constants.add(match)
            elif match.isupper() or match[0].isupper():
                predicates.add(match.lower())

    # mode_dec = {
    #     "modeha": f"#modeha(r(var(t1),const(t2))).",
    #     "modeh_p": f"#modeh({random.choice(list(predicates))}).",
    #     "modeb_1": f"#modeb(1,{random.choice(list(predicates))}).",
    #     "modeb_2": f"#modeb(2,{random.choice(list(predicates))}(var(t1))).",
    #     "constant_c1": f"#constant(t2,{random.choice(list(constants))}).",
    #     "constant_c1": f"#constant(t2,{random.choice(list(constants))}).",
    #     "maxv": "#maxv(2).",##btwn pred?
    # }

    modeh = {
        f"#modeh_{i}": f"modeh({pred}(var(t1)))." for i, pred in enumerate(predicates)
    }
    modeb = {
        f"#modeb_{i}": f"#modeb(1,{pred}(var(t1)))."
        for i, pred in enumerate(predicates)
    }
    consts = {
        f"#constant_{i}": f"#constant(t2,{const})." for i, const in enumerate(constants)
    }
    maxv_val = max(len(predicates), 2)  ##or just 2?
    maxv = {"maxv": f"#maxv({maxv_val})"}
    # return mode_dec
    return {**modeh, **modeb, **consts, **maxv}


def check_unique_mutations(new, mutated_specs):
    for e in mutated_specs:
        if new == e:
            return False
    return True


if __name__ == "__main__":
    main()
