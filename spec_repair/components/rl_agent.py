# AD
import random
import copy  #
import re
import numpy as np

from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.enums import Learning


class RLAgent:
    def __init__(self, inital_mode_dec, epsilon=0.1, decay=0.99,gamma=0.99,alpha=0.99):  ##exp epsilon & decay
        self.mode_dec = inital_mode_dec
        self.inital_mode_dec = inital_mode_dec
        # self.pruned_areas=set()
        self.epsilon = epsilon
        self.decay = decay
        self.gamma=gamma
        self.alpha=alpha
        self.rewards = {key: 0 for key in inital_mode_dec.keys()}
        self.fails = {key: 0 for key in inital_mode_dec.keys()}
        self.history = []  ##not sure
        self.iterations = 0  ##
        self.max_iterations = 100  #
        self.states = []
        self.state_rewards = []
        self.training = True
        self.spec = []  ##
        self.q_table = {}
        self.actions = ["add_c", "add_p", "add_r", "add_l", "rmv_l", "ilasp"]  # ilasp?

        self.oracle = SpecOracle()
        self.spec_learner = SpecLearner(self)
        # print("STARTQTABLE:", self.q_table)

        ##to ensure its not missing
        if "maxv" not in self.mode_dec:
            self.mode_dec["maxv"] = "#maxv(2)."

    def extract_features(self, spec, trace):
        num_ass = sum(1 for line in spec if "assumption" in line)
        num_guar = sum(1 for line in spec if "guarantee" in line)
        num_temp_operators = sum(
            1
            for line in spec
            if any(op in line for op in ["next", "prev", "eventually"])
        )
        num_violated = len(trace)
        spec_length = len(spec)

        feature_vec = np.array(
            [num_ass, num_guar, num_temp_operators, num_violated, spec_length]
        )
        return feature_vec

    def select_action(self, state):
        # print("actQTABLE:", self.q_table)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.best_action(state)

    def best_action(self, state):
        state_str = str(state)
        if state_str in self.q_table:
            print("FROMTABLE")
            return max(self.q_table[state_str], key=self.q_table[state_str].get)
        else:
            print("RANDOM")
            return np.random.choice(self.actions)

    def best_hypo(self, hypo):
        return max(hypo, key=lambda hyp: self.q_table.get(str(hyp), 0))

    def update_policy(self, state, action, reward, next_state):
        state_str = str(state)
        next_str = str(next_state)
        if state_str not in self.q_table:
            self.q_table[state_str] = {a: 0 for a in self.actions}

        if next_str not in self.q_table:
            self.q_table[next_str] = {a: 0 for a in self.actions}

        best_next_action = self.best_action(next_state)
        q_update = reward + self.gamma * self.q_table[next_str].get(best_next_action, 0)
        self.q_table[state_str][action] = (1 - self.alpha) * self.q_table[state_str][
            action
        ] + self.alpha * q_update
        # q_update = reward + self.decay * self.q_table[next_str].get(best_next_action, 0)
        # self.q_table[state_str][action] = (1 - self.decay) * self.q_table[state_str][
        #     action
        # ] + self.decay * q_update
        # .get(action, 0)

    def train(self, spec, trace):
        state = self.extract_features(spec, trace)
        while True:
            action = self.select_action(state)
            # modified_spec = self.apply_action(action, spec)
            print("action", action)
            if action == "ilasp":
                cs_traces = []
                modified_spec = self.spec_learner.learn_weaker_spec(
                    spec, trace, cs_traces, Learning.ASSUMPTION_WEAKENING
                )
            else:
                modified_spec = self.apply_action(action, spec)
            cs = self.oracle.synthesise_and_check(modified_spec)
            # print("CS:", cs)
            if not cs:
                feedback = "realisable"
            else:
                feedback = "counter_strategy_found"
            reward = self.get_reward(feedback)
            next_state = self.extract_features(modified_spec, trace)
            self.update_policy(state, action, reward, next_state)
            # ("QTABLE:", self.q_table)
            state = next_state
            if feedback == "realisable" or self.iterations >= self.max_iterations:
                break
            self.iterations += 1
        self.epsilon *= self.decay

    def test(self, spec, trace):
        self.iterations = 0  ##
        state = self.extract_features(spec, trace)
        while True:
            action = self.select_action(state)

            # modified_spec = self.apply_action(action, spec)
            print("action", action)
            if action == "ilasp":
                cs_traces = []
                modified_spec = self.spec_learner.learn_weaker_spec(
                    spec, trace, cs_traces, Learning.ASSUMPTION_WEAKENING
                )
            else:
                modified_spec = self.apply_action(action, spec)
            cs = self.oracle.synthesise_and_check(modified_spec)
            # print("CS:", cs)
            if not cs:
                feedback = "realisable"
            else:
                feedback = "counter_strategy_found"
            next_state = self.extract_features(modified_spec, trace)
            # self.update_policy(state, action, reward, next_state)
            # print("QTABLE:", self.q_table)
            state = next_state
            if feedback == "realisable" or self.iterations >= self.max_iterations:
                break
            self.iterations += 1

            # self.epsilon *= self.decay
        # return  total_reward

    def get_reward(self, feedback):
        if feedback == "counter_strategy_found":  ##unrealisable
            return -1
        elif feedback == "realisable":
            return 10
        else:
            return 0  ##no need?

    def loss(self, state, action, new_spec, trace):
        state_str = str(state)
        next_state = self.extract_features(new_spec, trace)
        next_str = str(next_state)
        cs = self.oracle.synthesise_and_check(new_spec)
        if not cs:
            feedback = "realisable"
        else:
            feedback = "counter_strategy_found"
        reward = self.get_reward(feedback)
        if next_str in self.q_table:
            best_action = self.best_action(next_state)
            target = reward + self.decay * self.q_table[next_str].get(best_action, 0)
        else:
            target = reward

        current_q = self.q_table[state_str].get(action, 0)
        loss = abs(current_q - (target))
        return loss

    def apply_action(self, action, spec):
        if action == "add_c":
            return self.add_c(spec)
        elif action == "add_p":
            return self.add_p(spec)
        elif action == "add_r":
            return self.add_r(spec)
        elif action == "add_l":
            return self.add_l(spec)  ##inc maxv
        elif action == "rmv_l":
            return self.rmv_l(spec)  ##dec maxv
        return spec

    def get_mode_dec(self):
        # return self.mode_dec
        return "\n".join(self.mode_dec.values())  ##

    def update_with_spec(self, spec):
        # for key, value in spec.items():
        #     if key not in self.mode_dec:
        #         self.mode_dec[key] = value
        #         self.fails[key] = 0
        self.mode_dec = copy.deepcopy(self.inital_mode_dec)
        self.spec = copy.deepcopy(list(spec))  ##
        self.rewards = {key: 0 for key in self.inital_mode_dec.keys()}
        self.fails = {key: 0 for key in self.inital_mode_dec.keys()}
        self.iterations = 0
        # for key, value in spec.items():
        #     self.mode_dec[key] = value
        #     self.fails[key] = 0

    def update_mode_dec(self, feedback):
        if self.training:
            #     self.update_rewards(self.get_reward(feedback))  #

            if feedback == "counter_strategy_found":  ##unrealisable
                self.prune_mode_dec()
                # self.expand_mode_dec()
            elif feedback == "realisable":
                print("Realisable specification found.")
                return "realisable"  # True  #

            if random.uniform(0, 1) < self.epsilon:
                self.expand_mode_dec()
            else:
                self.expand_best()  ##not sure

            self.epsilon *= self.decay
            self.iterations += 1
            # print(f"Iteration: {self.iterations}, Mode Decleration: {self.mode_dec}")

            self.history.append(copy.deepcopy(self.mode_dec))  ##
            self.states.append(copy.deepcopy(self.mode_dec))
            self.state_rewards.append(self.rewards.copy())
            # ("HISTORY", self.history)
            if self.iterations >= self.max_iterations:
                print("Maximum iterations reached.")
                return "max"

        # if len(self.history) > 10:
        #     if self.converged():
        #         print("Converged!")
        #         return "converged"

        return "continue"  # False  #

    def update_rewards(self, reward):  ##
        for key in self.mode_dec.keys():
            self.rewards[key] = self.decay * self.rewards.get(key, 0) + reward

    def prune_mode_dec(self):  ##
        check = [key for key in self.fails.keys() if key != "maxv"]
        if check:
            key = max(check, key=lambda k: self.fails[k])  # key=self.fails.get)
            self.mode_dec.pop(key, None)
            self.history.append(copy.deepcopy(self.mode_dec))
            self.states.append(copy.deepcopy(self.mode_dec))
            self.state_rewards.append(self.rewards.copy())
            print(f"Pruned moode dec: {key}")

    def expand_mode_dec(self):
        possibilities = [
            self.add_c,
            self.add_p,
            self.add_r,
            self.add_l,
            self.rmv_l,
            # self.add_maxbody,
        ]  #
        change = random.choice(possibilities)
        change(self.spec)
        print(f"expanded mode dec")

    def expand_best(self):
        possibilities = [
            self.add_c,
            self.add_p,
            self.add_r,
            self.add_l,
            self.rmv_l,
            # self.add_maxbody,
        ]  #
        best_change = None
        best_reward = float("-inf")

        for change in possibilities:  # search for best in all possibilities
            temp = copy.deepcopy(self)  ###
            change_func = getattr(temp, change.__name__)
            change_func(temp.spec)
            reward = self.evaluate(temp)
            if reward > best_reward:
                best_reward = reward
                best_change = change

        if best_change:  ##found
            best_change(self.spec)
            print(f"best expansion")
        else:  ##random
            self.expand_mode_dec()

    def evaluate(self, temp):
        # pass  ##TODO: write method
        return sum(temp.rewards.values())  ##improve

    def add_c(self, spec):  ##ideally no
        # if agent is None:  # do for all?
        #     agent = self  #
        new_c = f"#constant(t2,c{len(spec)+1})."
        # key = f"constant_c{len(self.mode_dec)+1}"
        if new_c not in spec:  ##
            spec.append(new_c)
            # self.mode_dec[key] = new_c
            # self.fails[key] = 0
            # # self.history.append(f"Added constant: {new_c}")
            # self.history.append(copy.deepcopy(self.mode_dec))
            # self.states.append(copy.deepcopy(self.mode_dec))
            # self.state_rewards.append(self.rewards.copy())
        return spec

    def add_p(self, spec):
        new_p = f"#modeh(new_pred_{len(spec)}(var(t1))))."
        # key = f"constant_c{len(self.mode_dec)+1}"
        if new_p not in spec:  ##
            spec.append(new_p)
        # new_p = f"#modeh(new_pred_{len(self.mode_dec)}(var(t1))))."
        # key = f"new_pred_{len(self.mode_dec)}"
        # if key not in self.mode_dec:  ##
        #     self.mode_dec[key] = new_p
        #     self.fails[key] = 0
        #     self.history.append(copy.deepcopy(self.mode_dec))
        #     self.states.append(copy.deepcopy(self.mode_dec))
        #     self.state_rewards.append(self.rewards.copy())
        return spec

    def add_r(self, spec):
        new_r = f"#modeb(new_body_pred_{len(spec)}(var(t1))))."  ##body?rule?
        if new_r not in spec:  ##
            spec.append(new_r)
        # key = f"new_body_pred_{len(self.mode_dec)}"  ##body?rule?
        # if key not in self.mode_dec:  ##
        #     self.mode_dec[key] = new_r
        #     self.fails[key] = 0
        #     self.history.append(copy.deepcopy(self.mode_dec))
        #     self.states.append(copy.deepcopy(self.mode_dec))
        #     self.state_rewards.append(self.rewards.copy())
        return spec

    def add_l(self, spec):

        # current_maxv = int(re.search(r"#maxv\((\d+)\)", self.mode_dec["maxv"]).group(1))
        # self.mode_dec["maxv"] = f"#maxv({current_maxv+1})."
        # self.history.append(copy.deepcopy(self.mode_dec))
        # self.states.append(copy.deepcopy(self.mode_dec))
        # self.state_rewards.append(self.rewards.copy())

        maxv_pattern = re.compile(r"#maxv\((\d+)\).")
        for i, line in enumerate(spec):
            match = maxv_pattern.match(line)
            if match:
                current_maxv = int(match.group(1))
                spec[i] = f"#maxv({current_maxv+1})."
                break
        return spec

    def rmv_l(self, spec):

        # current_maxv = int(re.search(r"#maxv\((\d+)\)", self.mode_dec["maxv"]).group(1))
        # if current_maxv > 1:
        #     self.mode_dec["maxv"] = f"#maxv({current_maxv-1})."
        #     self.history.append(copy.deepcopy(self.mode_dec))
        #     self.states.append(copy.deepcopy(self.mode_dec))  ##no need
        #     self.state_rewards.append(self.rewards.copy())
        maxv_pattern = re.compile(r"#maxv\((\d+)\).")
        for i, line in enumerate(spec):
            match = maxv_pattern.match(line)
            if match:
                current_maxv = int(match.group(1))
                if current_maxv > 1:
                    spec[i] = f"#maxv({current_maxv-1})."
                break
        return spec

    # def add_maxbody(self):
    #     if "maxbody" in self.mode_dec:
    #         current_maxbody = int(
    #             re.search(r"#maxbody\((\d+)\)", self.mode_dec["maxbody"]).group(1))
    #         self.mode_dec["maxbody"] = f"#maxbody({current_maxbody+1})."
    #     else:
    #         self.mode_dec["maxbody"] = f"#maxbody(3)."  # 3? or change..
    #     self.history.append(f"Increased maxbody to {self.mode_dec['maxbody']}")

    def converged(self):
        recent = self.history[-10:]  # last 10
        # recent = set(recent)
        # # recent=[frozenset(self.history[i].items()) for i in range(-10,0)]
        # if len(recent) <= 1:  ##so all are same
        #     return True  # converged to this mode dec
        # return False
        # recent = set(frozenset(d.items()) for d in recent)
        unique = set()
        for i in recent:
            unique.update(i.items())
        return len(unique) <= len(recent)  #
        # if len(recent) <= 1:
        #     return True
        # return False
