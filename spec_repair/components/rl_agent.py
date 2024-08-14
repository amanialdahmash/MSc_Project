# AD
import random
import copy  #
import re


class RLAgent:
    def __init__(self, inital_mode_dec, epsilon=0.1, decay=0.99):  ##exp epsilon & decay
        self.mode_dec = inital_mode_dec
        # self.pruned_areas=set()
        self.epsilon = epsilon
        self.decay = decay
        self.rewards = {key: 0 for key in inital_mode_dec.keys()}
        self.fails = {key: 0 for key in inital_mode_dec.keys()}
        self.history = []  ##not sure
        self.iterations = 0  ##
        self.max_iterations = 100  #
        self.states = []
        self.state_rewards = []

        ##to ensure its not missing
        if "maxv" not in self.mode_dec:
            self.mode_dec["maxv"] = "#maxv(2)."

    def get_mode_dec(self):
        # return self.mode_dec
        return "\n".join(self.mode_dec.values())  ##

    def update_with_spec(self, spec):
        for key, value in spec.items():
            if key not in self.mode_dec:
                self.mode_dec[key] = value
                self.fails[key] = 0

    def update_mode_dec(self, feedback):
        self.update_rewards(self.get_reward(feedback))  #

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
        print(f"Iteration: {self.iterations}, Mode Decleration: {self.mode_dec}")

        self.history.append(copy.deepcopy(self.mode_dec))  ##
        self.states.append(copy.deepcopy(self.mode_dec))
        self.state_rewards.append(self.rewards.copy())
        print("HISTORY", self.history)
        if self.iterations >= self.max_iterations:
            print("Maximum iterations reached.")
            return "max"

        # if len(self.history) > 10:
        #     if self.converged():
        #         print("Converged!")
        #         return "converged"

        return "continue"  # False  #

    def get_reward(self, feedback):
        if feedback == "counter_strategy_found":  ##unrealisable
            return -1
        elif feedback == "realisable":
            return 10
        else:
            return 0  ##no need?

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
        change()
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
            change_func()
            reward = self.evaluate(temp)
            if reward > best_reward:
                best_reward = reward
                best_change = change

        if best_change:  ##found
            best_change()
            print(f"best expansion")
        else:  ##random
            self.expand_mode_dec()

    def evaluate(self, temp):
        # pass  ##TODO: write method
        return sum(temp.rewards.values())  ##improve

    def add_c(self):  ##ideally no
        # if agent is None:  # do for all?
        #     agent = self  #
        new_c = f"#constant(t2,c{len(self.mode_dec)+1})."
        key = f"constant_c{len(self.mode_dec)+1}"
        if key not in self.mode_dec:  ##
            self.mode_dec[key] = new_c
            self.fails[key] = 0
            # self.history.append(f"Added constant: {new_c}")
            self.history.append(copy.deepcopy(self.mode_dec))
            self.states.append(copy.deepcopy(self.mode_dec))
            self.state_rewards.append(self.rewards.copy())

    def add_p(self):
        new_p = f"#modeh(new_pred_{len(self.mode_dec)}(var(t1))))."
        key = f"new_pred_{len(self.mode_dec)}"
        if key not in self.mode_dec:  ##
            self.mode_dec[key] = new_p
            self.fails[key] = 0
            self.history.append(copy.deepcopy(self.mode_dec))
            self.states.append(copy.deepcopy(self.mode_dec))
            self.state_rewards.append(self.rewards.copy())

    def add_r(self):
        new_r = f"#modeb(new_body_pred_{len(self.mode_dec)}(var(t1))))."  ##body?rule?
        key = f"new_body_pred_{len(self.mode_dec)}"  ##body?rule?
        if key not in self.mode_dec:  ##
            self.mode_dec[key] = new_r
            self.fails[key] = 0
            self.history.append(copy.deepcopy(self.mode_dec))
            self.states.append(copy.deepcopy(self.mode_dec))
            self.state_rewards.append(self.rewards.copy())

    def add_l(self):
        current_maxv = int(re.search(r"#maxv\((\d+)\)", self.mode_dec["maxv"]).group(1))
        self.mode_dec["maxv"] = f"#maxv({current_maxv+1})."
        self.history.append(copy.deepcopy(self.mode_dec))
        self.states.append(copy.deepcopy(self.mode_dec))
        self.state_rewards.append(self.rewards.copy())

    def rmv_l(self):
        current_maxv = int(re.search(r"#maxv\((\d+)\)", self.mode_dec["maxv"]).group(1))
        if current_maxv > 1:
            self.mode_dec["maxv"] = f"#maxv({current_maxv-1})."
            self.history.append(copy.deepcopy(self.mode_dec))
            self.states.append(copy.deepcopy(self.mode_dec))  ##no need
            self.state_rewards.append(self.rewards.copy())

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
