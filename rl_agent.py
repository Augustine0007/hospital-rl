import random

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

        self.actions = ["icu", "doctor", "ventilator", "wait"]

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0      # start high
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05

    def get_state_key(self, state, patient):
        return (
            state["icu_beds"],
            state["doctors"],
            state["ventilators"],
            patient["severity"],
            patient["needs_ventilator"]
        )

    def choose_action(self, state, patient):
        state_key = self.get_state_key(state, patient)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in self.actions}

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, state, patient, action, reward, next_state):
        state_key = self.get_state_key(state, patient)
        next_key = self.get_state_key(next_state, patient)

        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0 for a in self.actions}

        old_value = self.q_table[state_key][action]
        next_max = max(self.q_table[next_key].values())

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        self.q_table[state_key][action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay