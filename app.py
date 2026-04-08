import gradio as gr
from rl_agent import QLearningAgent
from env import HospitalEnv

env = HospitalEnv()
agent = QLearningAgent()

def train():
    episodes = 100
    logs = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = []

            for p in state["patients"]:
                action = agent.choose_action(state, p)
                actions.append((p["id"], action))

            next_state, reward, done, results = env.step(actions)

            for ((pid, action), (patient, status)) in zip(actions, results):
                if status == "treated":
                    if patient["severity"] == "critical":
                        r = 40
                    elif patient["severity"] == "medium":
                        r = 20
                    else:
                        r = 10
                else:
                    if patient["severity"] == "critical":
                        r = -30
                    elif patient["severity"] == "medium":
                        r = -15
                    else:
                        r = -5

                agent.update(state, patient, action, r, next_state)

            state = next_state
            total_reward += reward

        # ✅ decay exploration
        agent.decay_epsilon()

        logs.append(f"Episode {ep}: {total_reward}")

    return "\n".join(logs)

ui = gr.Interface(
    fn=train,
    inputs=[],
    outputs="text",
    title="🏥 Hospital RL Training"
)

if __name__ == "__main__":
    ui.launch()