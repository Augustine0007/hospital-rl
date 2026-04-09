import os

from rl_agent import QLearningAgent
from env import HospitalEnv

# ✅ Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "test")  # default for local




# 🔁 RL TRAINING
def train_agent(agent):
    env = HospitalEnv()

    for _ in range(50):
        state = env.reset()
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

        agent.decay_epsilon()

    return agent


# 🚀 RUN TASK
def run_task(task_name):
    env = HospitalEnv()
    agent = QLearningAgent()

    # ✅ Train agent (RL happens here)
    agent = train_agent(agent)

    state = env.reset()
    done = False

    step_count = 0
    rewards = []

    # ✅ START
    print(f"[START] task={task_name} env=hospital model={MODEL_NAME}")

    try:
        while not done:
            actions = []

            for p in state["patients"]:
                action = agent.choose_action(state, p)
                actions.append((p["id"], action))

            next_state, reward, done, _ = env.step(actions)

            step_count += 1
            rewards.append(f"{reward:.2f}")

            action_str = str(actions)

            # ✅ STEP
            print(
                f"[STEP] step={step_count} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            state = next_state

        success = True

    except Exception as e:
        success = False
        print(
            f"[STEP] step={step_count} action=null "
            f"reward=0.00 done=true error={str(e)}"
        )

    # ✅ FINAL SCORE (0 → 1)
    total_reward = sum([float(r) for r in rewards])

    min_r = -300
    max_r = 150

    score = (total_reward - min_r) / (max_r - min_r)
    score = max(0.0, min(1.0, score))

    # ✅ END
    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} rewards={','.join(rewards)} score={score:.2f}"
    )


# 🎯 MAIN
if __name__ == "__main__":
    run_task("easy")
    run_task("medium")
    run_task("hard")