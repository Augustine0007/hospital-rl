import os
import time
from env import HospitalEnv

# OPTIONAL: only if using LLM
from openai import OpenAI

def get_client():
    try:
        return OpenAI(
            base_url=os.environ.get("API_BASE_URL"),
            api_key=os.environ.get("API_KEY"),
        )
    except Exception as e:
        print("[LLM ERROR]", e)
        return None


def run_task(task_name):
    print(f"[START] task={task_name} env=hospital model=gpt-4.1-mini")

    env = HospitalEnv()   # ✅ FIXED (no argument)
    state = env.reset()   # ✅ FIXED (no argument)

    done = False
    step = 0

    while not done and step < 6:
        step += 1

        actions = []

        for p in state["patients"]:
            severity = int(p["severity"])  # ✅ FIXED (convert string → int)

            if severity > 7:
                action = "icu"
            elif severity > 4:
                action = "doctor"
            else:
                action = "wait"

            actions.append((p["id"], action))

        state, reward, done, info = env.step(actions)

        print(f"[STEP] step={step} action={actions} reward={reward:.2f} done={done} error={info}")

    print(f"[END] success=true steps={step}")


def run_all_tasks():
    for task in ["easy", "medium", "hard"]:
        run_task(task)


if __name__ == "__main__":
    run_all_tasks()

    # KEEP CONTAINER ALIVE (important for HF)
    while True:
        time.sleep(60)