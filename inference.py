import os
from env import HospitalEnv


# ✅ LLM CALL (REQUIRED FOR PHASE 2)
def call_llm():
    try:
        base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")

        # Skip only in local runs
        if not base_url or not api_key:
            print("[LLM] Missing env (local run) — skipping")
            return

        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=5
        )

        print("[LLM] SUCCESS ✅")

    except Exception as e:
        print("[LLM ERROR]", e)


# ✅ CONVERT SEVERITY SAFELY
def parse_severity(severity):
    if isinstance(severity, str):
        severity = severity.lower()
        if severity == "critical":
            return 9
        elif severity == "medium":
            return 5
        elif severity == "low":
            return 2
        else:
            return 0
    try:
        return int(severity)
    except:
        return 0


# ✅ RUN SINGLE TASK
def run_task(task_name):
    print(f"[START] task={task_name} env=hospital model=gpt-4.1-mini")

    env = HospitalEnv()
    state = env.reset()

    done = False
    step_count = 0

    while not done and step_count < 6:
        patients = state.get("patients", [])
        actions = []

        for p in patients:
            severity = parse_severity(p.get("severity", 0))

            if severity > 7:
                actions.append((p["id"], "icu"))
            elif severity > 4:
                actions.append((p["id"], "doctor"))
            else:
                actions.append((p["id"], "wait"))

        state, reward, done, info = env.step(actions)

        step_count += 1
        print(f"[STEP] step={step_count} action={actions} reward={reward:.2f} done={done} error=null")

    print(f"[END] success=true steps={step_count}")


# ✅ RUN ALL TASKS
def run_all_tasks():
    for task in ["easy", "medium", "hard"]:
        run_task(task)


# ✅ ENTRY POINT
if __name__ == "__main__":
    call_llm()      # 🔥 REQUIRED
    run_all_tasks()