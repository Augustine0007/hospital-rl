from env import HospitalEnv


# 🎯 Normalize reward → score (0.0 to 1.0)
def normalize_score(total_reward, min_r=-150, max_r=150):
    score = (total_reward - min_r) / (max_r - min_r)
    return max(0.0, min(1.0, score))


# 🟢 EASY TASK (low pressure)
def easy_task(agent):
    env = HospitalEnv()

    env.icu_beds = 3
    env.doctors = 2
    env.ventilators = 1

    env.patients = [
        {"id": 1, "severity": "medium", "needs_ventilator": False, "waiting_time": 0},
        {"id": 2, "severity": "low", "needs_ventilator": False, "waiting_time": 0},
    ]

    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        actions = []

        for p in state["patients"]:
            action = agent.choose_action(state, p)
            actions.append((p["id"], action))

        state, reward, done, results = env.step(actions)
        total_reward += reward

    return normalize_score(total_reward)


# 🟡 MEDIUM TASK (balanced difficulty)
def medium_task(agent):
    env = HospitalEnv()

    env.icu_beds = 2
    env.doctors = 2
    env.ventilators = 1

    env.patients = [
        {"id": 1, "severity": "critical", "needs_ventilator": True, "waiting_time": 0},
        {"id": 2, "severity": "medium", "needs_ventilator": False, "waiting_time": 0},
        {"id": 3, "severity": "low", "needs_ventilator": False, "waiting_time": 0},
    ]

    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        actions = []

        for p in state["patients"]:
            action = agent.choose_action(state, p)
            actions.append((p["id"], action))

        state, reward, done, results = env.step(actions)
        total_reward += reward

    return normalize_score(total_reward)


# 🔴 HARD TASK (resource pressure, but solvable)
def hard_task(agent):
    env = HospitalEnv()

    env.icu_beds = 2
    env.doctors = 1
    env.ventilators = 1

    env.patients = [
        {"id": 1, "severity": "critical", "needs_ventilator": True, "waiting_time": 0},
        {"id": 2, "severity": "critical", "needs_ventilator": False, "waiting_time": 0},
        {"id": 3, "severity": "medium", "needs_ventilator": False, "waiting_time": 0},
    ]

    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        actions = []

        for p in state["patients"]:
            action = agent.choose_action(state, p)
            actions.append((p["id"], action))

        state, reward, done, results = env.step(actions)
        total_reward += reward

    return normalize_score(total_reward)