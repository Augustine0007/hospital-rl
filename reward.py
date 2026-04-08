def calculate_reward(results):
    reward = 0

    for patient, status in results:
        severity = patient["severity"]
        wait = patient["waiting_time"]

        # 🎯 Priority-based rewards
        if status == "treated":
            if severity == "critical":
                reward += 20
            elif severity == "medium":
                reward += 10
            else:
                reward += 5

        # ⛔ Penalty for waiting
        if status == "waiting":
            if severity == "critical":
                reward -= 15
            elif severity == "medium":
                reward -= 8
            else:
                reward -= 3

        # ⏳ Extra penalty for long wait
        if wait > 2:
            reward -= wait * 2

    # ⚖️ Efficiency bonus
    treated_count = sum(1 for _, s in results if s == "treated")
    reward += treated_count * 2

    return reward