import random
from reward import calculate_reward

class HospitalEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.icu_beds = 3
        self.doctors = 2
        self.ventilators = 1

        # ✅ FIX: controlled patients (not too random)
        self.patients = [
            {"id": 1, "severity": "critical", "needs_ventilator": True, "waiting_time": 0},
            {"id": 2, "severity": "medium", "needs_ventilator": False, "waiting_time": 0},
            {"id": 3, "severity": "low", "needs_ventilator": False, "waiting_time": 0},
        ]

        self.time = 0
        return self.get_state()

    def get_state(self):
        return {
            "icu_beds": self.icu_beds,
            "doctors": self.doctors,
            "ventilators": self.ventilators,
            "patients": self.patients
        }

    def step(self, actions):
        results = []

        for pid, action in actions:
            patient = next(p for p in self.patients if p["id"] == pid)

            if action == "icu" and self.icu_beds > 0:
                self.icu_beds -= 1
                results.append((patient, "treated"))

            elif action == "doctor" and self.doctors > 0:
                self.doctors -= 1
                results.append((patient, "treated"))

            elif action == "ventilator" and self.ventilators > 0:
                self.ventilators -= 1
                results.append((patient, "treated"))

            else:
                patient["waiting_time"] += 1
                results.append((patient, "waiting"))

        reward = calculate_reward(results)

        self.time += 1
        done = self.time > 5  # shorter episodes

        return self.get_state(), reward, done, results