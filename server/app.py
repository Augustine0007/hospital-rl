from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

# -------- Request schema --------
class ResetRequest(BaseModel):
    task: str


class StepRequest(BaseModel):
    action: list


# -------- Dummy environment (replace with your logic if needed) --------
class HospitalEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {
            "patients": [
                {"id": 1, "severity": 8},
                {"id": 2, "severity": 5},
                {"id": 3, "severity": 3},
            ]
        }

    def step(self, action):
        self.step_count += 1

        reward = 10 - self.step_count * 5
        done = self.step_count >= 6

        return {
            "state": {
                "patients": [
                    {"id": 1, "severity": 7},
                    {"id": 2, "severity": 4},
                    {"id": 3, "severity": 2},
                ]
            },
            "reward": reward,
            "done": done,
        }


env = HospitalEnv()


# -------- ROUTES --------

@app.get("/")
def home():
    return {"message": "Hospital RL API is running 🚀"}


@app.post("/reset")
def reset(req: ResetRequest):
    state = env.reset()
    return state


@app.post("/step")
def step(req: StepRequest):
    result = env.step(req.action)
    return result


# -------- MAIN ENTRY (VERY IMPORTANT) --------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()