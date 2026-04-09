from fastapi import FastAPI
from env import HospitalEnv

app = FastAPI()

# Global environment
env = HospitalEnv()

# ✅ ROOT (fixes "Not Found")
@app.get("/")
def root():
    return {"message": "Hospital RL API is running"}

# ✅ RESET endpoint (REQUIRED)
@app.post("/reset")
def reset():
    global env
    env = HospitalEnv()
    state = env.reset()
    return {
        "state": state
    }

# ✅ STEP endpoint (REQUIRED)
@app.post("/step")
def step(action: dict):
    """
    Expected input:
    {
        "actions": [(patient_id, action), ...]
    }
    """

    actions = action.get("actions", [])

    next_state, reward, done, _ = env.step(actions)

    return {
        "state": next_state,
        "reward": reward,
        "done": done
    }