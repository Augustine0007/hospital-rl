# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from env import HospitalEnv

app = FastAPI()

# Global environment
env = HospitalEnv()


# Request format for /step
class ActionRequest(BaseModel):
    action: list


# RESET endpoint
@app.post("/reset")
def reset(task: str = "easy"):
    state = env.reset(task)
    return {"state": state}


# STEP endpoint
@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, info = env.step(req.action)
    return {
        "state": state,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }


# STATE endpoint
@app.get("/state")
def state():
    return {"state": env.state()}