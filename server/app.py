from fastapi import FastAPI
from pydantic import BaseModel
from env import HospitalEnv

app = FastAPI()

env = HospitalEnv()


class ResetRequest(BaseModel):
    task: str


@app.get("/")
def home():
    return {"message": "Hospital RL API running"}


@app.post("/reset")
def reset(req: ResetRequest):
    state = env.reset()
    return {"state": state}


@app.post("/step")
def step(actions: list):
    state, reward, done, info = env.step(actions)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ REQUIRED for OpenEnv validator
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()