from fastapi import FastAPI
from pydantic import BaseModel
from env import HospitalEnv

app = FastAPI()
env = HospitalEnv()


# ✅ ONLY for /step (needs body)
class ActionRequest(BaseModel):
    actions: list


@app.get("/")
def home():
    return {"message": "Hospital RL API running"}


# 🔥 IMPORTANT: NO BODY HERE
@app.post("/reset")
def reset():
    state = env.reset()
    return state


# ✅ STEP uses body
@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, info = env.step(req.actions)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ REQUIRED ENTRYPOINT
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()