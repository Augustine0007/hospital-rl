from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Dummy state (replace with your env logic)
state = {"step": 0}

class ActionRequest(BaseModel):
    actions: list

@app.get("/")
def root():
    return {"message": "Hospital RL Environment Running"}

@app.post("/reset")
def reset():
    global state
    state = {"step": 0}
    return {"state": state}

@app.post("/step")
def step(req: ActionRequest):
    global state
    state["step"] += 1

    return {
        "state": state,
        "reward": 1.0,
        "done": state["step"] >= 5
    }

# 🔥 REQUIRED MAIN FUNCTION
def main():
    return app

# 🔥 REQUIRED ENTRY CHECK
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)