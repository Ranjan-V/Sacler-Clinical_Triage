from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Action, ResetRequest, ResetResponse, StepResult
from environment import ClinicalTriageEnvironment
from graders import run_grader
from typing import Dict, Any

app = FastAPI(
    title="Clinical Triage OpenEnv",
    description="OpenEnv-compliant clinical triage environment for AI agent training",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance — persists across all requests
env = ClinicalTriageEnvironment()


@app.get("/")
def root():
    return {
        "name": "clinical-triage-coordinator",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest = None) -> ResetResponse:
    try:
        if request is None:
            request = ResetRequest()
        observation = env.reset(task_id=request.task_id)
        return ResetResponse(
            observation=observation,
            task_id=request.task_id,
            episode_id=observation.get("episode_id", ""),
            message=f"Environment reset for task: {request.task_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: Action) -> StepResult:
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call reset() first.")
    try:
        result = env.step(action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call reset() first.")
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade")
def grade() -> Dict[str, Any]:
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call reset() first.")
    try:
        current_state = env.get_state()
        task_id = current_state.get("task_id", "task_easy")
        score = run_grader(task_id, current_state)
        # Strictly enforce (0, 1) — never 0.0 or 1.0
        score = round(max(0.01, min(0.99, float(score))), 4)
        return {
            "task_id": task_id,
            "score": score,
            "episode_id": current_state.get("episode_id"),
            "done": current_state.get("done"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)