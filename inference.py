import os
import json
import time
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

TASKS = ["task_easy", "task_medium", "task_hard"]

# Maximum reward achievable per step (used to normalise the running total)
MAX_STEP_REWARD = 0.99


def _norm(value: float, max_possible: float) -> float:
    """Normalise an accumulated reward into the strictly-open interval (0.01, 0.99)."""
    if max_possible <= 0:
        return 0.01
    return round(max(0.01, min(0.99, float(value) / float(max_possible))), 4)


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        with httpx.Client(timeout=60) as http:
            if method == "POST":
                r = http.post(url, json=payload or {})
            else:
                r = http.get(url)
            if not r.is_success:
                print(f"[DEBUG] {r.status_code} error on {endpoint}: {r.text}", flush=True)
                return {"error": r.text, "reward": 0.01, "done": False}
            return r.json()
    except Exception as e:
        print(f"[DEBUG] network exception on {endpoint}: {e}", flush=True)
        return {"error": str(e), "reward": 0.01, "done": False}


def build_prompt(observation: dict, action_history: list) -> str:
    patients = observation.get("patients", [])
    beds = observation.get("available_beds", 0)
    resources = observation.get("resources", {})
    step = observation.get("step_count", 0)
    max_steps = observation.get("max_steps", 10)

    unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]
    assigned = [p for p in patients if p.get("current_priority") != "unassigned"]

    patient_text = ""
    for p in unassigned:
        v = p["vitals"]
        patient_text += (
            f"\n  [{p['patient_id']}] Age:{p['age']} {p['gender']}"
            f" | Complaint: {p['chief_complaint']}"
            f"\n  HR:{v['heart_rate']} BP:{v['blood_pressure_systolic']}/{v['blood_pressure_diastolic']}"
            f" RR:{v['respiratory_rate']} SpO2:{v['oxygen_saturation']}%"
            f" Temp:{v['temperature']}C Pain:{v['pain_score']}/10"
            f"\n  History: {', '.join(p['medical_history']) or 'None'}"
            f"\n  Diagnostics: {', '.join(p['diagnostics_ordered']) or 'None'}"
            f"\n  Admitted: {p['admitted']}\n"
        )

    already_done = ", ".join(
        [f"{p['patient_id']}=P{p['current_priority']}" for p in assigned]
    ) or "None"

    history_text = ""
    if action_history:
        last_3 = action_history[-3:]
        history_text = "Last actions: " + " | ".join(
            [f"{a['patient_id']}:{a['action_type']}={a['value']}" for a in last_3]
        )

    return f"""You are an expert ED triage nurse. Choose ONE action.

Step:{step}/{max_steps} | Beds:{beds} | Resources:{json.dumps(resources)}
Already handled: {already_done}
{history_text}

PATIENTS NEEDING ACTION:
{patient_text if patient_text else 'All triaged. Admit the most critical non-admitted patient.'}

ESI: 1=Life-threat 2=Emergent 3=Urgent 4=Less-urgent 5=Non-urgent

CRITICAL RULES:
- NEVER assign priority to already-handled patients
- Focus on unassigned patients first
- Pick the most critically ill patient

Respond ONLY with this exact JSON format:
{{"patient_id": "P001", "action_type": "assign_priority", "value": "1"}}"""


def get_agent_action(observation: dict, action_history: list) -> dict:
    prompt = build_prompt(observation, action_history)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=80,
    )
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    # Extract JSON safely
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]

    action = json.loads(content)

    # Validate and fix action
    valid_types = [
        "assign_priority", "order_diagnostic",
        "admit_patient", "discharge_patient", "reassess"
    ]
    if action.get("action_type") not in valid_types:
        action["action_type"] = "assign_priority"

    patients = observation.get("patients", [])
    unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]

    # Force agent to pick unassigned patient
    assigned_ids = [p["patient_id"] for p in patients if p.get("current_priority") != "unassigned"]
    if action.get("patient_id") in assigned_ids and unassigned:
        action["patient_id"] = unassigned[0]["patient_id"]

    if "patient_id" not in action and unassigned:
        action["patient_id"] = unassigned[0]["patient_id"]

    if "value" not in action or not action["value"]:
        action["value"] = "3"

    return action


def run_task(task_id: str) -> dict:
    """Wrapper that guarantees run_task never crashes the process."""
    print(f"[START] task_id={task_id} timestamp={time.time()}", flush=True)
    try:
        return _run_task_inner(task_id)
    except Exception as e:
        print(
            f"[END] task_id={task_id} episode_id=error total_steps=0"
            f" total_reward=0.01 final_score=0.01 exception={e}",
            flush=True
        )
        return {"task_id": task_id, "final_score": 0.01, "total_reward": 0.01}


def _run_task_inner(task_id: str) -> dict:
    # Reset environment
    reset_response = call_env("POST", "/reset", {"task_id": task_id})
    if "error" in reset_response:
        print(
            f"[END] task_id={task_id} episode_id=error total_steps=0"
            f" total_reward=0.01 final_score=0.01",
            flush=True
        )
        return {"task_id": task_id, "final_score": 0.01, "total_reward": 0.01}

    observation = reset_response.get("observation", {})
    episode_id = reset_response.get("episode_id", "unknown")

    # max_possible_reward: theoretical ceiling for the entire episode.
    # Used to normalise running totals into (0.01, 0.99) for every log field.
    max_steps = observation.get("max_steps", 10)
    max_possible_reward = max_steps * MAX_STEP_REWARD  # e.g. 5×0.99=4.95

    raw_total = 0.0      # raw accumulator — never printed directly
    step_num = 0
    done = False
    action_history = []

    while not done and step_num < max_steps:
        patients = observation.get("patients", [])
        unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]

        # If nothing left to do, break cleanly
        if not unassigned:
            break

        try:
            action = get_agent_action(observation, action_history)
            action_history.append(action)
        except Exception as e:
            print(f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num} error={str(e)} reward=0.01", flush=True)
            break

        step_result = call_env("POST", "/step", action)

        if "error" in step_result:
            print(f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num} error={step_result['error']} reward=0.01", flush=True)
            break

        # Individual step reward — environment.py guarantees (0.01, 0.99)
        reward = step_result.get("reward", 0.01)
        reward = round(max(0.01, min(0.99, float(reward))), 4)  # extra safety
        done = step_result.get("done", False)
        new_obs = step_result.get("observation", {})
        if new_obs:
            observation = new_obs
        raw_total += reward
        step_num += 1

        # Normalise running total into (0.01, 0.99) for the log
        norm_total = _norm(raw_total, max_possible_reward)
        print(
            f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num}"
            f" action={json.dumps(action)} reward={reward}"
            f" total_reward={norm_total} done={done}",
            flush=True
        )

    # Get final grade — graders return values strictly inside (0, 1) by construction
    grade_result = call_env("POST", "/grade")
    final_score = grade_result.get("score", 0.01)
    # Clamp defensively in case the external API returns an unexpected value
    final_score = round(max(0.01, min(0.99, float(final_score))), 4)

    # Normalise accumulated total into (0.01, 0.99) — never print raw totals > 1
    norm_total = _norm(raw_total if raw_total > 0 else 0.01, max_possible_reward)

    print(
        f"[END] task_id={task_id} episode_id={episode_id}"
        f" total_steps={step_num} total_reward={norm_total}"
        f" final_score={final_score}",
        flush=True
    )

    return {
        "task_id": task_id,
        "final_score": final_score,
        "total_reward": norm_total,  # normalised — always strictly in (0.01, 0.99)
    }


def main():
    results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] task_id={task_id} unhandled={e}", flush=True)
            result = {"task_id": task_id, "final_score": 0.01, "total_reward": 0.01}
        results.append(result)
        time.sleep(2)

    avg = round(sum(r["final_score"] for r in results) / len(results), 4)
    print(
        f"[SUMMARY] results={json.dumps(results)} average_score={avg}",
        flush=True
    )


if __name__ == "__main__":
    main()
