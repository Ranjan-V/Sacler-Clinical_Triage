import os
import json
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder") if OpenAI and HF_TOKEN else None

TASKS = ["task_easy", "task_medium", "task_hard"]


def safe_score(val) -> float:
    try:
        return round(max(0.1, min(0.9, float(val))), 2)
    except:
        return 0.1


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        with httpx.Client(timeout=60) as http:
            r = http.post(url, json=payload or {}) if method == "POST" else http.get(url)
            if not r.is_success:
                print(f"[DEBUG] {r.status_code} error on {endpoint}: {r.text}", flush=True)
                return {"error": r.text, "reward": 0.1, "done": False}
            return r.json()
    except Exception as e:
        print(f"[DEBUG] network exception on {endpoint}: {e}", flush=True)
        return {"error": str(e), "reward": 0.1, "done": False}


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


def estimate_priority(patient: dict) -> str:
    complaint = str(patient.get("chief_complaint", "")).lower()
    vitals = patient.get("vitals", {})
    hr = float(vitals.get("heart_rate", 80))
    sbp = float(vitals.get("blood_pressure_systolic", 120))
    rr = float(vitals.get("respiratory_rate", 16))
    spo2 = float(vitals.get("oxygen_saturation", 98))
    temp = float(vitals.get("temperature", 37))
    pain = float(vitals.get("pain_score", 0))

    if (
        spo2 <= 90 or sbp <= 90 or rr >= 30 or rr <= 8 or hr >= 140 or hr <= 50
        or "unresponsive" in complaint or "altered mental" in complaint
        or "sepsis" in complaint or "worst of life" in complaint
    ):
        return "1"

    if (
        spo2 <= 94 or sbp <= 100 or rr >= 24 or hr >= 115 or temp >= 39
        or "chest pain" in complaint or "difficulty breathing" in complaint
        or "palpitations" in complaint or "confusion" in complaint
    ):
        return "2"

    if pain >= 7 or "abdominal pain" in complaint or "vomiting" in complaint:
        return "3"

    if "laceration" in complaint or "ankle sprain" in complaint or pain >= 4:
        return "4"

    return "5"


def heuristic_action(observation: dict) -> dict:
    patients = observation.get("patients", [])
    unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]

    if unassigned:
        ranked = sorted(unassigned, key=lambda p: int(estimate_priority(p)))
        patient = ranked[0]
        return {
            "patient_id": patient["patient_id"],
            "action_type": "assign_priority",
            "value": estimate_priority(patient),
        }

    admitted_candidates = [
        p for p in patients
        if not p.get("admitted") and str(p.get("current_priority")) in ("1", "2")
    ]
    if admitted_candidates:
        patient = admitted_candidates[0]
        return {
            "patient_id": patient["patient_id"],
            "action_type": "admit_patient",
            "value": "admit",
        }

    return {
        "patient_id": patients[0]["patient_id"] if patients else "P001",
        "action_type": "reassess",
        "value": "reassess",
    }


def get_agent_action(observation: dict, action_history: list) -> dict:
    if client is None:
        return heuristic_action(observation)

    prompt = build_prompt(observation, action_history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]
        action = json.loads(content)
    except Exception:
        return heuristic_action(observation)

    valid_types = ["assign_priority", "order_diagnostic", "admit_patient", "discharge_patient", "reassess"]
    if action.get("action_type") not in valid_types:
        action["action_type"] = "assign_priority"

    patients = observation.get("patients", [])
    unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]
    assigned_ids = [p["patient_id"] for p in patients if p.get("current_priority") != "unassigned"]

    if action.get("patient_id") in assigned_ids and unassigned:
        action["patient_id"] = unassigned[0]["patient_id"]
    if "patient_id" not in action and unassigned:
        action["patient_id"] = unassigned[0]["patient_id"]
    if "value" not in action or not action["value"]:
        action["value"] = "3"

    return action


def run_task(task_id: str) -> dict:
    print(f"[START] task_id={task_id} timestamp={time.time()}", flush=True)
    try:
        return _run_task_inner(task_id)
    except Exception as e:
        print(f"[END] task_id={task_id} episode_id=error total_steps=0 total_reward=0.02 final_score=0.02", flush=True)
        return {"task_id": task_id, "score": 0.1, "total_reward": 0.1}


def _run_task_inner(task_id: str) -> dict:
    reset_response = call_env("POST", "/reset", {"task_id": task_id})
    if "error" in reset_response:
        print(f"[END] task_id={task_id} episode_id=error total_steps=0 total_reward=0.02 final_score=0.02", flush=True)
        return {"task_id": task_id, "score": 0.1, "total_reward": 0.1}

    observation = reset_response.get("observation", {})
    episode_id = reset_response.get("episode_id", "unknown")
    max_steps = observation.get("max_steps", 10)

    raw_total = 0.0
    step_num = 0
    done = False
    action_history = []

    while not done and step_num < max_steps:
        patients = observation.get("patients", [])
        unassigned = [p for p in patients if p.get("current_priority") == "unassigned"]
        if not unassigned:
            break

        try:
            action = get_agent_action(observation, action_history)
            action_history.append(action)
        except Exception as e:
            print(f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num} error={str(e)} reward=0.1", flush=True)
            break

        step_result = call_env("POST", "/step", action)
        if "error" in step_result:
            print(f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num} error={step_result['error']} reward=0.1", flush=True)
            break

        reward = safe_score(step_result.get("reward", 0.1))
        done = step_result.get("done", False)
        new_obs = step_result.get("observation", {})
        if new_obs:
            observation = new_obs
        raw_total += reward
        step_num += 1

        print(
            f"[STEP] task_id={task_id} episode_id={episode_id} step={step_num}"
            f" action={json.dumps(action)} reward={reward}"
            f" total_reward={safe_score(raw_total / max(step_num, 1))} done={done}",
            flush=True
        )

    grade_result = call_env("POST", "/grade")
    final_score = safe_score(grade_result.get("score", 0.1))
    total_reward = safe_score(raw_total / max(step_num, 1))

    print(
        f"[END] task_id={task_id} episode_id={episode_id}"
        f" total_steps={step_num} total_reward={total_reward}"
        f" final_score={final_score}",
        flush=True
    )

    return {"task_id": task_id, "score": final_score, "total_reward": total_reward}


def main():
    results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] task_id={task_id} unhandled={e}", flush=True)
            result = {"task_id": task_id, "score": 0.1, "total_reward": 0.1}
        results.append(result)
        time.sleep(2)

    try:
        avg = safe_score(sum(r["final_score"] for r in results) / len(results))
    except:
        avg = 0.1

    print(f"[SUMMARY] results={json.dumps(results)} average_score={avg}", flush=True)


if __name__ == "__main__":
    main()
