import math
from typing import Dict, Any

def enforce_bounds(val: Any) -> float:
    """Airtight boundary enforcer. Uses 0.05 to 0.95 to defeat math.isclose() float rounding traps."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return 0.05
        return float(max(0.05, min(0.95, round(v, 4))))
    except Exception:
        return 0.05

def grade_task_easy(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.05

        patient = patients[0]
        current = patient.get("current_priority", "unassigned")

        if str(current).lower() == "unassigned":
            return 0.05

        total_reward = observation.get("total_reward", 0.05)
        score = max(0.05, float(total_reward))
        
        return enforce_bounds(score)
    except Exception:
        return 0.05

def grade_task_medium(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.05

        n = len(patients)
        if n == 0:
            return 0.05

        triaged = [p for p in patients if str(p.get("current_priority", "unassigned")).lower() not in ("unassigned", "none", "")]
        triage_score = len(triaged) / n

        has_diagnostics = [p for p in patients if p.get("diagnostics_ordered")]
        diagnostic_score = len(has_diagnostics) / n

        admitted_critical = sum(
            1 for p in patients
            if p.get("admitted") and str(p.get("current_priority")) in ("1", "2", "P1", "P2")
        )
        denom = max(1, n // 2)
        admission_score = admitted_critical / denom

        final = (triage_score * 0.6) + (diagnostic_score * 0.2) + (admission_score * 0.2)
        return enforce_bounds(final)
    except Exception:
        return 0.05

def grade_task_hard(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.05

        n = len(patients)
        if n == 0:
            return 0.05

        resources = observation.get("resources", {})
        total_resources = sum(resources.values()) if isinstance(resources, dict) else 0

        triaged = [p for p in patients if str(p.get("current_priority", "unassigned")).lower() not in ("unassigned", "none", "")]
        triage_score = len(triaged) / n

        initial_resources = {"xray": 3, "ecg": 5, "blood_test": 10, "ct_scan": 2, "ultrasound": 2}
        initial_total = sum(initial_resources.values())
        used = initial_total - total_resources
        resource_score = max(used, 0) / (initial_total * 0.5)

        initial_beds = 4
        try:
            available_beds = int(observation.get("available_beds", initial_beds))
        except Exception:
            available_beds = initial_beds
            
        beds_used = max(initial_beds - available_beds, 0)
        bed_score = beds_used / max(initial_beds - 1, 1)

        final = (triage_score * 0.5) + (resource_score * 0.25) + (bed_score * 0.25)
        return enforce_bounds(final)
    except Exception:
        return 0.05

def run_grader(task_id: str, observation: Dict[str, Any]) -> float:
    """Route to correct grader safely without crashing on fake tasks."""
    try:
        graders = {
            "task_easy": grade_task_easy,
            "task_medium": grade_task_medium,
            "task_hard": grade_task_hard,
        }
        
        if task_id not in graders:
            return 0.05 
            
        return enforce_bounds(graders[task_id](observation))
    except Exception:
        return 0.05
