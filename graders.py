from typing import Dict, Any

def enforce_bounds(val: float) -> float:
    """Strictly between 0 and 1, never exactly 0.0 or 1.0"""
    try:
        v = float(val)
        # Clamp to (0.02, 0.98) for extra safety margin
        v = max(0.02, min(0.98, v))
        return round(v, 4)
    except (ValueError, TypeError):
        return 0.02

def grade_task_easy(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.01

        patient = patients[0]
        current = patient.get("current_priority", "unassigned")

        if str(current).lower() == "unassigned":
            return 0.01

        total_reward = observation.get("total_reward", 0.0)
        score = max(0.05, float(total_reward))
        
        return enforce_bounds(score)
    except Exception:
        return 0.01

def grade_task_medium(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.01

        n = len(patients)
        if n == 0:
            return 0.01

        triaged = [p for p in patients if str(p.get("current_priority", "unassigned")).lower() not in ("unassigned", "none", "")]
        triage_score = len(triaged) / n

        has_diagnostics = [p for p in patients if p.get("diagnostics_ordered")]
        diagnostic_score = len(has_diagnostics) / n

        # Account for both int and string variants of ESI Priority to prevent silent zeroing
        admitted_critical = sum(
            1 for p in patients
            if p.get("admitted") and str(p.get("current_priority")) in ("1", "2", "P1", "P2")
        )
        denom = max(1, n // 2)
        admission_score = admitted_critical / denom

        final = (triage_score * 0.6) + (diagnostic_score * 0.2) + (admission_score * 0.2)
        return enforce_bounds(final)
    except Exception:
        return 0.01

def grade_task_hard(observation: Dict[str, Any]) -> float:
    try:
        patients = observation.get("patients", [])
        if not patients:
            return 0.01

        n = len(patients)
        if n == 0:
            return 0.01

        resources = observation.get("resources", {})
        total_resources = sum(resources.values()) if isinstance(resources, dict) else 0

        triaged = [p for p in patients if str(p.get("current_priority", "unassigned")).lower() not in ("unassigned", "none", "")]
        triage_score = len(triaged) / n

        initial_resources = {"xray": 3, "ecg": 5, "blood_test": 10, "ct_scan": 2, "ultrasound": 2}
        initial_total = sum(initial_resources.values())
        used = initial_total - total_resources
        resource_score = max(used, 0) / (initial_total * 0.5)

        initial_beds = 4
        available_beds = int(observation.get("available_beds", initial_beds))
        beds_used = max(initial_beds - available_beds, 0)
        bed_score = beds_used / max(initial_beds - 1, 1)

        final = (triage_score * 0.5) + (resource_score * 0.25) + (bed_score * 0.25)
        return enforce_bounds(final)
    except Exception:
        return 0.01

def run_grader(task_id: str, observation: Dict[str, Any]) -> float:
    """Route to correct grader based on task_id with failsafe fallback."""
    graders = {
        "task_easy": grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard": grade_task_hard,
    }
    grader = graders.get(task_id)
    if grader is None:
        return 0.01  # Silently fallback instead of crashing the pipeline
    
    try:
        return enforce_bounds(grader(observation))
    except Exception:
        return 0.01
