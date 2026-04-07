from typing import Dict, Any
from models import ESIPriority


def grade_task_easy(observation: Dict[str, Any]) -> float:
    """
    Easy: Single patient correctly triaged.
    Score 0.0 - 1.0 based on priority accuracy.
    """
    patients = observation.get("patients", [])
    if not patients:
        return 0.0

    patient = patients[0]
    current = patient.get("current_priority", "unassigned")

    if current == "unassigned":
        return 0.0

    # We check via total_reward normalized
    total_reward = observation.get("total_reward", 0.0)
    max_possible = 1.0
    return min(round(total_reward / max_possible, 4), 1.0)


def grade_task_medium(observation: Dict[str, Any]) -> float:
    """
    Medium: 5 patients managed. Score based on:
    - Priority accuracy (60%)
    - Diagnostic appropriateness (20%)
    - Admission decisions (20%)
    """
    patients = observation.get("patients", [])
    if not patients:
        return 0.0

    triaged = [p for p in patients if p.get("current_priority") != "unassigned"]
    triage_score = len(triaged) / len(patients)

    has_diagnostics = [p for p in patients if p.get("diagnostics_ordered")]
    diagnostic_score = len(has_diagnostics) / len(patients)

    admitted_critical = sum(
        1 for p in patients
        if p.get("admitted") and p.get("current_priority") in ["1", "2"]
    )
    admission_score = min(admitted_critical / max(1, len(patients) // 2), 1.0)

    final = (triage_score * 0.6) + (diagnostic_score * 0.2) + (admission_score * 0.2)
    return round(final, 4)


def grade_task_hard(observation: Dict[str, Any]) -> float:
    """
    Hard: 10 patients MCI. Score based on:
    - Priority accuracy (50%)
    - Resource efficiency (25%)
    - Bed management (25%)
    """
    patients = observation.get("patients", [])
    if not patients:
        return 0.0

    resources = observation.get("resources", {})
    total_resources = sum(resources.values()) if resources else 1

    triaged = [p for p in patients if p.get("current_priority") != "unassigned"]
    triage_score = len(triaged) / len(patients)

    # Resource usage score (reward using resources efficiently)
    initial_resources = {"xray": 3, "ecg": 5, "blood_test": 10, "ct_scan": 2, "ultrasound": 2}
    initial_total = sum(initial_resources.values())
    used = initial_total - total_resources
    resource_score = min(used / (initial_total * 0.5), 1.0)

    # Bed management
    available_beds = observation.get("available_beds", 4)
    beds_used = 4 - available_beds
    bed_score = min(beds_used / 3, 1.0)

    final = (triage_score * 0.5) + (resource_score * 0.25) + (bed_score * 0.25)
    return round(final, 4)


def run_grader(task_id: str, observation: Dict[str, Any]) -> float:
    """Route to correct grader based on task_id."""
    graders = {
        "task_easy": grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard": grade_task_hard,
    }
    grader = graders.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return grader(observation)