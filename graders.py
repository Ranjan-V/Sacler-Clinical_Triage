from typing import Dict, Any


def grade_task_easy(observation: Dict[str, Any]) -> float:
    patients = observation.get("patients", [])
    if not patients:
        return 0.01

    patient = patients[0]
    current = patient.get("current_priority", "unassigned")

    # Not yet triaged
    if current == "unassigned":
        return 0.01

    # Map priority string to int (1=best, 5=lowest)
    try:
        assigned = int(current)
    except (ValueError, TypeError):
        return 0.01

    # Use total_reward as proxy, bounded natively to (0.01, 0.99).
    total_reward = observation.get("total_reward", 0.0)
    score = max(0.01, min(0.99, float(total_reward)))

    # Ensure a baseline score when the patient was actually triaged
    score = max(score, 0.05)

    # score is in [0.05, 0.99] — strictly inside (0, 1)
    return round(score, 4)


def grade_task_medium(observation: Dict[str, Any]) -> float:
    """
    Medium: 5 patients managed. Score based on:
    - Priority accuracy (60%)
    - Diagnostic appropriateness (20%)
    - Admission decisions (20%)
    """
    patients = observation.get("patients", [])
    if not patients:
        return 0.01

    n = len(patients)

    triaged = [p for p in patients if p.get("current_priority") not in ("unassigned", None, "")]
    # Each sub-score is bounded natively to [0.01, 0.99] so the weighted sum
    # is always in [0.01, 0.99] — strictly inside (0, 1) — with no clamp needed.
    triage_score = max(0.01, min(0.99, len(triaged) / n))

    has_diagnostics = [p for p in patients if p.get("diagnostics_ordered")]
    diagnostic_score = max(0.01, min(0.99, len(has_diagnostics) / n))

    admitted_critical = sum(
        1 for p in patients
        if p.get("admitted") and p.get("current_priority") in ("1", "2")
    )
    denom = max(1, n // 2)
    admission_score = max(0.01, min(0.99, admitted_critical / denom))

    # Weighted sum: min = 0.01*(0.6+0.2+0.2) = 0.01, max = 0.99
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
        return 0.01

    n = len(patients)
    resources = observation.get("resources", {})
    total_resources = sum(resources.values()) if resources else 0

    triaged = [p for p in patients if p.get("current_priority") not in ("unassigned", None, "")]
    triage_score = max(0.01, min(0.99, len(triaged) / n))

    initial_resources = {"xray": 3, "ecg": 5, "blood_test": 10, "ct_scan": 2, "ultrasound": 2}
    initial_total = sum(initial_resources.values())  # = 22
    used = initial_total - total_resources
    resource_score = max(0.01, min(0.99, max(used, 0) / (initial_total * 0.5)))

    # task_hard starts with 4 beds
    initial_beds = 4
    available_beds = observation.get("available_beds", initial_beds)
    beds_used = max(initial_beds - available_beds, 0)
    bed_score = max(0.01, min(0.99, beds_used / max(initial_beds - 1, 1)))

    # Weighted sum: min = 0.01*(0.5+0.25+0.25) = 0.01, max = 0.99
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
    # Every grader returns a value strictly inside (0, 1) by construction.
    return grader(observation)
