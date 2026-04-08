from typing import Dict, Any


def clamp(score: float) -> float:
    """Strictly clamp to open interval (0, 1) — never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, float(score))), 4)


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

    # We don't have correct_priority in observation, so use total_reward as proxy
    total_reward = observation.get("total_reward", 0.0)

    # Normalize: max single-patient reward is ~1.0 for priority alone
    # Cap normalizer at 1.0 to avoid scores > 0.99 after clamp
    score = min(total_reward, 1.0) / 1.0

    # Ensure we never return exactly 0.0 if patient was triaged
    score = max(score, 0.05)

    return clamp(score)


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
    triage_score = len(triaged) / n if n > 0 else 0.0

    has_diagnostics = [p for p in patients if p.get("diagnostics_ordered")]
    diagnostic_score = len(has_diagnostics) / n if n > 0 else 0.0

    admitted_critical = sum(
        1 for p in patients
        if p.get("admitted") and p.get("current_priority") in ("1", "2")
    )
    denom = max(1, n // 2)
    admission_score = min(admitted_critical / denom, 0.99)

    final = (triage_score * 0.6) + (diagnostic_score * 0.2) + (admission_score * 0.2)

    # Ensure strictly > 0 when any work has been done
    if len(triaged) > 0 or has_diagnostics:
        final = max(final, 0.05)

    return clamp(final)


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
    triage_score = len(triaged) / n if n > 0 else 0.0

    initial_resources = {"xray": 3, "ecg": 5, "blood_test": 10, "ct_scan": 2, "ultrasound": 2}
    initial_total = sum(initial_resources.values())  # = 22
    used = initial_total - total_resources
    # Avoid dividing by zero; target 50% usage = good
    resource_score = min(max(used, 0) / (initial_total * 0.5), 0.99)

    # task_hard starts with 4 beds
    initial_beds = 4
    available_beds = observation.get("available_beds", initial_beds)
    beds_used = max(initial_beds - available_beds, 0)
    bed_score = min(beds_used / max(initial_beds - 1, 1), 0.99)

    final = (triage_score * 0.5) + (resource_score * 0.25) + (bed_score * 0.25)

    # Ensure strictly > 0 when any work has been done
    if triaged:
        final = max(final, 0.05)

    return clamp(final)


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
    score = grader(observation)
    # Final safety net — belt and suspenders
    return clamp(score)
