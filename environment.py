import random
import uuid
from typing import Dict, Any, Tuple
from models import (
    EnvironmentState, Patient, Vitals, Resources,
    Action, ActionType, ESIPriority, StepResult
)


PATIENT_SCENARIOS = [
    {
        "chief_complaint": "Severe chest pain radiating to left arm, diaphoresis",
        "age": 58, "gender": "Male",
        "vitals": {"heart_rate": 110, "blood_pressure_systolic": 90,
                   "blood_pressure_diastolic": 60, "respiratory_rate": 22,
                   "oxygen_saturation": 94.0, "temperature": 37.1, "pain_score": 9},
        "medical_history": ["hypertension", "diabetes"],
        "correct_priority": ESIPriority.P1,
        "deterioration_risk": 0.9
    },
    {
        "chief_complaint": "Sudden severe headache, worst of life, neck stiffness",
        "age": 42, "gender": "Female",
        "vitals": {"heart_rate": 98, "blood_pressure_systolic": 180,
                   "blood_pressure_diastolic": 110, "respiratory_rate": 18,
                   "oxygen_saturation": 97.0, "temperature": 38.2, "pain_score": 10},
        "medical_history": [],
        "correct_priority": ESIPriority.P1,
        "deterioration_risk": 0.85
    },
    {
        "chief_complaint": "Difficulty breathing, wheezing, known asthma",
        "age": 29, "gender": "Female",
        "vitals": {"heart_rate": 118, "blood_pressure_systolic": 130,
                   "blood_pressure_diastolic": 85, "respiratory_rate": 28,
                   "oxygen_saturation": 91.0, "temperature": 37.4, "pain_score": 6},
        "medical_history": ["asthma"],
        "correct_priority": ESIPriority.P2,
        "deterioration_risk": 0.7
    },
    {
        "chief_complaint": "High fever, confusion, suspected sepsis",
        "age": 75, "gender": "Male",
        "vitals": {"heart_rate": 124, "blood_pressure_systolic": 88,
                   "blood_pressure_diastolic": 55, "respiratory_rate": 26,
                   "oxygen_saturation": 93.0, "temperature": 39.8, "pain_score": 4},
        "medical_history": ["diabetes", "CKD"],
        "correct_priority": ESIPriority.P1,
        "deterioration_risk": 0.95
    },
    {
        "chief_complaint": "Abdominal pain, nausea, vomiting for 6 hours",
        "age": 35, "gender": "Female",
        "vitals": {"heart_rate": 95, "blood_pressure_systolic": 118,
                   "blood_pressure_diastolic": 76, "respiratory_rate": 16,
                   "oxygen_saturation": 98.0, "temperature": 37.8, "pain_score": 7},
        "medical_history": [],
        "correct_priority": ESIPriority.P3,
        "deterioration_risk": 0.3
    },
    {
        "chief_complaint": "Laceration on forearm, bleeding controlled",
        "age": 22, "gender": "Male",
        "vitals": {"heart_rate": 82, "blood_pressure_systolic": 122,
                   "blood_pressure_diastolic": 78, "respiratory_rate": 14,
                   "oxygen_saturation": 99.0, "temperature": 36.8, "pain_score": 4},
        "medical_history": [],
        "correct_priority": ESIPriority.P4,
        "deterioration_risk": 0.05
    },
    {
        "chief_complaint": "Prescription refill request, mild back pain for 2 weeks",
        "age": 50, "gender": "Female",
        "vitals": {"heart_rate": 72, "blood_pressure_systolic": 126,
                   "blood_pressure_diastolic": 80, "respiratory_rate": 14,
                   "oxygen_saturation": 99.0, "temperature": 36.6, "pain_score": 3},
        "medical_history": ["hypertension"],
        "correct_priority": ESIPriority.P5,
        "deterioration_risk": 0.02
    },
    {
        "chief_complaint": "Altered mental status, unresponsive, found at home",
        "age": 67, "gender": "Male",
        "vitals": {"heart_rate": 55, "blood_pressure_systolic": 80,
                   "blood_pressure_diastolic": 50, "respiratory_rate": 8,
                   "oxygen_saturation": 88.0, "temperature": 35.2, "pain_score": 0},
        "medical_history": ["stroke", "atrial_fibrillation"],
        "correct_priority": ESIPriority.P1,
        "deterioration_risk": 0.98
    },
    {
        "chief_complaint": "Ankle sprain after fall, mild swelling",
        "age": 19, "gender": "Male",
        "vitals": {"heart_rate": 76, "blood_pressure_systolic": 118,
                   "blood_pressure_diastolic": 74, "respiratory_rate": 14,
                   "oxygen_saturation": 99.0, "temperature": 36.7, "pain_score": 5},
        "medical_history": [],
        "correct_priority": ESIPriority.P4,
        "deterioration_risk": 0.02
    },
    {
        "chief_complaint": "Chest tightness, palpitations, history of atrial fibrillation",
        "age": 63, "gender": "Female",
        "vitals": {"heart_rate": 148, "blood_pressure_systolic": 105,
                   "blood_pressure_diastolic": 68, "respiratory_rate": 20,
                   "oxygen_saturation": 95.0, "temperature": 36.9, "pain_score": 6},
        "medical_history": ["atrial_fibrillation", "hypertension"],
        "correct_priority": ESIPriority.P2,
        "deterioration_risk": 0.75
    },
]

TASK_CONFIG = {
    "task_easy": {"num_patients": 1, "max_steps": 5, "beds": 10},
    "task_medium": {"num_patients": 5, "max_steps": 15, "beds": 6},
    "task_hard": {"num_patients": 10, "max_steps": 30, "beds": 4},
}

VALID_DIAGNOSTICS = ["xray", "ecg", "blood_test", "ct_scan", "ultrasound"]


class ClinicalTriageEnvironment:
    def __init__(self):
        self.state: EnvironmentState = None

    def reset(self, task_id: str = "task_easy") -> Dict[str, Any]:
        config = TASK_CONFIG.get(task_id, TASK_CONFIG["task_easy"])
        scenarios = random.sample(PATIENT_SCENARIOS, config["num_patients"])

        patients = []
        for i, s in enumerate(scenarios):
            p = Patient(
                patient_id=f"P{i+1:03d}",
                age=s["age"],
                gender=s["gender"],
                chief_complaint=s["chief_complaint"],
                vitals=Vitals(**s["vitals"]),
                medical_history=s["medical_history"],
                correct_priority=s["correct_priority"],
                deterioration_risk=s["deterioration_risk"],
                arrival_time=i * random.randint(1, 5)
            )
            patients.append(p)

        self.state = EnvironmentState(
            task_id=task_id,
            episode_id=str(uuid.uuid4())[:8],
            max_steps=config["max_steps"],
            patients=patients,
            available_beds=config["beds"],
            resources=Resources()
        )
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.state.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already done"}
            )

        reward, info = self._apply_action(action)
        self.state.step_count += 1
        self.state.elapsed_time += 3
        self.state.total_reward += reward

        # Check done conditions
        all_triaged = all(
            p.current_priority != ESIPriority.UNASSIGNED
            for p in self.state.patients
        )
        if self.state.step_count >= self.state.max_steps or all_triaged:
            self.state.done = True
            final_score = self._compute_final_score()
            info["final_score"] = final_score
            info["episode_summary"] = self._episode_summary()

        return StepResult(
            observation=self._get_observation(),
            reward=round(reward, 4),
            done=self.state.done,
            info=info
        )

    def _apply_action(self, action: Action) -> Tuple[float, Dict]:
        patient = self._get_patient(action.patient_id)
        if patient is None:
            return -0.1, {"error": f"Patient {action.patient_id} not found"}

        reward = 0.0
        info = {"action": action.action_type, "patient_id": action.patient_id}

        if action.action_type == ActionType.ASSIGN_PRIORITY:
            reward, info = self._handle_priority(patient, action.value, info)

        elif action.action_type == ActionType.ORDER_DIAGNOSTIC:
            reward, info = self._handle_diagnostic(patient, action.value, info)

        elif action.action_type == ActionType.ADMIT_PATIENT:
            reward, info = self._handle_admit(patient, info)

        elif action.action_type == ActionType.DISCHARGE_PATIENT:
            reward, info = self._handle_discharge(patient, info)

        elif action.action_type == ActionType.REASSESS:
            reward = 0.05
            info["message"] = f"Reassessed patient {action.patient_id}"

        return reward, info

    def _handle_priority(self, patient: Patient, value: str, info: Dict) -> Tuple[float, Dict]:
        try:
            assigned = ESIPriority(value)
        except ValueError:
            return -0.1, {**info, "error": f"Invalid priority value: {value}"}

        if patient.current_priority != ESIPriority.UNASSIGNED:
            return -0.05, {**info, "message": "Priority already assigned"}

        correct = patient.correct_priority
        patient.current_priority = assigned

        # Partial reward based on closeness
        diff = abs(int(assigned.value) - int(correct.value))
        if diff == 0:
            reward = 1.0
        elif diff == 1:
            reward = 0.6
        elif diff == 2:
            reward = 0.3
        else:
            reward = 0.0

        # Bonus: penalize under-triaging critical patients
        if int(correct.value) <= 2 and int(assigned.value) >= 4:
            reward -= 0.3

        info["correct_priority"] = correct.value
        info["assigned_priority"] = assigned.value
        info["priority_reward"] = reward
        return max(reward, 0.0), info

    def _handle_diagnostic(self, patient: Patient, value: str, info: Dict) -> Tuple[float, Dict]:
        if value not in VALID_DIAGNOSTICS:
            return -0.1, {**info, "error": f"Unknown diagnostic: {value}"}

        resource = getattr(self.state.resources, value, 0)
        if resource <= 0:
            return -0.1, {**info, "error": f"No {value} available"}

        if value in patient.diagnostics_ordered:
            return -0.05, {**info, "message": "Diagnostic already ordered"}

        patient.diagnostics_ordered.append(value)
        setattr(self.state.resources, value, resource - 1)

        # Reward based on clinical appropriateness
        appropriate = self._is_appropriate_diagnostic(patient, value)
        reward = 0.3 if appropriate else 0.1
        info["diagnostic"] = value
        info["appropriate"] = appropriate
        return reward, info

    def _handle_admit(self, patient: Patient, info: Dict) -> Tuple[float, Dict]:
        if self.state.available_beds <= 0:
            return -0.1, {**info, "error": "No beds available"}
        if patient.admitted:
            return -0.05, {**info, "message": "Already admitted"}

        patient.admitted = True
        self.state.available_beds -= 1

        # Reward admitting high-priority patients
        priority_val = int(patient.current_priority.value) if patient.current_priority != ESIPriority.UNASSIGNED else 5
        reward = 0.4 if priority_val <= 2 else 0.2
        info["beds_remaining"] = self.state.available_beds
        return reward, info

    def _handle_discharge(self, patient: Patient, info: Dict) -> Tuple[float, Dict]:
        if patient.discharged:
            return -0.05, {**info, "message": "Already discharged"}

        priority_val = int(patient.correct_priority.value)
        if priority_val >= 4:
            patient.discharged = True
            reward = 0.3
        else:
            reward = -0.2  # discharging critical patients
        info["discharge_appropriate"] = priority_val >= 4
        return reward, info

    def _is_appropriate_diagnostic(self, patient: Patient, diagnostic: str) -> bool:
        complaint = patient.chief_complaint.lower()
        mapping = {
            "ecg": ["chest", "palpitation", "cardiac", "heart", "atrial"],
            "xray": ["chest", "fall", "trauma", "fracture", "ankle", "breathing"],
            "blood_test": ["sepsis", "fever", "confusion", "abdominal", "pain"],
            "ct_scan": ["headache", "altered mental", "stroke", "abdominal"],
            "ultrasound": ["abdominal", "pregnancy", "pelvic"],
        }
        keywords = mapping.get(diagnostic, [])
        return any(k in complaint for k in keywords)

    def _compute_final_score(self) -> float:
        if not self.state.patients:
            return 0.01  # Clamped from 0.0
        
        scores = []
        for p in self.state.patients:
            if p.current_priority == ESIPriority.UNASSIGNED:
                scores.append(0.0)
            else:
                diff = abs(int(p.current_priority.value) - int(p.correct_priority.value))
                scores.append(max(0.0, 1.0 - diff * 0.3))
                
        raw_score = sum(scores) / len(scores)
        # Strictly enforce (0, 1) bounds
        return round(max(0.01, min(0.99, float(raw_score))), 4)

    def _episode_summary(self) -> Dict:
        triaged = [p for p in self.state.patients if p.current_priority != ESIPriority.UNASSIGNED]
        return {
            "total_patients": len(self.state.patients),
            "triaged": len(triaged),
            "admitted": sum(1 for p in self.state.patients if p.admitted),
            "total_steps": self.state.step_count,
            "total_reward": round(self.state.total_reward, 4),
        }

    def _get_patient(self, patient_id: str):
        for p in self.state.patients:
            if p.patient_id == patient_id:
                return p
        return None

    def _get_observation(self) -> Dict[str, Any]:
        if self.state is None:
            return {}
        return {
            "task_id": self.state.task_id,
            "episode_id": self.state.episode_id,
            "step_count": self.state.step_count,
            "max_steps": self.state.max_steps,
            "elapsed_time": self.state.elapsed_time,
            "available_beds": self.state.available_beds,
            "resources": self.state.resources.model_dump(),
            "patients": [
                {
                    "patient_id": p.patient_id,
                    "age": p.age,
                    "gender": p.gender,
                    "chief_complaint": p.chief_complaint,
                    "vitals": p.vitals.model_dump(),
                    "medical_history": p.medical_history,
                    "current_priority": p.current_priority.value,
                    "diagnostics_ordered": p.diagnostics_ordered,
                    "admitted": p.admitted,
                    "discharged": p.discharged,
                    "arrival_time": p.arrival_time,
                }
                for p in self.state.patients
            ],
            "done": self.state.done,
            "total_reward": self.state.total_reward,
        }

    def get_state(self) -> Dict[str, Any]:
        return self._get_observation()
