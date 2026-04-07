from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ESIPriority(str, Enum):
    P1 = "1"  # Immediate - life threatening
    P2 = "2"  # Emergent - high risk
    P3 = "3"  # Urgent - stable but needs care
    P4 = "4"  # Less urgent - minor
    P5 = "5"  # Non-urgent - routine
    UNASSIGNED = "unassigned"


class ActionType(str, Enum):
    ASSIGN_PRIORITY = "assign_priority"
    ORDER_DIAGNOSTIC = "order_diagnostic"
    ADMIT_PATIENT = "admit_patient"
    DISCHARGE_PATIENT = "discharge_patient"
    REASSESS = "reassess"


class Vitals(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in bpm")
    blood_pressure_systolic: int = Field(..., description="Systolic BP in mmHg")
    blood_pressure_diastolic: int = Field(..., description="Diastolic BP in mmHg")
    respiratory_rate: int = Field(..., description="Breaths per minute")
    oxygen_saturation: float = Field(..., description="SpO2 percentage")
    temperature: float = Field(..., description="Body temperature in Celsius")
    pain_score: int = Field(..., ge=0, le=10, description="Pain score 0-10")


class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: Vitals
    medical_history: List[str] = []
    current_priority: ESIPriority = ESIPriority.UNASSIGNED
    correct_priority: ESIPriority  # ground truth, hidden from agent
    diagnostics_ordered: List[str] = []
    admitted: bool = False
    discharged: bool = False
    arrival_time: int = 0  # simulated minutes
    deterioration_risk: float = 0.0  # 0.0 to 1.0


class Resources(BaseModel):
    xray: int = Field(default=3, description="Available X-ray slots")
    ecg: int = Field(default=5, description="Available ECG machines")
    blood_test: int = Field(default=10, description="Available blood test kits")
    ct_scan: int = Field(default=2, description="Available CT scan slots")
    ultrasound: int = Field(default=2, description="Available ultrasound slots")


class EnvironmentState(BaseModel):
    task_id: str
    episode_id: str
    step_count: int = 0
    max_steps: int
    elapsed_time: int = 0  # simulated minutes
    patients: List[Patient] = []
    available_beds: int = 10
    resources: Resources = Resources()
    done: bool = False
    total_reward: float = 0.0
    info: Dict[str, Any] = {}


class Action(BaseModel):
    patient_id: str = Field(..., description="ID of patient to act on")
    action_type: ActionType = Field(..., description="Type of action to perform")
    value: str = Field(..., description="Value for the action")


class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str = Field(default="task_easy", description="Task to initialize")


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task_id: str
    episode_id: str
    message: str