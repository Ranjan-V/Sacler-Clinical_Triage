# 🏥 Clinical Triage Coordinator — OpenEnv Environment

An OpenEnv-compliant simulation of a real-world Emergency Department triage system.
An AI agent acts as a triage nurse — assessing patients, assigning ESI priority levels,
ordering diagnostics, managing beds, and coordinating care under resource constraints.

---

## 🌍 Real-World Task

Emergency Department triage is one of the most critical decision-making tasks in healthcare.
Every minute of delay in correctly prioritizing a patient can result in death or permanent harm.
This environment simulates that exact scenario — the agent must assess incoming patients and
make rapid, accurate triage decisions just like a real ED nurse would.

---

## 🎯 Tasks

| Task ID | Difficulty | Patients | Max Steps | Description |
|---|---|---|---|---|
| `task_easy` | Easy | 1 | 5 | Assign correct ESI priority to a single patient |
| `task_medium` | Medium | 5 | 15 | Manage a multi-patient queue with diagnostics and admissions |
| `task_hard` | Hard | 10 | 30 | Coordinate a mass casualty incident with resource constraints |

---

## 🔁 Action Space

The agent can take the following actions at each step:

| Action Type | Value | Description |
|---|---|---|
| `assign_priority` | `"1"` to `"5"` | Assign ESI triage priority to a patient |
| `order_diagnostic` | `"xray"`, `"ecg"`, `"blood_test"`, `"ct_scan"`, `"ultrasound"` | Order a diagnostic test |
| `admit_patient` | `"admit"` | Admit patient to an available bed |
| `discharge_patient` | `"discharge"` | Discharge a lower-priority patient |
| `reassess` | `"reassess"` | Reassess a patient's condition |

---

## 👁️ Observation Space

Each observation contains:
```json
{
  "task_id": "task_easy",
  "episode_id": "abc12345",
  "step_count": 0,
  "max_steps": 5,
  "elapsed_time": 0,
  "available_beds": 10,
  "resources": {
    "xray": 3,
    "ecg": 5,
    "blood_test": 10,
    "ct_scan": 2,
    "ultrasound": 2
  },
  "patients": [
    {
      "patient_id": "P001",
      "age": 58,
      "gender": "Male",
      "chief_complaint": "Severe chest pain radiating to left arm",
      "vitals": {
        "heart_rate": 110,
        "blood_pressure_systolic": 90,
        "blood_pressure_diastolic": 60,
        "respiratory_rate": 22,
        "oxygen_saturation": 94.0,
        "temperature": 37.1,
        "pain_score": 9
      },
      "medical_history": ["hypertension", "diabetes"],
      "current_priority": "unassigned",
      "diagnostics_ordered": [],
      "admitted": false,
      "discharged": false,
      "arrival_time": 0
    }
  ],
  "done": false,
  "total_reward": 0.0
}
```

---

## 🏆 Reward Function

Rewards are partial and continuous — not just 0 or 1:

| Situation | Reward |
|---|---|
| Correct priority assigned | +1.0 |
| Priority off by 1 level | +0.6 |
| Priority off by 2 levels | +0.3 |
| Priority off by 3+ levels | 0.0 |
| Under-triaging a critical patient (P1/P2 → P4/P5) | -0.3 penalty |
| Appropriate diagnostic ordered | +0.3 |
| Inappropriate diagnostic | +0.1 |
| Admitting high-priority patient | +0.4 |
| Correct discharge of low-priority patient | +0.3 |
| Discharging a critical patient | -0.2 |

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Reset environment for a new episode |
| `POST` | `/step` | Take an action and receive reward |
| `GET` | `/state` | Get current environment state |
| `POST` | `/grade` | Get grader score (0.0 - 1.0) |

---

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/clinical-triage-openenv.git
cd clinical-triage-openenv
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Set environment variables
Create a `.env` file:
```env
API_BASE_URL=[https://router.huggingface.co/v1]
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct:cerebras
HF_TOKEN=your_huggingface_token
ENV_URL=http://localhost:7860
```

### 5. Run the server
```bash
python main.py
```

Visit `http://localhost:7860/docs` for the Swagger UI.

---

## 🤖 Run the Agent (inference.py)

Make sure the server is running, then in a separate terminal:
```bash
python inference.py
```

The script will run the agent through all 3 tasks and print structured logs.

---

## 🐳 Docker
```bash
docker build -t clinical-triage .
docker run -p 7860:7860 clinical-triage
```

---

## 📋 Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (e.g. OpenAI) |
| `MODEL_NAME` | Model to use (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | HuggingFace / OpenAI API key |
| `ENV_URL` | URL where the environment is running |

---

## 📁 Project Structure
clinical_triage/
├── main.py            # FastAPI server
├── environment.py     # Core triage logic
├── models.py          # Pydantic typed models
├── graders.py         # Task graders (easy/medium/hard)
├── inference.py       # Agent inference script
├── openenv.yaml       # OpenEnv config
├── requirements.txt   # Dependencies
├── setup.py           # Package setup
├── Dockerfile         # HuggingFace Spaces deployment
└── README.md          # This file

---

## 👥 Team

Built for the Meta x Scaler OpenEnv Hackathon 2025.
