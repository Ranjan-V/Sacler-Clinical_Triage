from setuptools import setup, find_packages

setup(
    name="clinical-triage-openenv",
    version="1.0.0",
    description="OpenEnv environment simulating a clinical triage coordinator for AI agent training",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi==0.115.0",
        "uvicorn[standard]==0.30.6",
        "pydantic==2.8.2",
        "openai==1.51.0",
        "python-dotenv==1.0.1",
        "httpx==0.27.2",
        "pytest==8.3.3",
        "pyyaml==6.0.2",
    ],
)