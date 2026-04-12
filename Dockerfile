FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV ENV_URL=http://localhost:7860

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn main:app --host 0.0.0.0 --port 7860 &\n\
sleep 5\n\
python inference.py\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]