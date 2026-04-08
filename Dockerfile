FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install ONLY required dependency
RUN pip install --no-cache-dir openai

ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4.1-mini

CMD ["python", "inference.py"]