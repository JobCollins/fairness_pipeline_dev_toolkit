FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY fairness_pipeline_dev_toolkit/ ./fairness_pipeline_dev_toolkit/
COPY fairpipe/ ./fairpipe/
RUN pip install --no-cache-dir ".[api]"
EXPOSE 8000
CMD ["fairpipe", "serve", "--host", "0.0.0.0", "--port", "8000"]
