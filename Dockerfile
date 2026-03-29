FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user pyproject.toml .
RUN mkdir -p server && touch server/__init__.py
RUN pip install --user --no-cache-dir .

COPY --chown=user . .

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$HOME/app

EXPOSE 7860

# Entry point: start the FastAPI server via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
