FROM python:3.12-slim

# Hugging Face Spaces exécute le conteneur avec l'UID 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user corpus ./corpus
COPY --chown=user eval ./eval

# Disque éphémère : l'index est reconstruit au démarrage (voir ensure_index).
ENV PYTHONPATH=/home/user/app/src \
    CHROMA_DIR=/tmp/chroma \
    PORT=7860
USER user
EXPOSE 7860

CMD uvicorn rag.api:app --host 0.0.0.0 --port ${PORT}
