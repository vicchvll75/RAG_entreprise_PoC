FROM python:3.12-slim

# Utilisateur non root.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user corpus ./corpus
COPY --chown=user eval ./eval

# Disque éphémère : l'index est reconstruit au démarrage (voir ensure_index).
# PORT est fourni par l'hébergeur (Render) ; 7860 par défaut.
ENV PYTHONPATH=/home/user/app/src \
    CHROMA_DIR=/tmp/chroma \
    PORT=7860
USER user
EXPOSE 7860

CMD uvicorn rag.api:app --host 0.0.0.0 --port ${PORT}
