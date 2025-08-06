FROM python:3.11-slim

WORKDIR /app

# Install libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY faq_portal_transparencia.json .
COPY Create_Chromadb.py .
COPY Run_Query.py .
COPY app.py .

# Disable Chromadb telemetry
ENV ANONYMIZED_TELEMETRY=FALSE

EXPOSE 8501

CMD ["sh", "-c", "python Create_Chromadb.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]