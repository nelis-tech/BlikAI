#app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /BlikAI

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/huggingface/transformers \
    pip install streamlit \
    pip install python-wordpress-xmlrpc \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    pip install datasets \
    pip install accelerate \
    pip install sentencepiece 

ENTRYPOINT ["streamlit", "run"]

CMD ["blik.py"]