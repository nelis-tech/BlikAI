#app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /codegen-1

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nelis-tech/codegen.git
    
RUN pip install torch \
    pip install datasets \
    pip install accelerate \
    pip install sentencepiece \
    pip install git+https://github.com/huggingface/transformers.git 

ENTRYPOINT ["streamlit", "run", "app.py", "model.py", "--server.port=8501", "--server.address=0.0.0.0"]