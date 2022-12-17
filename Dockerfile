FROM python:3.8

RUN apt-get update \
    && apt-get install --assume-yes --no-install-recommends --quiet \
            python3 \
            python3-pip \
    && apt-get clean all

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        setuptools==65.5.1 \
        wheel==0.38.4 \
        spacy==3.4.2 \
        ipywidgets \
        argilla[server]==1.0.1 \
        spacy[transformers,lookups] \
        datasets==2.6.1 \
        matplotlib \
        jsonlines \
        flake8

ADD . /app
WORKDIR /app

CMD ["python3", "active_learning_loop.py"]
