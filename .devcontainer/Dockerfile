FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    nano \
    make \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .devcontainer/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN pip install black pre-commit flake8 isort

COPY . /app

# Uruchom formatowanie kodu i instalację pre-commit
RUN black --line-length 120 . && \
    pre-commit install && \
    pre-commit run --all-files || echo "Pre-commit hook finished with modified files."

# Instalacja pipx i poetry
RUN python3 -m pip install --user pipx && \
    python3 -m pipx ensurepath && \
    python3 -m pipx completions && \
    python3 -m pipx install poetry

# RUN typer --install-completion && \
#     typer --install-completion


RUN echo "eval \"\$(register-python-argcomplete pipx)\"" >> ~/.bashrc

CMD ["python3 -c 'import torch; print(f'CUDA: {torch.cuda.is_available()}')'"]