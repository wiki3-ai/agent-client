FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

# Remove stale Yarn repo (GPG key expired) — not needed for a Python project.
RUN rm -f /etc/apt/sources.list.d/yarn.list /usr/share/keyrings/yarnkey.gpg

# System deps for jupyter / nbclient (most already present in base image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pre-warm pip cache; actual editable install happens in postCreateCommand
# so the mounted workspace is used.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspaces/agent-client
