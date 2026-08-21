# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG CODEX_VERSION=latest
ARG CLAUDE_CODE_VERSION=latest

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

COPY --from=uvbin /uv /uvx /usr/local/bin/

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        gnupg \
        jq \
        less \
        openssh-client \
        ripgrep \
        shellcheck \
    && git lfs install --system \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh nodejs \
    && npm install -g \
        "@openai/codex@${CODEX_VERSION}" \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && codex --version \
    && claude --version \
    && npm cache clean --force \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid "${GID}" "${USERNAME}" \
    && adduser \
        --disabled-password \
        --gecos "" \
        --shell /bin/bash \
        --uid "${UID}" \
        --gid "${GID}" \
        "${USERNAME}" \
    && mkdir -p \
        "/home/${USERNAME}/.codex" \
        "/home/${USERNAME}/.claude" \
        "/home/${USERNAME}/.config/gh" \
    && chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}"

USER ${USERNAME}

ENV HOME=/home/${USERNAME} \
    CODEX_HOME=/home/${USERNAME}/.codex \
    PATH=/home/${USERNAME}/.local/bin:${PATH} \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

CMD ["/bin/bash"]
