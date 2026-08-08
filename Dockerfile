# Reproducible GPU development image. SGLang is cloned and patched inside the
# image; no SGLang checkout is accepted from the build context.
ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12-full python3.12-dev python3-pip git git-lfs \
        build-essential cmake ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /workspace

COPY . /workspace/lightcone-spec
RUN git clone --filter=blob:none https://github.com/sgl-project/sglang.git /workspace/sglang \
    && git -C /workspace/sglang checkout --detach 3312645a307453893a00778592f105581e3d1c3d \
    && /workspace/lightcone-spec/patches/sglang/apply.sh /workspace/sglang \
    && pip install -e "/workspace/sglang/python[all]" \
    && pip install -e "/workspace/lightcone-spec[gpu]"

ENTRYPOINT ["lightcone-spec"]
CMD ["--help"]
