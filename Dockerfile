# KLA PS01 — inference container.
#
# Pinned base image. ENTRYPOINT is inference.py, so the container takes the
# same --input_dir / --output_dir arguments as the bare script.
#
#   docker build -t kla-ps01 .
#   docker run --gpus all \
#       -v /host/in:/data/in -v /host/out:/data/out \
#       kla-ps01 --input_dir /data/in --output_dir /data/out
#
# NOTE: the base image must carry a CUDA 12.8+ build of PyTorch. Earlier CUDA
# builds do not run on recent NVIDIA architectures at all, and KLA benchmarks on H100
# (sm_90), which needs 11.8+. cu128 covers both.

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# libGL/libglib are needed only if OpenCV is used for PNG I/O; the .npy path
# never imports cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY inference.py .
COPY weights/ ./weights/

ENTRYPOINT ["python", "inference.py"]
