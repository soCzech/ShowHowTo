FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir \
    einops==0.8.0  \
    omegaconf==2.3.0 \
    pillow \
    transformers==4.46.3 \
    open_clip_torch==2.22.0 \
    kornia==0.7.4 \
    tqdm \
    timm==1.0.11
