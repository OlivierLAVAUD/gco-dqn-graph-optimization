FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    # Cible de compilation JIT (RTX 3060 = Ampere, compute capability 8.6)
    TORCH_CUDA_ARCH_LIST=8.6

WORKDIR /app

COPY requirements.txt .

# Torch GPU : le wheel PyPI par défaut embarque le runtime CUDA + cuDNN
# (image ~3 Go). Pas besoin de --index-url cpu.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Vérification GPU (affichée dans les logs au démarrage du conteneur)
RUN python -c "import torch; print(f'[build] torch={torch.__version__} cuda_wheel={torch.version.cuda}')"

# Exécution par défaut : pipeline complet (génération -> neo4j -> train -> baselines)
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--config", "config.yaml", "--source", "auto"]
