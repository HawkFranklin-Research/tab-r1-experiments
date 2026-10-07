FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY environment/requirements.txt environment/requirements.txt
RUN pip install --no-cache-dir -r environment/requirements.txt
COPY tabr1_paths.py ./
COPY 03_analysis_and_figures ./03_analysis_and_figures
COPY release_tools/rebuild.py ./release_tools/rebuild.py
COPY results ./results
COPY data/frozen_test_sets ./data/frozen_test_sets
ENTRYPOINT ["python", "release_tools/rebuild.py", "--output-root", "/outputs/rebuilt"]
