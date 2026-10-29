FROM python:3.11-slim
RUN pip install --no-cache-dir pandas==2.3.3 numpy==2.3.5 scipy==1.16.3 scikit-learn==1.9.1 matplotlib==3.10.0 seaborn==0.13.2 pyarrow==19.0.1
ENV MPLCONFIGDIR=/tmp/mpl OPENBLAS_NUM_THREADS=1
USER 65534:65534
WORKDIR /work
