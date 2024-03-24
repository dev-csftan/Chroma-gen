# Container for building the environment
FROM --platform=linux/x86-64 mambaorg/micromamba:1.5.1

# Install Conda/Mamba dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    find -name '*.a' -delete && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
    rm -rf $MAMBA_ROOT_PREFIX/conda-meta \
    $MAMBA_ROOT_PREFIX/include \
    $MAMBA_ROOT_PREFIX/lib/libpython*.so.* \
    $MAMBA_ROOT_PREFIX/lib/python*/idlelib \
    $MAMBA_ROOT_PREFIX/lib/libasan.so.* \
    $MAMBA_ROOT_PREFIX/lib/libtsan.so.* \
    $MAMBA_ROOT_PREFIX/lib/liblsan.so.* \
    $MAMBA_ROOT_PREFIX/lib/libubsan.so.* \
    $MAMBA_ROOT_PREFIX/bin/x86_64-conda-linux-gnu-ld \
    $MAMBA_ROOT_PREFIX/bin/sqlite3 \
    $MAMBA_ROOT_PREFIX/bin/openssl \
    $MAMBA_ROOT_PREFIX/share/terminfo 

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
ENV STREAMLIT_SERVER_PORT="8080"
# server.fileWatcherType=none is needed on Mac OS M1 to prevent OSError: [Errno 38] Function not implemented:
# https://github.com/streamlit/streamlit/issues/4842#issuecomment-1309856767
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Copy application code
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ /home/src/
WORKDIR /home/src/
CMD ["/opt/conda/bin/python", "-m", "streamlit", "run", "GenProtein.py"]