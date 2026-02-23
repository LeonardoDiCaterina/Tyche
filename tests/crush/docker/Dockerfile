FROM python:3.10-slim

# Install build tools and PractRand dependencies
RUN apt-get update && apt-get install -y \
    g++ curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Build PractRand from source
RUN curl -L -o /tmp/PractRand.zip \
    https://sourceforge.net/projects/pracrand/files/PractRand-pre0.95.zip/download && \
    unzip /tmp/PractRand.zip -d /tmp/PractRand && \
    cd /tmp/PractRand && \
    g++ -std=c++14 -O3 -o RNG_test \
        tools/RNG_test.cpp \
        src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp \
        -Iinclude -pthread && \
    mv RNG_test /usr/local/bin/ && \
    rm -rf /tmp/PractRand /tmp/PractRand.zip

# Install Python dependencies
RUN pip install jax jaxlib numpy scipy

WORKDIR /tyche