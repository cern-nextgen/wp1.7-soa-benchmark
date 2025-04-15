# wp1.7-soa-benchmark
Repository for benchmarking the different SoA libraries.

# Build and Run
```
git clone https://github.com/cern-nextgen/wp1.7-soa-benchmark.git
cd wp1.7-soa-benchmark
cmake -B build
cmake --build build
./build/soa_wrapper
./build/soa_boost
```

# Docker Container
The following container contains the dependencies to run the benchmark.
```
docker pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest
```
The Dockerfile fo this image can be found here: [wp1.7-soa-benchmark-image](https://github.com/cern-nextgen/wp1.7-soa-benchmark-image)
