# wp1.7-soa-benchmark
Repository for benchmarking different SoA libraries.

## Build and Run
```
git clone https://github.com/cern-nextgen/wp1.7-soa-benchmark.git
cd wp1.7-soa-benchmark
cmake -B build
cmake --build build
./build/soa_boost
./build/soa_wrapper
./build/soa_manual
```

## Docker Container
The following container contains the dependencies to build and run the benchmark.
Pull and run the container, then follow the steps of the section [Build and Run](#build-and-run).
```
docker pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest
docker run -it --rm registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest bash
```
The corresponding Dockerfile can be found here: [wp1.7-soa-benchmark-image](https://github.com/cern-nextgen/wp1.7-soa-benchmark-image)
