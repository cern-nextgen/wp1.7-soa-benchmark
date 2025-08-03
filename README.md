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

for running vtune compilare, specify vtune in the name of build folder, like this:
```
cmake -B build-vtune
cmake --build build-vtune
```

For running benchmarks without debug symbols, use the script:
```
pyhton3 run_benchmarks.py build/ soa_versions.csv
```

For running with debug symbols, use:
```
python3 run_benchmarks_with_vtune.py build-vtune/ soa_versions.csv
```

## Docker Container
The following container contains the dependencies to build and run the benchmark.
Pull and run the container, then follow the steps of the section [Build and Run](#build-and-run).
```
docker pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest
docker run -it --rm registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest bash
```
The corresponding Dockerfile can be found here: [wp1.7-soa-benchmark-image](https://github.com/cern-nextgen/wp1.7-soa-benchmark-image)
