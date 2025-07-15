#ifndef DECORATOR_H
#define DECORATOR_H

#ifdef __CUDACC__
#define DECORATOR() __host__ __device__
#else
#define DECORATOR()
#endif

#endif  // DECORATOR_H
