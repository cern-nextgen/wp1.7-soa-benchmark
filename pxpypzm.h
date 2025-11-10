
struct PxPyPzM {

    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;


    static size_t size_bytes(size_t n) { return align_size(sizeof(double[n])) * 4; }

    PxPyPzM(std::byte *buf, size_t n) {
        size_t offset = 0;

        x = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        y = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        z = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        M = reinterpret_cast<double *__restrict__>(buf + offset);

    }
};
