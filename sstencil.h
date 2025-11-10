
struct Sstencil {

    double *__restrict__ src, *__restrict__ dst, *__restrict__ rhs;


    static size_t size_bytes(size_t n) { return align_size(sizeof(double[n])) * 3; }

    Sstencil(std::byte *buf, size_t n) {
        size_t offset = 0;

        src = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        dst = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        rhs = reinterpret_cast<double *__restrict__>(buf + offset);

    }
};
