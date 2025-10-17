
struct PxPyPzM {
	double *__restrict__ b0, *__restrict__ b1, *__restrict__ b2, *__restrict__ b3, *__restrict__ b4, *__restrict__ b5, *__restrict__ b6, *__restrict__ b7, *__restrict__ b8, *__restrict__ b9, *__restrict__ b10, *__restrict__ b11, *__restrict__ b12, *__restrict__ b13, *__restrict__ b14, *__restrict__ b15;
    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;
	double *__restrict__ a0, *__restrict__ a1, *__restrict__ a2, *__restrict__ a3, *__restrict__ a4, *__restrict__ a5, *__restrict__ a6, *__restrict__ a7, *__restrict__ a8, *__restrict__ a9, *__restrict__ a10, *__restrict__ a11, *__restrict__ a12, *__restrict__ a13, *__restrict__ a14, *__restrict__ a15;

    static size_t size_bytes(size_t n) { return align_size(sizeof(double[n])) * 36; }

    PxPyPzM(std::byte *buf, size_t n) {
        size_t offset = 0;
		b0 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b1 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b2 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b3 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b4 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b5 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b6 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b7 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b8 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b9 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b10 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b11 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b12 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b13 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b14 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		b15 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));

        x = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        y = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        z = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        M = reinterpret_cast<double *__restrict__>(buf + offset);
		a0 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a1 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a2 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a3 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a4 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a5 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a6 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a7 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a8 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a9 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a10 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a11 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a12 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a13 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a14 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));
		a15 = reinterpret_cast<double *__restrict__>(buf + offset);
		offset += align_size(n * sizeof(double));

    }
};
