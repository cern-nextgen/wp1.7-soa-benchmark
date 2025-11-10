
struct Snbody {

    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;


    static size_t size_bytes(size_t n) { return align_size(sizeof(float[n])) * 6; }

    Snbody(std::byte *buf, size_t n) {
        size_t offset = 0;

        x = reinterpret_cast<float *__restrict__>(buf);
        offset += align_size(n * sizeof(float));
        y = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        z = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vx = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vy = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vz = reinterpret_cast<float *__restrict__>(buf + offset);

    }
};
