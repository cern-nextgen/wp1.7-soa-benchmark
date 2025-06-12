#ifndef FACTORY_H
#define FACTORY_H

#include <array>
#include <vector>

#include "allocator.h"
#include "helper.h"
#include "wrapper.h"

namespace factory {

namespace pmr {

template <class T>
struct vector : std::vector<T, allocator::BufferAllocator<T>> {
    using Base = std::vector<T, allocator::BufferAllocator<T>>;
    vector(allocator::BufferAllocator<T> allocator) : Base(allocator.size_ / sizeof(T), allocator) {}
};

}

template <class T>
struct buffer {
    using type = T;
    operator allocator::BufferAllocator<T>() { return {begin, size}; };
    std::byte * begin;
    std::size_t size;
};

template <template <template <class> class> class S, wrapper::layout L>
constexpr std::size_t get_size_in_bytes() {
    if constexpr (L == wrapper::layout::aos) {
        return sizeof(S<wrapper::value>);
    } else if constexpr (L == wrapper::layout::soa) {
        std::size_t total_size = 0;
        auto sum_sizes = [&total_size](auto member, std::size_t m) -> decltype(auto) {
            total_size += sizeof(typename decltype(member)::type);
            return member;
        };
        S<buffer> S_buffer;
        helper::apply_to_members<S, buffer, buffer>(S_buffer, sum_sizes);
        return total_size;
    }
}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::soa> buffer_wrapper_soa(S<buffer> buffers) {
    return {S<allocator::BufferAllocator>(buffers)};
}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::aos> buffer_wrapper_aos(buffer<S<wrapper::value>> buffer) {
    return {allocator::BufferAllocator<S<wrapper::value>>(buffer)};
}

template <template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<pmr::vector, S, L> buffer_wrapper(std::byte* buffer_ptr, std::size_t bytes) {
    if constexpr (L == wrapper::layout::aos) {
        return buffer_wrapper_aos<S>({buffer_ptr, bytes});
    } else if constexpr (L == wrapper::layout::soa) {
        constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
        std::array<std::size_t, M + 1> buffer_offset = {0};
        std::size_t total_size = 0;
        auto sizeof_members = [&buffer_offset, &total_size](auto member, std::size_t m) -> decltype(auto) {
            buffer_offset[m + 1] = sizeof(typename decltype(member)::type);
            total_size += buffer_offset[m + 1];
            return member;
        };
        S<buffer> buffers;
        helper::apply_to_members<S, buffer, buffer>(buffers, sizeof_members);

        std::size_t N = bytes / total_size;
        for (int i = 1; i < M + 1; ++i) { buffer_offset[i] = buffer_offset[i - 1] + buffer_offset[i] * N; }

        auto create_buffers = [&buffer_offset, buffer_ptr](auto member, std::size_t m) -> decltype(auto) {
            using T = typename decltype(member)::type;
            return buffer<T>{buffer_ptr + buffer_offset[m], buffer_offset[m + 1] - buffer_offset[m]};
        };
        buffers = helper::apply_to_members<S, buffer, buffer>(buffers, create_buffers);

        return buffer_wrapper_soa<S>(buffers);
    }
}

}  // namespace factory

#endif  // FACTORY_H