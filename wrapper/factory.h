#ifndef FACTORY_H
#define FACTORY_H

#include <array>
#include <vector>
#include <numeric>

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

struct Buffer {
    template <class T>
    operator allocator::BufferAllocator<T>() { return {begin, size}; };
    std::byte * begin;
    std::size_t size;
};

template <class T>
using AllBuffer = Buffer;

template <class T>
using AllBufferPtr = Buffer *;

struct get_sizeof {
    template <class T>
    std::size_t operator()(T&) const { return sizeof(T); }
};

template <class T>
using Allsize_t = std::size_t;

template <class T>
using value = T;

template <template <template <class> class> class S, wrapper::layout L>
constexpr std::size_t get_size_in_bytes() {
    if constexpr (L == wrapper::layout::aos) {
        return sizeof(S<wrapper::value>);
    } else if constexpr (L == wrapper::layout::soa) {
        constexpr static std::size_t M = helper::CountMembers<S>();
        std::array<std::size_t, M> sizes = helper::apply_to_members<Allsize_t, value>(S<value>(), get_sizeof());
        return std::reduce(sizes.cbegin(), sizes.cend());
    }
}

template <template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<S, pmr::vector, L> buffer_wrapper(std::byte* buffer_ptr, std::size_t bytes) {
    if constexpr (L == wrapper::layout::aos) {
        return {allocator::BufferAllocator<S<wrapper::value>>(buffer_ptr, bytes)};
    } else if constexpr (L == wrapper::layout::soa) {
        constexpr static std::size_t M = helper::CountMembers<S>();

        std::array<std::size_t, M> sizes = helper::apply_to_members<Allsize_t, value>(S<value>(), get_sizeof());
        std::size_t N = bytes / std::reduce(sizes.cbegin(), sizes.cend());

        S<AllBuffer> S_buffer;
        auto get_pointer = [] (Buffer& buffer) -> Buffer* { return &buffer; };
        std::array<Buffer*, M> buffer_ref_array = helper::apply_to_members<AllBufferPtr, AllBuffer>(S_buffer, get_pointer);

        std::size_t offset = 0;
        for (int m = 0; m < M; ++m) {
            std::size_t step = sizes[m] * N;
            *buffer_ref_array[m] = {buffer_ptr + offset, sizes[m]};
            offset += step;
        }

        return S<allocator::BufferAllocator>(S_buffer);
    }
}

}  // namespace factory

#endif  // FACTORY_H