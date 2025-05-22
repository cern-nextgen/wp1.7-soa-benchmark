#ifndef FACTORY_H
#define FACTORY_H

#include <array>
#include <numeric>
#include <vector>
#include <type_traits>

#include "allocator.h"
#include "helper.h"
#include "wrapper.h"

namespace factory {

namespace pmr {

template <class T>
using vector = std::vector<T, allocator::BufferAllocator<T>>;

}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::aos> buffer_wrapper_aos(std::byte* buffer, std::size_t bytes) {
    using value_type = S<wrapper::value>;
    std::size_t size = bytes / sizeof(value_type);
    allocator::BufferAllocator<value_type> alloc(buffer, bytes);
    pmr::vector<value_type> data(size, alloc);
    return wrapper::wrapper<pmr::vector, S, wrapper::layout::aos>(std::move(data));
}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::soa> buffer_wrapper_soa(std::byte* buffer, std::size_t bytes) {
    using value_type = S<wrapper::value>;
    constexpr std::size_t M = helper::CountMembers<value_type>();
    auto size_of = [](auto& member, std::size_t) -> std::size_t { return sizeof(member); };
    std::array<std::size_t, M> member_bytes = helper::apply_to_members<M, value_type, std::array<std::size_t, M>>(value_type(), size_of);
    std::size_t N = bytes / std::reduce(member_bytes.cbegin(), member_bytes.cend());

    std::array<std::size_t, M + 1> buffer_bytes = {0};
    for (int i = 1; i < M + 1; ++i) {
        buffer_bytes[i] = member_bytes[i - 1] * N + buffer_bytes[i - 1];
    }

    auto get_member_vector = [N, buffer, buffer_bytes](auto& member, std::size_t m) -> decltype(auto) {
        using member_type = typename std::remove_reference<decltype(member)>::type;
        allocator::BufferAllocator<member_type> alloc(buffer + buffer_bytes[m], buffer_bytes[m + 1] - buffer_bytes[m]);
        return pmr::vector<member_type>(N, alloc);
    };

    return helper::apply_to_members<M, value_type, wrapper::wrapper<pmr::vector, S, wrapper::layout::soa>>(value_type(), get_member_vector);
}

template <template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<pmr::vector, S, L> buffer_wrapper(std::byte* buffer, std::size_t bytes) {
    if constexpr (L == wrapper::layout::aos) return buffer_wrapper_aos<S>(buffer, bytes);
    else if constexpr (L == wrapper::layout::soa) return buffer_wrapper_soa<S>(buffer, bytes);
}

template <
    template <class> class F,
    template <template <class> class> class S
>
wrapper::wrapper<F, S, wrapper::layout::aos> default_aos_wrapper(std::size_t N) {
    return { F<S<wrapper::value>>(N) };
}

template <
    template <class> class F,
    template <template <class> class> class S
>
wrapper::wrapper<F, S, wrapper::layout::soa> default_soa_wrapper(std::size_t N) {
    constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
    auto forward_to_F_constructor = [N](auto member, std::size_t) -> decltype(auto) { return F<decltype(member)>(N); };
    return { helper::apply_to_members<M, S<wrapper::value>, S<F>>(S<wrapper::value>(), forward_to_F_constructor)  };
}

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
wrapper::wrapper<F, S, L> default_wrapper(std::size_t N) {
    if constexpr (L == wrapper::layout::aos) return default_aos_wrapper<F, S>(N);
    else if constexpr (L == wrapper::layout::soa) return default_soa_wrapper<F, S>(N);
}

}  // namespace factory

#endif  // FACTORY_H