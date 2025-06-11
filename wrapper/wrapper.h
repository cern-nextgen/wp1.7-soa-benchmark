#ifndef WRAPPER_H
#define WRAPPER_H

#include "helper.h"

#include <cstddef>

namespace wrapper {

enum class layout { aos = 0, soa = 1 };

template <class T>
using value = T;

template <class T>
using reference = T&;  // std::vector<T>::reference;

template <class T>
using const_reference = const T&;  // std::vector<T>::const_reference;

template<
    template <class> class F,
    template <template <class> class> class S,
    layout L
>
struct wrapper;

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::aos> {
    using value_type = S<value>;
    using array_type = F<value_type>;

    constexpr static std::size_t M = helper::CountMembers<value_type>();

    array_type data;

    template <template <class> class F_out>
    operator wrapper<F_out, S, layout::aos>() { return {data}; };

    [[gnu::always_inline]] S<reference> operator[](std::size_t i) { return data[i]; }
    [[gnu::always_inline]] S<const_reference> operator[](std::size_t i) const { return data[i]; }
};

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::soa> : S<F> {
    using value_type = S<value>;
    using array_type = S<F>;

    template <template <class> class F_out>
    operator wrapper<F_out, S, layout::soa>() { return {*this}; };

    [[gnu::always_inline]] S<reference> operator[](std::size_t i) {
        return helper::evaluate_members_at<S, F, reference>(*this, i);
    }
    [[gnu::always_inline]] S<const_reference> operator[](std::size_t i) const {
        return helper::evaluate_members_at<S, F, const_reference>(*this, i);
    }
};

}  // namespace wrapper

#endif  // WRAPPER_H