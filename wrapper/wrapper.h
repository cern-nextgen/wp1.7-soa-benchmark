#ifndef WRAPPER_H
#define WRAPPER_H

#include "helper.h"

#include <cstddef>

namespace wrapper {

enum class layout { aos = 0, soa = 1 };

template <class T>
using value = T;

template <class T>
using reference = T&;

template <class T>
using const_reference = const T&;

template<
    template <template <class> class> class S,
    template <class> class F,
    layout L
>
struct wrapper;

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, layout::aos> {
    F<S<value>> data;

    template <template <class> class F_out>
    operator wrapper<S, F_out, layout::aos>() { return {data}; };

    [[gnu::always_inline]] S<reference> operator[](std::size_t i) { return data[i]; }
    [[gnu::always_inline]] S<const_reference> operator[](std::size_t i) const { return data[i]; }
};

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, layout::soa> : S<F> {
    template <template <class> class F_out>
    operator wrapper<S, F_out, layout::soa>() { return {*this}; };

    [[gnu::always_inline]] S<reference> operator[](std::size_t i) {
        return helper::evaluate_members_at<S, F, reference>(*this, i);
    }
    [[gnu::always_inline]] S<const_reference> operator[](std::size_t i) const {
        return helper::evaluate_members_at<S, F, const_reference>(*this, i);
    }
};

}  // namespace wrapper

#endif  // WRAPPER_H