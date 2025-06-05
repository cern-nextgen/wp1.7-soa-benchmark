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

template <template <class> class F_in, template <template <class> class> class S>
struct proxy_type : S<F_in> {
    constexpr static std::size_t M = helper::CountMembers<S<value>>();
    template<template <class> class F_out>
    [[gnu::always_inline]] operator S<F_out>() const {
        return helper::cast_type<M, S<F_in>, S<F_out>>(*this);
    }
};

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

    value_type& get_reference(std::size_t i) { return data[i]; }
    const value_type& get_reference(std::size_t i) const { return data[i]; }

    [[gnu::always_inline]] proxy_type<reference, S> operator[](std::size_t i) {
        using output_type = proxy_type<reference, S>;
        return helper::cast_type<M, array_type&, output_type>(data[i]);
    }
    [[gnu::always_inline]] proxy_type<const_reference, S> operator[](std::size_t i) const {
        using output_type = proxy_type<const_reference, S>;
        return helper::cast_type<M, const array_type&, output_type>(data[i]);
    }
};

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::soa> {
    using value_type = S<value>;
    using array_type = S<F>;

    constexpr static std::size_t M = helper::CountMembers<value_type>();

    array_type data;

    template <template <class> class F_out>
    [[gnu::always_inline]] operator wrapper<F_out, S, layout::soa>() {
        using output_type = wrapper<F_out, S, layout::soa>;
        return helper::cast_type<M, array_type&, output_type>(data);
    };

    [[gnu::always_inline]] proxy_type<reference, S> operator[](std::size_t i) {
        using output_type = proxy_type<reference, S>;
        return helper::evaluate_members_at<M, array_type&, output_type>(data, i);
    }
    [[gnu::always_inline]] proxy_type<const_reference, S> operator[](std::size_t i) const {
        using output_type = proxy_type<const_reference, S>;
        return helper::evaluate_members_at<M, const array_type&, output_type>(data, i);
    }
};

}  // namespace wrapper

#endif  // WRAPPER_H