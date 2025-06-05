#ifndef HELPER_H
#define HELPER_H

#include <cstddef>

namespace helper {

namespace detail {

struct UniversalType {
    template<class T>
    operator T() const;
};

template <typename T, typename Is, typename=void>
struct is_aggregate_constructible_from_n_impl : std::false_type {};

template <typename T, std::size_t...Is>
struct is_aggregate_constructible_from_n_impl<T, std::index_sequence<Is...>, std::void_t<decltype(T{(void(Is), UniversalType{})...})>> : std::true_type {};

template <typename T, std::size_t N>
using is_aggregate_constructible_from_n_helper = is_aggregate_constructible_from_n_impl<T, std::make_index_sequence<N>>;

template <typename T, std::size_t N>
struct is_aggregate_constructible_from_n {
    constexpr static bool value = is_aggregate_constructible_from_n_helper<T, N>::value && !is_aggregate_constructible_from_n_helper<T, N+1>::value;
};

}  // namespace detail

template <class T>
constexpr std::size_t CountMembers() {
    if constexpr (detail::is_aggregate_constructible_from_n<T, 2>::value) return 2;
    else if constexpr (detail::is_aggregate_constructible_from_n<T,  10>::value) return  10;
    else if constexpr (detail::is_aggregate_constructible_from_n<T,  64>::value) return  64;
    else return 100;  // Silence warnings about missing return value
}

template <std::size_t M, class T, class S, class Functor>
[[gnu::always_inline]] constexpr S apply_to_members(T t, Functor&& f) {
    if constexpr (M == 2) {
        auto& [m00, m01] = t;
        return {f(m00, 0), f(m01, 1)};
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5), f(m06, 6), f(m07, 7), f(m08, 8), f(m09, 9)};
    } else if constexpr (M == 64) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09,
            m10, m11, m12, m13, m14, m15, m16, m17, m18, m19,
            m20, m21, m22, m23, m24, m25, m26, m27, m28, m29,
            m30, m31, m32, m33, m34, m35, m36, m37, m38, m39,
            m40, m41, m42, m43, m44, m45, m46, m47, m48, m49,
            m50, m51, m52, m53, m54, m55, m56, m57, m58, m59,
            m60, m61, m62, m63] = t;
        return {f(m00,  0), f(m01,  1), f(m02,  2), f(m03,  3), f(m04,  4), f(m05,  5), f(m06,  6), f(m07,  7), f(m08,  8), f(m09,  9),
                f(m10, 10), f(m11, 11), f(m12, 12), f(m13, 13), f(m14, 14), f(m15, 15), f(m16, 16), f(m17, 17), f(m18, 18), f(m19, 19),
                f(m20, 20), f(m21, 21), f(m22, 22), f(m23, 23), f(m24, 24), f(m25, 25), f(m26, 26), f(m27, 27), f(m28, 28), f(m29, 29),
                f(m30, 30), f(m31, 31), f(m32, 32), f(m33, 33), f(m34, 34), f(m35, 35), f(m36, 36), f(m37, 37), f(m38, 38), f(m39, 39),
                f(m40, 40), f(m41, 41), f(m42, 42), f(m43, 43), f(m44, 44), f(m45, 45), f(m46, 46), f(m47, 47), f(m48, 48), f(m49, 49),
                f(m50, 50), f(m51, 51), f(m52, 52), f(m53, 53), f(m54, 54), f(m55, 55), f(m56, 56), f(m57, 57), f(m58, 58), f(m59, 59),
                f(m60, 60), f(m61, 61), f(m62, 62), f(m63, 63)};
    } else return {};
}

}  // namespace helper

#endif  // HELPER_H