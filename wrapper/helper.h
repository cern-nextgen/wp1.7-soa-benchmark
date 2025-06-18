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

template <class T> using value = T;

template <template <template <class> class> class S>
constexpr std::size_t CountMembers() {
    using R = S<value>;
    if constexpr (detail::is_aggregate_constructible_from_n<R, 2>::value) return 2;
    else if constexpr (detail::is_aggregate_constructible_from_n<R,  10>::value) return  10;
    else if constexpr (detail::is_aggregate_constructible_from_n<R,  64>::value) return  64;
    else return 100;  // Silence warnings about missing return value
}

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class FunctionObject
>
[[gnu::always_inline]] constexpr S<F_out> apply_to_members(S<F_in> & s, FunctionObject f) {
    constexpr std::size_t M = helper::CountMembers<S>();
    if constexpr (M == 2) {
        auto& [m00, m01] = s;
        return {f(m00), f(m01)};
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = s;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08), f(m09)};
    } else if constexpr (M == 64) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09,
               m10, m11, m12, m13, m14, m15, m16, m17, m18, m19,
               m20, m21, m22, m23, m24, m25, m26, m27, m28, m29,
               m30, m31, m32, m33, m34, m35, m36, m37, m38, m39,
               m40, m41, m42, m43, m44, m45, m46, m47, m48, m49,
               m50, m51, m52, m53, m54, m55, m56, m57, m58, m59,
               m60, m61, m62, m63] = s;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08), f(m09),
                f(m10), f(m11), f(m12), f(m13), f(m14), f(m15), f(m16), f(m17), f(m18), f(m19),
                f(m20), f(m21), f(m22), f(m23), f(m24), f(m25), f(m26), f(m27), f(m28), f(m29),
                f(m30), f(m31), f(m32), f(m33), f(m34), f(m35), f(m36), f(m37), f(m38), f(m39),
                f(m40), f(m41), f(m42), f(m43), f(m44), f(m45), f(m46), f(m47), f(m48), f(m49),
                f(m50), f(m51), f(m52), f(m53), f(m54), f(m55), f(m56), f(m57), f(m58), f(m59),
                f(m60), f(m61), f(m62), f(m63)};
    } else return {};
}

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class FunctionObject
>
[[gnu::always_inline]] constexpr S<F_out> apply_to_members(const S<F_in> & s, FunctionObject f) {
    return apply_to_members<F_out, F_in, S>(const_cast<S<F_in>&>(s), f);
}

}  // namespace helper

#endif  // HELPER_H