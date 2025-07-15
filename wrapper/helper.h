#ifndef HELPER_H
#define HELPER_H

#include <cstddef>

#include "decorator.h"

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

template <class Argument>
constexpr std::size_t CountMembers() {
    if constexpr (detail::is_aggregate_constructible_from_n<Argument, 2>::value) return 2;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  10>::value) return  10;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  64>::value) return  64;
    else return 100;  // Silence warnings about missing return value
}

template <
    class Argument,
    class FunctionObject
>
DECORATOR() constexpr auto invoke(Argument & arg, FunctionObject&& f) {
    constexpr std::size_t M = helper::CountMembers<Argument>();
    if constexpr (M == 2) {
        auto& [m00, m01] = arg;
        return f(m00, m01);
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09);
    } else if constexpr (M == 64) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09,
               m10, m11, m12, m13, m14, m15, m16, m17, m18, m19,
               m20, m21, m22, m23, m24, m25, m26, m27, m28, m29,
               m30, m31, m32, m33, m34, m35, m36, m37, m38, m39,
               m40, m41, m42, m43, m44, m45, m46, m47, m48, m49,
               m50, m51, m52, m53, m54, m55, m56, m57, m58, m59,
               m60, m61, m62, m63] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09,
                m10, m11, m12, m13, m14, m15, m16, m17, m18, m19,
                m20, m21, m22, m23, m24, m25, m26, m27, m28, m29,
                m30, m31, m32, m33, m34, m35, m36, m37, m38, m39,
                m40, m41, m42, m43, m44, m45, m46, m47, m48, m49,
                m50, m51, m52, m53, m54, m55, m56, m57, m58, m59,
                m60, m61, m62, m63);
    } else return void();  // Silence warnings about missing return value
}

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class  FunctionObject
>
struct memberwise {
    FunctionObject f;

    template <class... Args>
    DECORATOR() constexpr S<F_out> operator()(Args&... args) const { return {f(args)...}; }
};

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class  FunctionObject
>
DECORATOR() constexpr S<F_out> invoke_on_members(S<F_in> & s, FunctionObject&& f) {
    return invoke(s, memberwise<F_out, F_in, S, FunctionObject>{f});
}

}  // namespace helper

#endif  // HELPER_H