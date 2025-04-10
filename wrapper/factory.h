#include "helper.h"
#include "wrapper.h"

namespace factory {

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