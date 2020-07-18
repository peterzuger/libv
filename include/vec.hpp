/**
 * @file   libv/include/vec.hpp
 * @author Peter Züger
 * @date   18.07.2020
 * @brief  N dimensional vector
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Peter Züger
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef LIBV_VEC_HPP
#define LIBV_VEC_HPP

#include <algorithm>
#include <array>
#include <iterator>
#include <math.h>
#include <numeric>
#include <utility>

namespace v{
    template<typename T, std::size_t N>
    class vec{
        std::array<T, N> p;

    public:
        using value_type             = T;
        using pointer                = value_type*;
        using const_pointer          = const value_type*;
        using reference              = value_type&;
        using const_reference        = const value_type&;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using iterator               = pointer;
        using const_iterator         = const_pointer;
        using reverse_iterator       = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        void swap(vec& other)noexcept(std::is_nothrow_swappable_v<value_type>){
            p.swap(other.p);
        }

        void fill(const_reference u){
            p.fill(u);
        }

        const vec& normalize(){
            return *this /= magnitude();
        }

        value_type mean()const{
            return sum() / N;
        }

        const value_type dot(const vec& o)const{
            return std::transform_reduce(p.first(), p.last(),
                                         o.p.first(),
                                         value_type{});
        }

        constexpr const std::enable_if_t<N == 3, vec>& cross(const vec& o)const{
            return {
                (p[1] * o.p[2]) - (p[2] * o.p[1]),
                (p[2] * o.p[0]) - (p[0] * o.p[2]),
                (p[0] * o.p[1]) - (p[1] * o.p[0])
            };
        }

        constexpr const value_type magnitude2()const{
            return std::transform_reduce(p.first(), p.last(),
                                         p.first(),
                                         value_type{});
        }

        const value_type magnitude()const{
            return std::sqrt(magnitude2());
        }

        const value_type sum()const{
            return std::accumulate(p.first(), p.second(),
                                   value_type{});
        }

        const value_type prod()const{
            return std::accumulate(++p.first(), p.last(),
                                   *p.first(),
                                   std::multiplies<value_type>{});
        }

        const value_type min()const{
            return *std::min_element(p.first(), p.last());
        }

        const value_type max()const{
            return *std::max_element(p.first(), p.last());
        }


        constexpr const vec& operator+=(const vec& rhs)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] += rhs.p[i];
            }
            return *this;
        }

        constexpr const vec& operator-=(const vec& rhs)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] -= rhs.p[i];
            }
            return *this;
        }

        constexpr const vec& operator*=(const vec& rhs)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] *= rhs.p[i];
            }
            return *this;
        }

        constexpr const vec& operator/=(const vec& rhs)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] /= rhs.p[i];
            }
            return *this;
        }


        constexpr const vec& operator*=(const_reference v)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] *= v;
            }
            return *this;
        }

        constexpr const vec& operator/=(const_reference v)noexcept{
            for(size_type i = 0; i < N; ++i){
                p[i] /= v;
            }
            return *this;
        }


        constexpr reference operator[](size_type n){
            return p[n];
        }

        constexpr const_reference operator[](size_type n)const{
            return p[n];
        }
    };


    template<typename T, std::size_t N>
    constexpr vec<T, N> normalize(vec<T, N> v)noexcept{
        return v.normalize();
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> dot(vec<T, N> a, const vec<T, N>& b)noexcept{
        return a.dot(b);
    }


    template<typename T, std::size_t N>
    constexpr vec<T, N> operator+(vec<T, N> lhs, const vec<T, N>& rhs)noexcept{
        return lhs += rhs;
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> operator-(vec<T, N> lhs, const vec<T, N>& rhs)noexcept{
        return lhs -= rhs;
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> operator*(vec<T, N> lhs, const vec<T, N>& rhs)noexcept{
        return lhs *= rhs;
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> operator/(vec<T, N> lhs, const vec<T, N>& rhs)noexcept{
        return lhs /= rhs;
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> operator*(vec<T, N> lhs, const T& rhs)noexcept{
        return lhs *= rhs;
    }

    template<typename T, std::size_t N>
    constexpr vec<T, N> operator/(vec<T, N> lhs, const T& rhs)noexcept{
        return lhs /= rhs;
    }


    template<typename T, std::size_t N>
    constexpr bool operator==(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return lhs.p == rhs.p;
    }

    template<typename T, std::size_t N>
    constexpr bool operator!=(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return !(lhs == rhs);
    }


    template<typename T, std::size_t N>
    constexpr bool operator<(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return lhs.p < rhs.p;
    }

    template<typename T, std::size_t N>
    constexpr bool operator>(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return rhs < lhs;
    }

    template<typename T, std::size_t N>
    constexpr bool operator<=(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return !(lhs > rhs);
    }

    template<typename T, std::size_t N>
    constexpr bool operator>=(const vec<T, N>& lhs, const vec<T, N>& rhs)noexcept{
        return !(lhs < rhs);
    }
}

#endif /* LIBV_VEC_HPP */
