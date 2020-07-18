/**
 * @file   libv/include/vec3.hpp
 * @author Peter Züger
 * @date   18.07.2020
 * @brief  fixed 3 dimensional vector
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
#ifndef LIBV_VEC3_HPP
#define LIBV_VEC3_HPP

#include <array>
#include <math.h>
#include <algorithm>
#include <stdexcept>

namespace v{
    namespace impl{
        [[noreturn]] inline void throw_out_of_range(const char* s){
#if defined(__cpp_exceptions)
            throw std::out_of_range(s);
#else
            (void)s;
            terminate();
#endif /* defined(__cpp_exceptions) */
        }
    }

    template<typename T>
    class vec3{
        std::array<T, 3> p;

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

        vec3() = default;

        vec3(const_reference x, const_reference y, const_reference z):
            p{x, y, z}{}

        void swap(vec3& other)noexcept(std::is_nothrow_swappable_v<value_type>){
            p.swap(other.p);
        }

        void fill(const_reference u){
            p.fill(u);
        }

        reference x()noexcept{return p[0];}
        reference y()noexcept{return p[1];}
        reference z()noexcept{return p[2];}

        const_reference x()const noexcept{return p[0];}
        const_reference y()const noexcept{return p[1];}
        const_reference z()const noexcept{return p[2];}

        void x(const_reference v)noexcept{p[0] = v;}
        void y(const_reference v)noexcept{p[1] = v;}
        void z(const_reference v)noexcept{p[2] = v;}

        const vec3& normalize()noexcept{
            return *this /= magnitude();
        }

        value_type mean()const noexcept{
            return sum() / 3;
        }

        const value_type dot(const vec3& o)const noexcept{
            return (p[0] * o.p[0]) + (p[1] * o.p[1]) + (p[2] * o.p[2]);
        }

        constexpr const vec3& cross(const vec3& o)const noexcept{
            return {
                (p[1] * o.p[2]) - (p[2] * o.p[1]),
                (p[2] * o.p[0]) - (p[0] * o.p[2]),
                (p[0] * o.p[1]) - (p[1] * o.p[0])
            };
        }

        constexpr value_type magnitude2()const noexcept{
            return (p[0] * p[0]) + (p[1] * p[1]) + (p[2] * p[2]);
        }

        const value_type magnitude()const noexcept{
            return std::sqrt(magnitude2());
        }

        const value_type sum()const noexcept{
            return p[0] + p[1] + p[2];
        }

        const value_type prod()const noexcept{
            return p[0] * p[1] * p[2];
        }

        const value_type min()const noexcept{
            return *std::min_element(p.first(), p.last());
        }

        const value_type max()const noexcept{
            return *std::max_element(p.first(), p.last());
        }


        constexpr const vec3& operator-=(const vec3& rhs)noexcept{
            p[0] -= rhs.p[0];
            p[1] -= rhs.p[1];
            p[2] -= rhs.p[2];
            return *this;
        }

        constexpr const vec3& operator+=(const vec3& rhs)noexcept{
            p[0] += rhs.p[0];
            p[1] += rhs.p[1];
            p[2] += rhs.p[2];
            return *this;
        }

        constexpr const vec3& operator*=(const vec3& v)noexcept{
            p[0] *= v.p[0];
            p[1] *= v.p[1];
            p[2] *= v.p[2];
            return *this;
        }

        constexpr const vec3& operator/=(const vec3& v)noexcept{
            p[0] /= v.p[0];
            p[1] /= v.p[1];
            p[2] /= v.p[2];
            return *this;
        }


        constexpr const vec3& operator*=(const_reference v)noexcept{
            p[0] *= v;
            p[1] *= v;
            p[2] *= v;
            return *this;
        }

        constexpr const vec3& operator/=(const_reference v)noexcept{
            p[0] /= v;
            p[1] /= v;
            p[2] /= v;
            return *this;
        }


        constexpr reference operator[](size_type n){
            return p[n];
        }

        constexpr const_reference operator[](size_type n)const{
            return p[n];
        }

        constexpr reference at(size_type n){
            if(n >= 3)
                impl::throw_out_of_range("v::vec3::at");
            return p[n];
        }

        constexpr const_reference at(size_type n)const{
            if(n >= 3)
                impl::throw_out_of_range("v::vec3::at");
            return p[n];
        }
    };


    template<typename T>
    constexpr vec3<T> normalize(vec3<T> v)noexcept{
        return v.normalize();
    }

    template<typename T>
    constexpr vec3<T> dot(vec3<T> a, const vec3<T>& b)noexcept{
        return a.dot(b);
    }


    template<typename T>
    constexpr vec3<T> operator+(vec3<T> lhs, const vec3<T>& rhs)noexcept{
        return lhs += rhs;
    }

    template<typename T>
    constexpr vec3<T> operator-(vec3<T> lhs, const vec3<T>& rhs)noexcept{
        return lhs -= rhs;
    }

    template<typename T>
    constexpr vec3<T> operator*(vec3<T> lhs, const vec3<T>& rhs)noexcept{
        return lhs *= rhs;
    }

    template<typename T>
    constexpr vec3<T> operator/(vec3<T> lhs, const vec3<T>& rhs)noexcept{
        return lhs /= rhs;
    }

    template<typename T>
    constexpr vec3<T> operator*(vec3<T> lhs, const T& rhs)noexcept{
        return lhs *= rhs;
    }

    template<typename T>
    constexpr vec3<T> operator/(vec3<T> lhs, const T& rhs)noexcept{
        return lhs /= rhs;
    }


    template<typename T>
    constexpr bool operator==(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return lhs.p == rhs.p;
    }

    template<typename T>
    constexpr bool operator!=(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return !(lhs == rhs);
    }


    template<typename T>
    constexpr bool operator<(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return lhs.p < rhs.p;
    }

    template<typename T>
    constexpr bool operator>(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return rhs < lhs;
    }

    template<typename T>
    constexpr bool operator<=(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return !(lhs > rhs);
    }

    template<typename T>
    constexpr bool operator>=(const vec3<T>& lhs, const vec3<T>& rhs)noexcept{
        return !(lhs < rhs);
    }
}

#endif /* LIBV_VEC3_HPP */
