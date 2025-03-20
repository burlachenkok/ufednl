#pragma once

#include "dopt/system/include/PlatformSpecificMacroses.h"

namespace dopt
{
    /** If function f(x) is strongly convex with strong convexity parameter "strongConvexityConstant" [m or mu] then f(x)-f* <= 1/2m \cdot \|\grad f(x)\|^2
    * @param grad_f_sqr - squared norm of gradient of function f(x) at point x
    * @param strongConvexityConstant - strong convexity parameter of function f(x)
    * @return suboptimality gap of function f(x) at point x, i.e. f(x) - min(f(x))
    */
    template<class TValue>
    inline TValue functionValueSuboptimalityGap(TValue grad_f_sqr, TValue strongConvexityConstant)
    {
        return grad_f_sqr / (strongConvexityConstant + strongConvexityConstant);
    }

    /** If function f(x) is strongly convex with strong convexity parameter "strongConvexityConstant" [m or mu] then \|x-x_*\|_2 <= 2/m \cdot \|\grad f(x)\|^2
    * @param grad_f_sqr - squared norm of gradient of function f(x) at point x
    * @param strongConvexityConstant - strong convexity parameter of function f(x)
    * @return suboptimality gap to optimal solution, i.e. \|x - x_*\|_2
    */
    template<class TValue>
    inline TValue distanceSuboptimalityGap(TValue grad_f_sqr, TValue strongConvexityConstant)
    {
        return (grad_f_sqr + grad_f_sqr) / strongConvexityConstant;
    }
}
