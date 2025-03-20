#pragma once

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include <stddef.h>

namespace dopt
{
    /* Make one iteration of Jacobi Solver for solving Ax=b in x
    * @param[out] x placeholder for new iterate
    * @param[in] a matrix of linear system
    * @param[in] b right hand side part of linear system
    * @param[in] xPrev x from previous iteration
    * @return true if all is ok
    * @remark https://www-old.math.gatech.edu/academic/courses/core/math2601/Web-notes/4num.pdf, p.51
    * @remark If "A" is p.d. AND matrix has diagonal dominance => method is converging
    */
    template<class Mat, class Vec, class TTtem = typename Vec::TElementType>
    bool jacobiSolverIteration(Vec& x, const Mat& a, const Vec& b,
                               const Vec& xPrev, TTtem epsTolerance = TTtem())
    {
        size_t dim = x.size();
        for (size_t i = 0; i < dim; ++i)
        {
            auto xi = b.get(i);
            for (size_t j = 0; j < i; ++j)
                xi -= a.get(i, j) * xPrev[j]; // Use x from previous iteration
            for (size_t j = i + 1; j < dim; ++j)
                xi -= a.get(i, j) * xPrev[j]; // Use x from previous iteration

            if (dopt::abs(a.get(i, i)) < epsTolerance)
            {
                return false;
            }
            else
            {
                xi = xi / a.get(i, i);
                x.set(i, xi);
            }
        }

        return true;
    }

    /* Make one iteration of Gauss-Seidel iteration for solving Ax=b in x
    * @param[in,out] x placeholder for new iterate and for old iterate
    * @param[in] a matrix of linear system
    * @param[in] b right hand side part of linear system
    * @param[in] xPrev x from previous iteration
    * @return true if all is ok
    * @remark https://www-old.math.gatech.edu/academic/courses/core/math2601/Web-notes/4num.pdf, p.55
    * @remark Golub G.H., Van Loan C.F.- Matrix Computations,p.512. Th.10.1.2  If A=A^T and A is p.d. => methods converges always for any x0
    * @remark If A=A^T a[i,i]>0 => A is p.d. <=> Seidel converges method converges for any inital x0
    */
    template<class Mat, class Vec, class TTtem = typename Vec::TElementType>
    bool seidelSolverIteration(Vec& x, const Mat& a, const Vec& b,
                               TTtem epsTolerance = TTtem())
    {
        size_t dim = x.size();

        for (size_t i = 0; i < dim; ++i)
        {
            auto xi = b[i];

            for (size_t j = 0; j < i; ++j)
                xi -= a.get(i, j) * x[j];

            for (size_t j = i + 1; j < dim; ++j)
                xi -= a.get(i, j) * x[j];
            
            if (dopt::abs(a.get(i, i)) < epsTolerance)
            {
                return false;
            }
            else
            {
                xi = xi / a.get(i, i);
                x.set(i, xi);
            }
        }

        return true;
    }

    /** Conjugate-Gradient method. Initialization
    * @param[out] rawPreviousPreviousOut L2 norm square of residual r=b-Ax one iteration before. (Can be zero for first iteration)
    * @param[out] r residual placeholder where r = b-Ax
    * @param[out] p auxiliary variable for CG implementation
    * @param a matrix of linear system
    * @param b vector of right hand side of linear equalities
    * @param x start iterate
    */
    template<class Mat, class Vec, class TTtem = typename Vec::TElementType>
    void cgSolverInit(TTtem& rawPreviousPreviousOut, Vec& r, Vec& p,
                      const Mat& a, const Vec& b, const Vec& x)
    {
        size_t xSize = x.size();
        rawPreviousPreviousOut = TTtem(0);
        r = b - a * x;
        p = Vec(xSize);
    }

    /** Conjugate gradient method.
    * @param[in,out] x previous iterate and new iterate placeholder
    * @param[in,out] r residual vector (r = b - Ax)
    * @param[in,out] p auxiliary variable for CG implementation
    * @param[in,out] rawPreviousPreviousInOut L2 norm square of residual one iteration before
    * @param[in] a matrix of linear system
    * @param[in] b vector of right hand side of linear equalities
    * @param[in] kIteration iteration number of the method. Increment this if cgSolverIteration() happens more then once.
    * @return true, if all is ok.
    * @remark Proposed by Hestenes and Stiefel and in 1952. 
    * @remark A should be symmetric and p.d.
    * @remark The worst case for method when b is uniform mixture of all eigenvectors and all eigenvalues uniform distributed on some segment of R    
    */
    template<class Mat, class Vec, class TTtem = typename Vec::TElementType>
    bool cgSolverIteration(Vec& x,
                           Vec& r,
                           Vec& p,
                           TTtem& rawPreviousPreviousInOut,
                           const Mat& a, const Vec& b, const size_t kIteration,
                           TTtem epsTolerance = TTtem())
    {
        // C.T. Kelley, page 22
        TTtem rawPrev = r.vectorL2NormSquare();

        if (rawPrev < epsTolerance)
            return false;

        if (kIteration == 0)
            p = r;
        else
            p = r + (rawPrev / rawPreviousPreviousInOut) * p;

        Vec w = a * p;

        TTtem alpha = rawPrev / (p & w);

        x = x + alpha * p;
        r = r - alpha * w;

        rawPreviousPreviousInOut = rawPrev;

        return true;
    }
}
