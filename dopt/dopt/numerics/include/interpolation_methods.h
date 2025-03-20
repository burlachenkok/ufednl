#if 0
#pragma once

#include "dopt/random/include/MathStatistics.h"
#include <math.h>

namespace dopt 
{
    namespace interpolators 
    {
        /** @class Interpolator1Dim
        * @brief Base class for single variable function interpolation methods.
        */
        template<class TCoordX, class TCoordY>
        class Interpolator1Dim
        {
        public:
            typedef TCoordX TCoordTypeX;
            typedef TCoordY TCoordTypeY;

            /** Ctor.
            * @param xValues x values for knots
            * @param yValues y values for knots
            * @param count number of knots
            */
            Interpolator1Dim(const TCoordTypeX* theXValues, 
                             const TCoordTypeY* theYValues, 
                             size_t count,
                             bool xAreSortedFlag)
            : xValues(theXValues)
            , yValues(theYValues)
            , knots(count)
            , xAreSorted(xAreSortedFlag)
            {
            }

            /** Default ctor
            */
            Interpolator1Dim()
            : xValues(nullptr)
            , yValues(nullptr)
            , knots(0)
            , xAreSorted(false)
            {}

            /** Get value at point x
            * @param x point in which you want to evaluate interpolator
            * @param isOk optional boolean variable. If not nullptr then *isOk will be equal to false iff x not lie in the function domain
            * @return return value at point x
            */
            virtual TCoordTypeY interpolate(const TCoordTypeX& x) const = 0;

            /** Receive number of knots
            */
            int size() const {
                return knots;
            }

            bool getXmaxXmin(TCoordTypeX& xmin, TCoordTypeX& xmax) const
            {
                if (size() == 0)
                    return false;
                
                if (xAreSorted)
                {
                    TCoordTypeX front = xValues[0];
                    TCoordTypeX back = xValues[knots - 1];

                    xmin = front;
                    xmax = back;

                    return true;
                }
                else
                {
                    size_t minIndex = 0, maxIndex = 0;
                    dopt::mathstats::maxAndMinIndex(minIndex, maxIndex, xValues, knots);
                    xmin = xValues[minIndex];
                    xmax = xValues[maxIndex];                    
                    return true;
                }
            }

            bool getYmaxYmin(TCoordTypeY& ymin, TCoordTypeY& ymax) const
            {
                if (size() == 0)
                    return false;

                size_t minIndex = 0, maxIndex = 0;
                dopt::mathstats::maxAndMinIndex(minIndex, maxIndex, yValues, knots);
                ymin = yValues[minIndex];
                ymax = yValues[maxIndex];
                return true;
            }

        public:
             const TCoordTypeX* xValues;
             const TCoordTypeY* yValues;
             size_t knots;
             bool xAreSorted;
        };

        /** @class ConstInterpolation
        *   @brief Interpolate function with constant value
        */
        template<class TCoordTypeX, class TCoordTypeY>
        class ConstInterpolation final: public Interpolator1Dim<TCoordTypeX, TCoordTypeY>
        {
        private:
            TCoordTypeX xValues[2];
            TCoordTypeX yValues[2];
            
        public:
            typedef Interpolator1Dim<TCoordTypeX, TCoordTypeY> BaseClass;

            ConstInterpolation(TCoordTypeX xa, TCoordTypeX xb, TCoordTypeY y) 
            : BaseClass(xValues, yValues, 2, true)
            {
                if (xa < xb)
                {
                    xValues[0] = xa;
                    xValues[1] = xb;
                }
                else
                {
                    xValues[0] = xb;
                    xValues[1] = xa;
                }

                yValues[0] = y;
                yValues[1] = y;
            }

            virtual TCoordTypeY interpolate(const TCoordTypeX& x) const
            {
                if (x >= xValues[0] && x <= xValues[1])
                    return yValues[0];
                else
                    return TCoordTypeY(0);
            }
        };

        /** @class LinearInterpolation
        *   @brief Interpolate function with partial linear function
        */
        template<class TCoordTypeX, class TCoordTypeY>
        class LinearInterpolation final: public Interpolator1Dim<TCoordTypeX, TCoordTypeY>
        {
        public:
            typedef Interpolator1Dim<TCoordTypeX, TCoordTypeY> BaseClass;
            
            LinearInterpolation(const TCoordTypeX* xValues,
                                const TCoordTypeY* yValues,
                                size_t count)
            : BaseClass(xValues, yValues, count, true)
            {
            }
            
            virtual TCoordTypeY interpolate(const TCoordTypeX& x) const
            {
                size_t n = BaseClass::size();
                if (n == 0)
                    return TCoordTypeY(0);
                
                for (size_t i = 1; i < n; ++i)
                {
                    if (x >= BaseClass::xValues[i - 1] && x<= BaseClass::xValues[i])
                    {
                        const TCoordTypeX& x1 = BaseClass::xValues[i - 1];
                        const TCoordTypeY& y1 = BaseClass::yValues[i - 1];
                        const TCoordTypeX& x2 = BaseClass::xValues[i];
                        const TCoordTypeY& y2 = BaseClass::yValues[i];
                        
                        return y1 * (x - x2) / (x1 - x2) + y2 * (x - x1) / (x2 - x1);

                        /** Error during using such schema in interval [x_1, x_2] is bounded with
                        * h = x_2 - h_1
                        * M2 = max(|f''(x)|) in x in (x_1, x_2)
                        * |max error| = max |predicted value - real value | <= M2*h^2/8
                        */
                    }
                }

                return TCoordTypeY(0);
            }

            virtual TCoordTypeY interpolateAndGetError(const TCoordTypeX& x, TCoordTypeY& maxSecondDerivative, TCoordTypeY& error) const
            {
                size_t n = BaseClass::size();
                if (n == 0)
                    return TCoordTypeY(0);

               
                for (size_t i = 1; i < n; ++i)
                {

                    if (x >= BaseClass::xValues[i - 1] && x <= BaseClass::xValues[i])
                    {
                        const TCoordTypeX& x1 = BaseClass::xValues[i - 1];
                        const TCoordTypeY& y1 = BaseClass::yValues[i - 1];
                        const TCoordTypeX& x2 = BaseClass::xValues[i];
                        const TCoordTypeY& y2 = BaseClass::yValues[i];

                        return y1 * (x - x2) / (x1 - x2) + y2 * (x - x1) / (x2 - x1);

                        /** Error during using such schema in interval [x_1, x_2] is bounded with
                        * h = x_2 - h_1
                        * M2 = max(|f''(x)|) in x in (x_1, x_2)
                        * |max error| = max |predicted value - real value | <= M2*h^2/8
                        */
                        
                        error = maxSecondDerivative * (x2 - x1) * (x2 - x1) / 8.0;
                    }
                }

                return TCoordTypeY(0);
            }
            
            /** Assume that we know upper bound on second derivative of the unknown function, which we awnt to interpolate.
            * Get step s.t. max deviation of the unknown function from this partial linear intrerpolation is less then eps.
            * @param maxAbsSecondDerivative max(|f''(x)|) or estimate of this
            * @param eps max( |f(x) - interpolation(x)| )
            * @return size of the step
            */
            static double recomendationAboutTheStep(double maxAbsSecondDerivative, double eps) {
                return sqrt(8.0 * eps / maxAbsSecondDerivative);
            }
        };
    }
}
#endif
