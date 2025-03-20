#pragma once

#include "dopt/optimization_problems/include/problem_specification.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/random/include/RandomGenRealLinear.h"
#include "dopt/copylocal/include/Copier.h"
#include "dopt/linalg_vectors/include/LightVectorND.h"
#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <stddef.h>
#include <iostream>

namespace dopt
{
    /** L2 regularized logistic regression written in a form loss function
    *   for logistic loss in terms of used classification margins.
    *
    *   f(x) = 1/m \sum_i log(1 + exp(-bi(ai' * x))) + \lambda/2 \|x\|^2
    */
    template<class Mat, class TVec, class TElemenType>
    class L2RegulirizeLogisticRegression final: public ProblemSpecification
    {
    public:
        virtual ~L2RegulirizeLogisticRegression() = default;

        L2RegulirizeLogisticRegression(const Mat& theXFeaturesTranspose,
                                       const TVec& theBLabels,
                                       TElemenType theLambda)
            : lambda(theLambda)
            , lambda_half(theLambda / 2.0)
        {
            assert(theXFeaturesTranspose.columns() == theBLabels.size());

            size_t myM = theXFeaturesTranspose.columns();        // number of samples
            size_t inDimension = theXFeaturesTranspose.rows();   // number of atributes

            invM = 1.0 / myM;                                    // iverse of number of samples
            minusInvM = -invM;                                   // minus iverse of number of samples
            
            Atr = Mat(inDimension, myM);

            size_t readPosForColumnI = 0;
            size_t writePosForColumnI = 0;
                
            for (size_t i = 0; i < myM; ++i,
                                        readPosForColumnI += theXFeaturesTranspose.LDA, 
                                        writePosForColumnI += Atr.LDA
                )
            {
                dopt::LightVectorND<TVec> colOfXTr(&(const_cast<Mat&>(theXFeaturesTranspose).matrixByCols[readPosForColumnI]), inDimension);
                dopt::LightVectorND<TVec> colOfAtr(&Atr.matrixByCols[writePosForColumnI], inDimension);

                // Special Logic because n Binary Classification with logistic loss in such form class is always in {-1, 1}
                using TItemType = typename TVec::TElementType;

                TItemType bi = theBLabels.get(i);

                if (bi == TItemType(1))
                {
                    colOfAtr.assignAllValues(colOfXTr);
                }
                else if (bi == TItemType(-1))
                {
                    colOfAtr.assignWithVectorMultiple(colOfXTr, TItemType(-1));
                }
                else
                {
                    assert(!"SOMETHING WRONG");
                    colOfAtr.assignAllValues(colOfXTr);
                }

#if 0
                Atr.setColumn(i, theXFeaturesTranspose.getColumn(i) * bLabels.get(i));
#endif
            }
        }

        TElemenType operator()(const TVec& x)
        {
            return evaluateFunction(x);
        }

        /** For train sample pair "(aj,bj)" evaluate vector (bj <aj, x>) and put result into margin[j]
        * @param x current model
        * @return vector of margins.
        * @note this is a well-known quanity in classical Machine Learning. Easy motivation if bj and <aj,x> have the same sign and big it is very good and loss should be small for this pair. 
        */
        TVec evaluateClassificationMargin(const TVec& x)
        {
            TVec margin = Mat::matrixVectorMultiplyWithPreTranspose(Atr, x);
            return margin;
        }

        /** Evaluate sigmoid of classification margin
        * @param classificationMargin classification margin
        * @return vector of sigmoid([classificationMargin]_i), where sigmoid(x)=1.0/(1.0 + e^(-x))
        */
        TVec evaluateClassificationMarginSigmoid(const TVec& classificationMargin)
        {
            return classificationMargin.elementwiseSigmoid();
        }

        TElemenType evaluateFunction(const TVec& x)
        {
#if 0
            TVec margin = A * x;   // margin for classifier
#else
            TVec margin = Mat::matrixVectorMultiplyWithPreTranspose(Atr, x);
#endif            
            return evaluateFunction(x, margin);
        }

        TElemenType evaluateFunction(const TVec& x, const TVec& classificationMargin)
        {
            TElemenType res = TVec::template logisticUnweightedLossFromMargin<TElemenType>(classificationMargin);
            return res + lambda_half * x.vectorL2NormSquare();
        }

        TElemenType evaluateFunction(const TVec& x, const TVec& classificationMargin, const TVec& classificationMarginSigmoid)
        {
            TElemenType res = TVec::template logisticUnweightedLossFromMarginSigmoid<TElemenType>(classificationMarginSigmoid);
            return res + lambda_half * x.vectorL2NormSquare();
        }

        TVec evaluateGradient(const TVec& x)
        {
            TVec ax = Mat::matrixVectorMultiplyWithPreTranspose(Atr, x);
            return evaluateGradient(x, ax);
        }

        TVec evaluateGradient(const TVec& x, const TVec& classificationMargin)
        {
            size_t myM = getNumberOfTrainingSamples();
            double minusInvM = this->minusInvM;

            const TVec& ax = classificationMargin;

            TVec g = TVec::getUninitializedVector(myM);

            for (size_t i = 0; i < myM; ++i)
            {
                g[i] = minusInvM / (1.0 + exp(ax[i]));
            }

            return Mat::matrixVectorMultiply(Atr, g, lambda, x);
        }

        TVec evaluateGradient(const TVec& x,
            const TVec& classificationMargin,
            const TVec& classificationMarginSigmoid)
        {
            double minusInvM = this->minusInvM;

            const TVec& ax = classificationMargin;
            const TVec& sigmoid_ax = classificationMarginSigmoid;

            // 1 - ( 1 / (1+ exp(-ax)) ) = (1+ exp(-ax)) / (1+ exp(-ax)) - ( 1 / (1+ exp(-ax)) ) = exp(-ax)/(1+ exp(-ax)) = 1 / (exp(ax) + 1)
            // g[i] =  (-1/m) * (1 / ( 1 + exp(ax[i]) )
            TVec g = TVec::scaledDifferenceWithEye(1, sigmoid_ax, minusInvM);
            return Mat::matrixVectorMultiply(Atr, g, lambda, x);
        }

        template<bool kSymmetriseHessian = true, size_t kAlgortihm = 4>
        Mat evaluateHessian(const TVec& x)
        {
            TVec ax = Mat::matrixVectorMultiplyWithPreTranspose(Atr, x);
            return evaluateHessian<kSymmetriseHessian, kAlgortihm>(x, ax);
        }

        template<bool kSymmetriseHessian = true, size_t kAlgortihm = 4>
        Mat evaluateHessian(const TVec& x, const TVec& classificationMargin)
        {
            TVec classificationMarginSigmoid = classificationMargin.elementwiseSigmoid();
            return evaluateHessian<kSymmetriseHessian, kAlgortihm>(x, classificationMargin, classificationMarginSigmoid);
        }

        template<bool kSymmetriseHessian = true, size_t kAlgortihm = 4>
        Mat evaluateHessian(const TVec& x, const TVec& classificationMargin, const TVec& classificationMarginSigmoid)
        {
            const size_t d = getInputVariableDimension();
            size_t myM = getNumberOfTrainingSamples();

            double myinvM = invM;

            const TVec& ax = classificationMargin;
            const TVec& sigmoid_ax = classificationMarginSigmoid;

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD
            constexpr bool kSimdIsSupported = true;
#else
            constexpr bool kSimdIsSupported = false;
#endif

            if constexpr (kAlgortihm == 1)
            {
                // Naive Hessian Evaluation Algorithm:
                //  1. Copmpute need dirived quanitites for diagonal matrix in analytical formula for Hessian oracle for Logistic Regression
                //  2. Form the diagonal multiplication part (in form of vector)
                //  3. Perfor need multiplications for Hessian oracle
                //  4. Add regulirization term contibution into hessian with special add-to-diagonal operation.

                TVec D = TVec::getUninitializedVector(myM);

                for (size_t i = 0; i < myM; ++i)
                {
                    double z1 = (1 - sigmoid_ax[i]); // z1 =  1 - 1 / (1 + exp(-ax) ) = (1 + exp(-ax) )/(1 + exp(-ax) ) - 1 / (1 + exp(-ax) ) =...= 1 / (1 + exp(ax) )
                    double z2 = z1 * sigmoid_ax[i];  // z2 = 1 / (1 + exp(ax) ) * 1 / (1 + exp(-ax) ) = 1 / (1 + exp(ax) ) * exp(ax) / (1 + exp(ax) ) =  exp(ax) / { (1+exp(ax))^2 }
                    D[i] = myinvM * z2;
                }

                Mat hessian = Atr * Mat::multiplyDiagonalByDense(D, Atr.getTranspose());
                hessian.addToAllDiagonalEntries(lambda);
                return hessian;
            }
            else if constexpr (kAlgortihm == 2 || !kSimdIsSupported)
            {
                // Better Hessain Oracle implementation: Non-Vectorized version with sum of Rank-1 matrices.
                //  1. Copmpute need dirived quanitites for diagonal matrix in analytical formula for Hessian oracle for Logistic Regression
                //  2. Form the diagonal multiplication part (in form of vector)
                //  3. Perfor need multiplications for Hessian oracle, but look into multiplication as series of summ of rank-1 matrices. Dimension of outter product is "d", and number elements in the sum is "m".
                //
                // According to experiments it's better oracles in practice, especially if CPU does not have any SIMD support (or it is prohibited to use)
                //
                TVec D = TVec::getUninitializedVector(myM);

                for (size_t i = 0; i < myM; ++i)
                {
                    double z1 = (1 - sigmoid_ax[i]); // z1 =  1 / (1 + exp(ax) )
                    double z2 = z1 * sigmoid_ax[i];  // z2 =  exp(ax) / { (1+exp(ax))^2 }

                    D[i] = myinvM * z2;
                }

                Mat hessian(d, d);

                for (size_t ii = 0; ii < myM; ++ii)                      // loop through samples
                {
                    dopt::LightVectorND<TVec> lightColumnU(&Atr.getRaw(0, ii), d);
                    dopt::LightVectorND<TVec> lightColumnV(&Atr.getRaw(0, ii), d);

                    auto d_ii = D[ii];

                    {
                        // semantically: hessian += D[i] * Mat::outerProduct(lightColumn, lightColumn) [but only for upper triangular part]

                        for (size_t j = 0; j < d; ++j)                    // loop through columns
                        {
                            auto vj_d_ii = d_ii * lightColumnV.get(j); // scale column element by diaongal

                            size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                            for (size_t i = 0; i <= j; ++i, write_pos++)  // loop through rows
                            {
                                hessian.matrixByCols[write_pos] += lightColumnU.get(i) * vj_d_ii; // perform need addition
                            }
                        }
                    }
                }

                // Symmetrize hessian. Copy elements from upper triangular part excluding diagonal.
                if constexpr (kSymmetriseHessian)
                {
                    hessian.symmetrizeLowerTriangInPlace();
                }

                // Add lambda to diagonal
                hessian.addToAllDiagonalEntries(lambda);
                return hessian;
            }

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD
            if constexpr (kAlgortihm == 3)
            {
                // Vectorized version of Algorithm - 2. Better Hessain Oracle implementation - implicitly vectorized version with sum of Rank - 1 matrices.

                typedef typename dopt::VectorSimdTraits<TElemenType, dopt::cpu_extension>::VecType VecType;
                constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

                TVec D = TVec::getUninitializedVector(myM);

                {
                    const size_t sz = myM;
                    size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(myM);

                    size_t i = 0;

                    VecType sigmoid_ax_reg;
                    VecType z1_reg;
                    VecType z2_reg;

                    for (; i < items; i += kVecBatchSize)
                    {
                        sigmoid_ax_reg.load(&sigmoid_ax.getRaw(i));
                        z1_reg = TElemenType(1) - sigmoid_ax_reg;
                        z2_reg = z1_reg * sigmoid_ax_reg;
                        (z2_reg * invM).store(&D.getRaw(i));
                    }
                    for (; i < sz; ++i)
                    {
                        double z1 = (1 - sigmoid_ax[i]); // z1 =  1 / (1 + exp(ax) )
                        double z2 = z1 * sigmoid_ax[i];  // z2 =  exp(ax) / { (1+exp(ax))^2 }
                        D[i] = myinvM * z2;
                    }
                }

                Mat hessian(d, d);

                VecType column_u_vec, orig_vec, res_vec;

                for (size_t ii = 0; ii < myM; ++ii)                      // loop through samples
                {
                    dopt::LightVectorND<TVec> lightColumnU(&Atr.getRaw(0, ii), d);
                    dopt::LightVectorND<TVec> lightColumnV(&Atr.getRaw(0, ii), d);

                    auto d_ii = D[ii];

                    {
                        // hessian += D[i] * Mat::outerProduct(lightColumn, lightColumn);
                        
                        for (size_t j = 0; j < d; ++j)                   // loop through columns
                        {
                            auto vj_d_ii = d_ii * lightColumnV.get(j);
                            VecType vj_d_ii_vec(vj_d_ii);                // scale column element by diaongal and fill vector register with this value
                            
                            size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                            const size_t sz = j + 1;

                            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

                            size_t i = 0;                                                     // starting from 0 row
                            for (; i < items; i += kVecBatchSize, write_pos += kVecBatchSize) // loop through rows
                            {
                                orig_vec.load(&hessian.matrixByCols[write_pos]);
                                column_u_vec.load(&lightColumnU.getRaw(i));    
#if SUPPORT_CPU_FMA_EXT
                                // (1st * 2nd) + 3rd [for explicit FMA]
                                res_vec = ::mul_add(column_u_vec, vj_d_ii_vec, orig_vec);
#else
                                // for situation when FMA is not available or it is implicitly implemented (and actually works better)
                                res_vec = column_u_vec * vj_d_ii_vec + orig_vec;
#endif
                                // write back result to destination
                                res_vec.store(&hessian.matrixByCols[write_pos]);
                            }

                            for (; i < sz; ++i, ++write_pos)                                  // loop through rest rows
                            {
                                hessian.matrixByCols[write_pos] += lightColumnU.get(i) * vj_d_ii;
                            }
                        }
                    }
                }

                // Symmetrize Hessian. Copy elements from upper triangular part excluding diagonal.
                if constexpr (kSymmetriseHessian)
                {
                    hessian.symmetrizeLowerTriangInPlace();
                }

                // Add lambda to diagonal
                hessian.addToAllDiagonalEntries(lambda);
                return hessian;
            }
            else if constexpr (kAlgortihm == 4)
            {
                // Vectorized version of Algorithm - 2. Better Hessain Oracle implementation - implicitly vectorized version with sum of Rank - 1 matrices.
                // The diffrence with Algorithm - 3 is that this version uses batches of samples to hold temporary results in vector register for longer.
                typedef typename dopt::VectorSimdTraits<TElemenType, dopt::cpu_extension>::VecType VecType;
                constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

                Mat hessian(d, d);

                if (myM != 0)
                {
                    {
                        // Tunable constant (For possible exploiting ILP)
                        constexpr size_t kVecBatchSizeSamplesMulitplier = 1;           
                        
                        // Number of samples in "virtual batch"
                        constexpr size_t kVecBatchSizeSamples = kVecBatchSize * kVecBatchSizeSamplesMulitplier;

                        // Temporaly variables
                        VecType orig_vec;
                        VecType column_u_vec;

                        // Current index sample
                        size_t ii = 0;

                        // Final index where we need to stop batching by samples process                       
                        size_t myMBatched = dopt::roundToNearestMultipleDown<kVecBatchSizeSamples>(myM);

                        for (ii = 0; ii < myMBatched; ii += kVecBatchSizeSamples)                      // loop through samples
                        {
                            VecType D_ii[kVecBatchSizeSamplesMulitplier];

                            for (size_t kk = 0, k_reg = 0; kk < kVecBatchSizeSamples; kk += kVecBatchSize, k_reg++)
                            {
                                VecType sigmoid;
                                sigmoid.load_a(&sigmoid_ax.getRaw(ii + kk));
                                VecType Z1 = (VecType(1.0) - sigmoid); // z1 =  1 / (1 + exp(ax) )
                                VecType Z2 = Z1 * sigmoid;             // z2 =  exp(ax) / { (1+exp(ax))^2 }
                                D_ii[k_reg] = myinvM * Z2;             // d_ii = 1/m * exp(ax) / { (1+exp(ax))^2 }
                            }

                            // Load U, V in rank-one representaion for all samples in the batch with size "kVecBatchSizeSamples"
                            dopt::LightVectorND<TVec> lightColumnU[kVecBatchSizeSamples];
                            dopt::LightVectorND<TVec> lightColumnV[kVecBatchSizeSamples];

                            for (size_t kk = 0; kk < kVecBatchSizeSamples; ++kk)
                            {
                                lightColumnU[kk].components = lightColumnV[kk].components = &Atr.getRaw(0, ii + kk);
                                lightColumnU[kk].componentsCount = lightColumnV[kk].componentsCount = d;
                            }

                            // Compute summ of rank-1 matrices in Hessian representation across "kVecBatchSizeSamples"
                            {
                                // Hessian += D[i] * Mat::outerProduct(lightColumnU, lightColumnV);
                                
                                for (size_t j = 0; j < d; ++j)                      // loop through columns
                                {
                                    TElemenType vj_d_ii[kVecBatchSizeSamples];

                                    // v[j] * d_ii for each element in batch
                                    {
                                        size_t kkk = 0;

                                        for (size_t k_reg = 0; k_reg < kVecBatchSizeSamplesMulitplier; k_reg++)
                                        {
                                            for (size_t k = 0; k < kVecBatchSize; ++k, ++kkk)
                                            {
                                                vj_d_ii[kkk] = D_ii[k_reg][k] * lightColumnV[kkk].get(j);
                                            }
                                        }
                                    }

                                    // write position
                                    size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                                    // number of items to write
                                    const size_t sz = j + 1;

                                    // where to stop compute batching by samples process
                                    size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

                                    size_t i = 0;                                                      // starting from 0 row
                                    for (; i < items; i += kVecBatchSize, write_pos += kVecBatchSize)  // loop through rows
                                    {
                                        // load original vector from result matrix
                                        orig_vec.load_a(&hessian.matrixByCols[write_pos]);

                                        for (size_t kk = 0; kk < kVecBatchSizeSamples; ++kk)
                                        {
                                            // load u column for batch
                                            column_u_vec.load_a(&lightColumnU[kk].getRaw(i));

#if SUPPORT_CPU_FMA_EXT
                                            // (1st * 2nd) + 3rd
                                            orig_vec = ::mul_add(column_u_vec, vj_d_ii[kk], orig_vec);
#else
                                            // for situation when FMA is not available or it is implicitly implemented (and actually works better)
                                            
                                            // vec - scalar multiplicatino
                                            column_u_vec *= vj_d_ii[kk];
                                            
                                            // addition to result
                                            orig_vec += column_u_vec;
                                            
                                            // no write! (in contrast to previous method). Benefit: store for a longer result in vector register
#endif
                                        }

                                        // write back result to destination
                                        orig_vec.store_a(&hessian.matrixByCols[write_pos]);
                                    }

                                    // process the rest
                                    for (; i < sz; ++i, ++write_pos)
                                    {
                                        for (size_t kk = 0; kk < kVecBatchSizeSamples; ++kk)
                                        {
                                            hessian.matrixByCols[write_pos] += lightColumnU[kk].get(i) * vj_d_ii[kk];
                                        }
                                    }
                                }
                            }
                        }

                        // Process the rest samples which can not be fit into bach of samples
                        for (; ii < myM; ++ii)                  // loop through samples
                        {
                            double z1 = (1.0 - sigmoid_ax[ii]); // z1 =  1 / (1 + exp(ax) )
                            double z2 = z1 * sigmoid_ax[ii];    // z2 =  exp(ax) / { (1+exp(ax))^2 }
                            double d_ii = myinvM * z2;

                            dopt::LightVectorND<TVec> lightColumnU(&Atr.getRaw(0, ii), d);
                            dopt::LightVectorND<TVec> lightColumnV(&Atr.getRaw(0, ii), d);

                            {
                                // hessian += D[i] * Mat::outerProduct(lightColumnU, lightColumnV);
                                
                                for (size_t j = 0; j < d; ++j)                      // loop through columns
                                {
                                    auto vj_d_ii = d_ii * lightColumnV.get(j);
                                    VecType vj_d_ii_vec(vj_d_ii);

                                    size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                                    const size_t sz = j + 1;

                                    size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

                                    size_t i = 0;
                                    for (; i < items; i += kVecBatchSize, write_pos += kVecBatchSize)   // loop through rows
                                    {
                                        orig_vec.load_a(&hessian.matrixByCols[write_pos]);

                                        column_u_vec.load_a(&lightColumnU.getRaw(i));
                                        
#if SUPPORT_CPU_FMA_EXT
                                        // (1st * 2nd) + 3rd  [for explicit FMA]
                                        orig_vec = ::mul_add(column_u_vec, vj_d_ii_vec, orig_vec);
#else
                                        orig_vec += column_u_vec * vj_d_ii_vec;
#endif
                                        orig_vec.store_a(&hessian.matrixByCols[write_pos]);
                                    }

                                    for (; i < sz; ++i, ++write_pos)                                    // loop through rows
                                    {
                                        hessian.matrixByCols[write_pos] += lightColumnU.get(i) * vj_d_ii;
                                    }
                                }
                            }
                        }
                    }

                    // Symmetrize hessian. Copy elements from upper triangular part excluding diagonal.
                    if constexpr (kSymmetriseHessian)
                        hessian.symmetrizeLowerTriangInPlace();

                    // Add lambda to diagonal
                    hessian.addToAllDiagonalEntries(lambda);
                }

                return hessian;
            }
            else if constexpr (kAlgortihm == 5)
            {
                // Vectorized version of Algorithm - 2. Better Hessain Oracle implementation - implicitly vectorized version with sum of Rank - 1 matrices.
                // Similar to Algorithm - 4 is that this version uses batches of samples to hold temporary results in vector register for longer.
                // In contrast to Algorithm - 4 it has first loop (first sample) to be completely unrolled:
                //   1. Benefit - we use less read operations (in first iteration we know that hessian is zero)
                //   2. Drawback - cached code for first iteration and decoded instruction will not actually will be used in CPU except this first iteration

                typedef typename dopt::VectorSimdTraits<TElemenType, dopt::cpu_extension>::VecType VecType;
                constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

                Mat hessian(d, d);

                if (myM != 0) [[likely]]
                {
                    // Special unroll first update with less iterations
                    {
                        constexpr size_t ii = 0;
                        VecType column_u_vec, orig_vec;

                        // Process first samples
                        {
                            double z1 = (1.0 - sigmoid_ax[ii]); // z1 =  1 / (1 + exp(ax) )
                            double z2 = z1 * sigmoid_ax[ii];    // z2 =  exp(ax) / { (1+exp(ax))^2 }
                            double d_ii = myinvM * z2;

                            dopt::LightVectorND<TVec> lightColumnU(&Atr.getRaw(0, ii), d);
                            dopt::LightVectorND<TVec> lightColumnV(&Atr.getRaw(0, ii), d);

                            {
                                // hessian += D[i] * Mat::outerProduct(lightColumnU, lightColumnV);
                                
                                for (size_t j = 0; j < d; ++j)                      // loop through columns
                                {
                                    auto vj_d_ii = d_ii * lightColumnV.get(j);
                                    VecType vj_d_ii_vec(vj_d_ii);

                                    size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                                    const size_t sz = j + 1;

                                    size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

                                    size_t i = 0;
                                    for (; i < items; i += kVecBatchSize, write_pos += kVecBatchSize)  // loop through rows
                                    {
                                        column_u_vec.load(&lightColumnU.getRaw(i));
                                        
                                        orig_vec = column_u_vec * vj_d_ii_vec;

                                        orig_vec.store(&hessian.matrixByCols[write_pos]);
                                    }

                                    for (; i < sz; ++i, ++write_pos)                                  // loop through rows
                                    {
                                        hessian.matrixByCols[write_pos] = lightColumnU.get(i) * vj_d_ii;
                                    }
                                }
                            }
                        }
                    }

                    {
                        VecType column_u_vec, orig_vec;
                        
                        for (size_t ii = 1; ii < myM; ++ii)     // Loop through remaining samples
                        {
                            double z1 = (1.0 - sigmoid_ax[ii]); // z1 =  1 / (1 + exp(ax) )
                            double z2 = z1 * sigmoid_ax[ii];    // z2 =  exp(ax) / { (1+exp(ax))^2 }
                            double d_ii = myinvM * z2;

                            dopt::LightVectorND<TVec> lightColumnU(&Atr.getRaw(0, ii), d);
                            dopt::LightVectorND<TVec> lightColumnV(&Atr.getRaw(0, ii), d);

                            {
                                // hessian += D[i] * Mat::outerProduct(lightColumnU, lightColumnV);
                                for (size_t j = 0; j < d; ++j)
                                {
                                    auto vj_d_ii = d_ii * lightColumnV.get(j);
                                    VecType vj_d_ii_vec(vj_d_ii);

                                    size_t write_pos = hessian.template getFlattenIndexFromColumn</*i*/0>(j);

                                    const size_t sz = j + 1;

                                    size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

                                    size_t i = 0;
                                    for (; i < items; i += kVecBatchSize, write_pos += kVecBatchSize)  // loop through rows
                                    {
                                        orig_vec.load(&hessian.matrixByCols[write_pos]);
                                        column_u_vec.load(&lightColumnU.getRaw(i));
                                        
#if SUPPORT_CPU_FMA_EXT
                                        // (1st * 2nd) + 3rd
                                        orig_vec = ::mul_add(column_u_vec, vj_d_ii_vec, orig_vec);
#else
                                        orig_vec += column_u_vec * vj_d_ii_vec;
#endif
                                        orig_vec.store(&hessian.matrixByCols[write_pos]);
                                    }

                                    for (; i < sz; ++i, ++write_pos)                                  // loop through rows
                                    {
                                        hessian.matrixByCols[write_pos] += lightColumnU.get(i) * vj_d_ii;
                                    }
                                }
                            }
                        }
                    }

                    // Symmetrize hessian. Copy elements from upper triangular part excluding diagonal.
                    if constexpr (kSymmetriseHessian)
                        hessian.symmetrizeLowerTriangInPlace();

                    // Add lambda to diagonal
                    hessian.addToAllDiagonalEntries(lambda);
                }
                else
                {
                    hessian.addToAllDiagonalEntries(lambda);
                }
                
                return hessian;
            }
#endif

        }

        /** Get number of train samples
        * @return number of train samples
        */
        size_t getNumberOfTrainingSamples() const {
            return Atr.columns();
        }

        /** Get input dimension of optimization variable
        * @return dimension
        */
        size_t getInputVariableDimension() override
        {
            return Atr.rows();
        }

        bool isGradientOracleAvailable() override
        {
            return true;
        }

        bool isHessianOracleAvailable() override
        {
            return true;
        }

        TElemenType computeMuStrongConvexity()
        {
            return lambda;
        }

        TElemenType computeLSmoothness(double epsTolerance = 0.001, size_t* iterations = nullptr)
        {
            Mat A = Atr.getTranspose();

            Mat hessianBound = Atr * A * invM / 4.0;
            TVec v(hessianBound.rows());

            dopt::RandomGenRealLinear rnd_generator = dopt::RandomGenRealLinear(0.0, 1.0);
            v.setAllRandomly(rnd_generator);

            TVec vPrev = v;
            vPrev /= vPrev.vectorLinfNorm();
            size_t numIteration = 0;

            TElemenType maxEigenValue = TElemenType();

            for (;; ++numIteration)
            {
                v = hessianBound * vPrev;
                maxEigenValue = v.vectorLinfNorm();
                v /= maxEigenValue;

                if ((v - vPrev).vectorLinfNorm() < epsTolerance) {
                    break;
                }

                dopt::CopyHelpers::swap(v, vPrev);
            }

            if (iterations)
                *iterations = numIteration;

            return maxEigenValue + lambda;
        }
        //=========================================================//
    private:
        double invM;        ///< Inverse number of train samples
        double minusInvM;   ///< Minus inverse number of train samples
        double lambda;      ///< L2 regularization coefficient
        double lambda_half; ///< L2 regularization coefficient divided by two
        Mat Atr;            ///< Transpose of matrix A. Each column j is a scalar-vector product of vector of feature a[j] by sign of train data b[j] \in {-1,+1}.
        
    };
}
