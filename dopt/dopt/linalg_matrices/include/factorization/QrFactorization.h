#pragma once

#include <algorithm>
#include <vector>

#include <math.h>
#include <stddef.h>

namespace dopt
{
    /** Various interesting things that can be derived from QR factorization
    */
    template<class Mat>
    struct QrFactorizeInfo
    {
        Mat& Q; ///< placeholder for Q matrix 
        Mat& R; ///< placeholder for R matrix

        typedef typename Mat::MatrixColumn MatrixColumn;
        typedef typename Mat::MatrixRow MatrixRow;

        QrFactorizeInfo(Mat& theQ, Mat& theR)
            : Q(theQ)
            , R(theR)
        {}

        /** Get number of rows in matrix A for which we made QR decomposition
        * @return number of rows in A or which is the same as rows in Q.
        */
        size_t rowsInA() const
        {
            return Q.rows();
        }

        /** Get number of columns in matrix A for which we made QR decomposition
        * @return number of columns in A or which is the same as columns in R.
        */
        size_t columnsInA() const
        {
            return R.columns();
        }

        /** Get dimension of the range of operator A, i.e. rank(A)
        * @return rank of A or which is the same as number of columns Q.
        */
        size_t getDimOfRangeA() const {
            return Q.columns();
        }

        /* Check that matrix A = QR is full rank, iff rank(A(m,n)) = min(m, n)
        * @return true if A is full rank
        */
        bool isFullRank() const
        {
            size_t my_rows = rowsInA();
            size_t my_cols = rowsInA();
            size_t min_rows_and_cols = my_rows < my_cols ? my_rows : my_cols;
            return getDimOfRangeA() == min_rows_and_cols;
        }

        /** Get orthonormal basis for the range of A. In fact it is a columns in the matrix Q.
        * @return matrix which hold by column by column this orthonormal basis
        */
        Mat& getOrthoBasisForRangeA() const
        {
            return Q;
        }

        /** Evaluate A times X
        * @return evaluated vector
        */
        MatrixColumn evaluateAX(const MatrixColumn& x) const
        {
            return Q * (R * x);
        }
    };

    namespace qrFactorization
    {
        /** Full QR factorization for A=[Q1 Q2] * [R 0]^T.
        * - Any columns vector in Q2 is orthogonal to any vector in Q1.
        * - [Q1 Q2] is square matrix n*n.
        * - Q1 and Q2 are complementary subspaces for R^n and this spaces are orthogonal to each other.
        * @param Q1 Part of block matrix [Q1 Q2]. Q1 is column orthogonal matrix.
        * @param Q2 Part of block matrix [Q1 Q2]. Q2 is column orthogonal matrix.
        * @param R1 If do usual QR factorization A=QR=Q1 R1, then it is a part of it. Number of rows in R equal to rank of matrix A.
        * @param A input rectangular matrix
        * @param nullItem value which is used as threshold for zero
        */
        template<class Mat>
        void qrFactorizeFull(Mat* outQ1, Mat* outQ2, Mat* outR1,
                             const Mat& A,
                             typename Mat::TElementType nullValue = typename Mat::TElementType())
        {
            Mat R1(A.columns(), A.columns());

            typedef typename Mat::TElementType ItemType;
            typedef typename Mat::MatrixColumn MatrixColumn;

            std::vector<MatrixColumn> q1;    // future q1 columns
            std::vector<MatrixColumn> q2;    // future q2 columns

            size_t rankQ1 = 0;

            {
                std::vector<MatrixColumn> a;
                for (size_t i = 0; i < A.columns(); ++i)
                    a.push_back(A.getColumn(i));

                // array a contains columns of A

                // q'[3] =  a[3] - (q[1]&a[3])*q[1] - (q[2]&a[3])*q[2]
                //  q[3] = q'[3]/ |q'[3]|
                // => a[3] = (q[1]&a[3])*q[1] + (q[2]&a[3])*q[2] + q'[3] = (q[1]&a[3])*q[1] + (q[2]&a[3])*q[2] + |q'[3]| * q[3]  (**)

                size_t aSize = a.size();

                for (size_t i = 0; i < aSize; ++i)
                {
                    MatrixColumn qTest = a[i];

                    // Orthogonalization with respect to things which we have processed
                    for (size_t j = 0; j < rankQ1; ++j)
                    {
                        qTest -= (q1[j] & a[i]) * q1[j];
                        R1.set(j, i, q1[j] & a[i]);
                        // (a[i] & q[j])*q[j] in signal processing was named as "innovation" in statistical context.
                    }

                    // normalization part
                    auto len = sqrt(qTest.vectorL2NormSquare());

                    if (len >= nullValue)
                    {
                        R1.set(rankQ1, i, len);  // store item like (q'[3]) from equation **
                        qTest /= len;          // normalize
                        q1.push_back(qTest);   // update matrix Q
                        rankQ1 += 1;           // update rank
                    }
                    else
                    {
                        /* column of A [0...i] are not linear independent.
                           Do not push qTest into Q and do not spend R for store zero item.
                           Don't increment rank
                        */
                    }
                }
            }

            size_t rankQ2 = 0;
            {
                // Create Q2 (TODO: maximum rank max(m,n) - #columns(Q1))
                Mat idMatrix = Mat::getIdentitySquareMatrix(A.rows());
                std::vector<MatrixColumn> a2;
                for (size_t i = 0; i < idMatrix.columns(); ++i)
                    a2.push_back(idMatrix.getColumn(i));

                size_t a2Size = a2.size();

                for (size_t i = 0; i < a2Size; ++i)
                {
                    MatrixColumn qTest = a2[i];

                    // Orthogonalization with respect to things which we have processed during this stage
                    for (size_t j = 0; j < rankQ2; ++j)
                        qTest -= (q2[j] & a2[i]) * q2[j];

                    // orthogonalization with respect to the q1 vectors which we received in previous stage
                    size_t q1Size = q1.size();

                    for (size_t j = 0; j < q1Size; ++j)
                        qTest -= (q1[j] & a2[i]) * q1[j];

                    // normalization part
                    auto len = sqrt(qTest.vectorL2NormSquare());

                    if (len >= nullValue)
                    {
                        qTest /= len;
                        q2.push_back(qTest);
                        rankQ2 += 1;
                    }
                }
            }

            // Write down results back
            // 
            // Create result Q1 matrix
            if (outQ1)
            {
                *outQ1 = Mat(A.rows(), rankQ1);
                for (size_t i = 0; i < A.rows(); ++i)
                {
                    for (size_t j = 0; j < rankQ1; ++j)
                    {
                        const MatrixColumn& qColumn = q1[j];
                        outQ1->set(i, j, qColumn[i]);
                    }
                }
            }
            // Create result R1 matrix
            if (outR1)
            {
                *outR1 = Mat(rankQ1, A.columns());
                for (size_t i = 0; i < rankQ1; ++i)
                {
                    for (size_t j = 0; j < A.columns(); ++j)
                    {
                        outR1->set(i, j, R1.get(i, j));
                    }
                }
            }
            // Create result Q2 matrix
            if (outQ2)
            {
                *outQ2 = Mat(A.rows(), q2.size());
                for (size_t i = 0; i < A.rows(); ++i)
                {
                    for (size_t j = 0; j < q2.size(); ++j)
                    {
                        const MatrixColumn& qColumn = q2[j];
                        outQ2->set(i, j, qColumn[i]);
                    }
                }
            }

            // Range(Q2) == Nullspace(A transpose)
        }

        //==========================================================================================//

        /** Find orthogonal nullspace basis for matrix A.
        * @param A input matrix for which we are looking for it's nullspace
        * @param nullValue numerical value which is considered as null
        * @return matrix which by columns store orthonormal basis for null space of A
        */
        template<class Mat>
        Mat orthoBasis4NullSpace(Mat& A, typename Mat::TElementType nullValue = typename Mat::TElementType()) 
        {
            // After full QR factorization it can be shown that: Q1^T * z = 0 => z in the range of Q2 iff A^T * z = 0.
            // So Range(Q2) = NullSpace(A^T)
            // And Range(A) = span(Q1) (by full QR factorization construction)
            //
            // => As Range(Q1) is complementary to Range(Q2) => NullSpace(A^T) (+,ortho) Range(A) = R^n
            // Not idea is to apply full QR factorization for A^T and then Q2=NullSpace(A^T^T)=NullSpace(A)
            Mat aTr = A.getTranspose();
            Mat Q2(1, 1);
            qrFactorizeFull(static_cast<Mat*>(nullptr), &Q2, static_cast<Mat*>(nullptr), aTr, nullValue);
            return Q2;
        }

        /** Find orthogonal basis for range of A.
        * @param A input matrix       
        * @param nullValue numerical value which is considered as null
        * @return matrix which by columns store orthonormal basis for null space of A
        */
        template<class Mat>
        Mat orthoBasis4Range(Mat& A, typename Mat::TElementType nullValue = typename Mat::TElementType())
        {
            Mat Q1(1, 1);
            qrFactorizeFull(&Q1, static_cast<Mat*>(nullptr), static_cast<Mat*>(nullptr), A, nullValue);
            return Q1;
        }
    }
}
