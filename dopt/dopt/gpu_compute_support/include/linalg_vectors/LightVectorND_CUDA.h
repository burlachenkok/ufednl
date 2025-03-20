#pragma once

#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/VectorND_CUDA_Raw_kernels.h"
#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"

#include "dopt/copylocal/include/Copier.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

// Non-blocking operation primitives
#include "dopt/system/include/threads/Thread.h"

#include <vector>
#include <sstream>
#include <iostream>

#include <math.h>
#include <assert.h>
#include <stddef.h>

namespace dopt
{    
    /** Light vector view. It's a view into some elements of the linear vector.
    * @tparam T Type of elements inside vector
    * @remark The vector does not copy any elements. Please use it only if you're really understand what you're doing.
    */
    template <typename Vec>
    class LightVectorND_CUDA
    {
    public:
        using TElementType = typename Vec::TElementType;   ///< Typedef for elements types
        
        size_t componentsCount;                            ///< Components count inside vector view
        TElementType* components;                          ///< Components of the vector
        mutable GpuManagement gpuManagement;               ///< Device in which data is allocated
        
        /** Ctor. Create light view into parentVector[theStartIndex:theStartIndex+theCount]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        * @param theCount number of items which should be in LightVectorND_CUDA
        */
        LightVectorND_CUDA(Vec& parentVector, size_t theStartIndex, size_t theCount)
        : components(parentVector.rawDevData() + theStartIndex)
        , componentsCount(theCount)
        , gpuManagement(parentVector.device())
        {}

        /** Ctor. Create light view into parentVector[theStartIndex:to the end]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        */
        LightVectorND_CUDA(Vec& parentVector, size_t theStartIndex)
        : components(parentVector.rawDevData() + theStartIndex)
        , componentsCount(0)
        , gpuManagement(parentVector.device())
        {
            size_t parentVectorSize = parentVector.size();

            if (theStartIndex >= parentVectorSize) [[unlikely]]
            {
                components = nullptr;
                componentsCount = 0;
            }
            else
            {
                componentsCount = parentVectorSize - theStartIndex;
            }
        }

        /** Ctor. Create light view into parentVector[theStartIndex:theStartIndex+theCount]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        * @param theCount number of items which should be in LightVectorND_CUDA
        * @reamrk Using the complete(full) template identifier within a template definition scope is not essential.
        */
        LightVectorND_CUDA(LightVectorND_CUDA& rhs, size_t theStartIndex, size_t theCount)
        : components(rhs.components + theStartIndex)
        , componentsCount(theCount)
        , gpuManagement(rhs.gpuManagement)
        {}

        /** Ctor. Create light view into parentVector[theStartIndex:to the end]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        */
        LightVectorND_CUDA(LightVectorND_CUDA& parentVector, size_t theStartIndex)
        : components(parentVector.rawDevData() + theStartIndex)
        , componentsCount(0)
        , gpuManagement(parentVector.gpuManagement)
        {
            size_t parentVectorSize = parentVector.size();

            if (theStartIndex >= parentVectorSize) [[unlikely]]
            {
                components = nullptr;
                componentsCount = 0;
            }
            else
            {
                componentsCount = parentVectorSize - theStartIndex;
            }
        }

        /** Create vector from some arbitarily buffer in memory
        * @param theDevComponents beginning of buffer of elements in memory
        * @param theComponentsCount number of components
        */
        LightVectorND_CUDA(TElementType* theDevComponents, size_t theComponentsCount, GpuManagement gpuDevice)
        : components(theDevComponents)
        , componentsCount(theComponentsCount)
        , gpuManagement(gpuDevice)
        {}

        /** Create empty vector
        */
        LightVectorND_CUDA()
        : components(nullptr)
        , componentsCount(0)
        , gpuManagement(GpuManagement::defaultGPUDevice())
        {}

        /** Debug print with debug representation of the vector
        * @param out output steam for which operator << (const char* str) and  operator << (const T&) is defined.
        * @param variableName used variable name
        * @param delimiter used delimiter during printing values
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out, const char* variableName  = "x=", const char* delimiter = ",") const
        {
            out << variableName << "[";
            
            for (size_t i = 0; i < componentsCount; ++i)
            {
                if (i != 0)
                    out << delimiter;
                out << get(i);
            }
            out << "];\n";
        }

        ~LightVectorND_CUDA()
        {
            components = nullptr;
            componentsCount = 0;
        }

        /** Dump vector (which is almost in all applied math is column matrix with one column) by rows
        * @param outMemory pointer to memory in which content of the vector will be dumped
        * @param startRow first row. The first row is included into output.
        * @param endRow last row. The end row is included into output.
        */
        void dumpByRows(TElementType* outMemory, size_t startRow, size_t endRow) const
        {
            for (size_t i = startRow; i <= endRow; ++i)
                *(outMemory++) = get(i);
        }

        /** Dump all light vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        */
        void dumpByRows(TElementType* outMemory) const
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                *(outMemory++) = get(i);
        }

        /** Create string representation of the content of the vector
        * @param delimiter symbol used as delimiter for vector elements
        * @return formed string
        */
        ::std::string toString(const char* delimiter = ",") const
        {
            ::std::stringstream s;
            s << "[";

            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
            {
                if (i != 0)
                    s << delimiter;
                s << get(i);
            }
            s << "]";
            return s.str();
        }

        /** Effective swap content with other vector
        * @param other other vector
        */
        void swap(LightVectorND_CUDA& other) noexcept {
            if (this != &other)
            {
                dopt::CopyHelpers::swapDifferentObjects(&components, &other.components);
                dopt::CopyHelpers::swapDifferentObjects(&componentsCount, &other.componentsCount);
                dopt::CopyHelpers::swapDifferentObjects(&gpuManagement, &other.gpuManagement);
            }
        }

        /** Size of the vector in terms of number of items in it
        * @return size of the vector
        */
        size_t size() const {
            return componentsCount;
        }

        /** Memory reserved for this vector
        * @return real number of elements reserved for the vector content
        */
        size_t capacity() const {
            return componentsCount;
        }

        /** Evaluate sum of all elements
        * @return result sum of all elements.
        * @remark Implementation contains naive way. 
        * @reamrk More robust way in terms of numeric is firstly sort array, or use numerical stable Kahan summation algorithm.
        */
        template <class TAccumulator = typename Vec::TElementType>
        TAccumulator sum() const
        {
            TElementType res = TElementType();
            TElementType* devPtr = (TElementType*)gpuManagement.allocateBytesInDevice(sizeof(TElementType));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateSum(devPtr, components, componentsCount, gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            TAccumulator resAccum = TAccumulator(res);
            return resAccum;
        }
#if 0
        /** Evaluate sum of all elements and evaluated then natural logarithm for sum
        * @return result
        */
        TElementType logSum() const
        {
            TElementType res = TElementType();
            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
                res += ::log(components[i]);
            return res;
        }
#endif

        /** Get const reference item by index i
        * @param i index of item
        * @return reference to item offset by "startIndex + i"
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        */
        TElementType get(size_t i) const  {
#if DOPT_DEBUG_BUILD
            if (i >= componentsCount || i < 0)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
#endif
            TElementType result = TElementType();
            gpuManagement.copyDevice2HostSync(&result, components + i, 1);
            return result;
        }

        /** Create hard copy of vector just create another complete new vector from current view
        * @param result destination vector
        * @return number of elements in a new vector
        */
        template <class VecOther>
        size_t hardcopy(VecOther& result) const
        {
            result = Vec(componentsCount, gpuManagement);
            gpuManagement.copyDevice2DeviceSync(result.rawDevData(), components, componentsCount);
            return componentsCount;
        }

        /** Get non-const reference item by index i
        * @param i index of item
        * @return reference to item offset by "startIndex + i"
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        */
        TElementType& getRaw(size_t i) {
#if DOPT_DEBUG_BUILD
            if (i >= componentsCount || i < 0)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
#endif

            TElementType& result = components[i];
            return result;
        }
        
        /** Get sub vector of vector (*this)
        * @param theStartIndex startIndex of slice
        * @param theCount count number of items in sliced vector
        * @return constructed vector        
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        */
        LightVectorND_CUDA get(size_t theStartIndex, size_t theCount) const
        {
#if DOPT_DEBUG_BUILD
            if (theStartIndex + theCount > size())
            {
                assert(!"Not enough items to get subvector");
                return LightVectorND_CUDA();
            }
#endif
            LightVectorND_CUDA result;
            result.components = components + theStartIndex;
            result.componentsCount = theCount;
            result.gpuManagement = gpuManagement;
            return result;
        }

        /** Set it-the item to value
        * @param i index of setup
        * @param value for which item should be setup
        * @return reference to itself        
        * @remark For DEBUG build in case if index is out of range assertion is raised and nothing happens
        */
        LightVectorND_CUDA& set(size_t i, TElementType value) {
#if DOPT_DEBUG_BUILD
            if (i >= componentsCount || i < 0)
            {
                assert(!"Not correct index for set value!");
                return *this;
            }
#endif
            gpuManagement.copyHost2DeviceSync(components + i, &value, 1);
            return *this;
        }

        /** Set all items of vector to specific value
        * @param value for which all items should be setted
        * @return reference to itself
        */
        LightVectorND_CUDA& setAll(TElementType value) {
            applyKernelToSetAllItemsToValue(components, componentsCount, value, gpuManagement);
            return *this;
        }

        /** Set all items of vector to default value
        * @return reference to itself
        */
        LightVectorND_CUDA& setAllToDefault() {
            return setAll(TElementType());
        }

        /** Set all randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        template<class Generator>
        LightVectorND_CUDA& setAllRandomly(Generator& generator)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                set(i, generator.generateReal());
            return *this;
        }

        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        LightVectorND_CUDA& zeroOutItems(TElementType eps)
        {
            applyKernelToZeroOutItems(components, componentsCount, eps, gpuManagement);
            return *this;
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return reference to item
        * @remark For DEBUG build in case if index is out of range assertion is raised and reference to dummy variable is provided
        */
        TElementType operator [] (size_t index) const
        {
            TElementType res = TElementType();
            gpuManagement.copyDevice2HostSync(&res, components + index, 1);
            return res;
        }

        /** Method provided by API for generality. Idea to compress data storage if possible.
        * @return 0 when compression have been finished
        */
        size_t compress() {
            return 0;
        }

        /** Check that vector is null vector. Null vector contains all default items
        * @return true if all items have been initialized to default value
        */
        bool isNull() const {
            return nnz() == 0;
        }

        /** Check that vector item i is null.
        * @param i index of item to check
        * @return true if items i have been initialized to default value
        */
        bool isNull(size_t i) const
        {
            if (i >= size()) [[unlikely]]
                return true;

            TElementType nullItem = TElementType();
            return get(i) == nullItem;
        }

        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        uint64_t nnz() const {
            TElementType res = TElementType();
            TElementType* devPtr = (TElementType*)gpuManagement.allocateBytesInDevice(sizeof(TElementType));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateNnz(devPtr, components, componentsCount, gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return (uint64_t)res;
        }

        /** Return vector with indices of item which possibly are non-zero
        * @return vector of indices
        */
        ::std::vector<size_t> maybeNoEmptyIndicies() const
        {
            ::std::vector<size_t> res;
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                if (!isNull(i))
                    res.push_back(i);
            }
            return res;
        }

        /** Clean vector via initialize all it's items to default value
        * @return reference to itself
        */
        LightVectorND_CUDA& clean()
        {
            setAllToDefault();
            return *this;
        }

        /* Unary operator+
        * @return copy of object
        */
        LightVectorND_CUDA operator + () const
        {
            return *this;
        }

        /* Append element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        LightVectorND_CUDA& operator += (const LightVectorND_CUDA& v) {
            assert(size() == v.size());
            applyKernelToEvaluateAppendItems(components, v.components, componentsCount, gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        void addInPlaceVectorWithMultiple(TElementType multiple, const LightVectorND_CUDA& other) 
        {
            assert(size() == other.size());
            applyKernelToEvaluateAppendItemsWithMultiplier(components, other.components, componentsCount, multiple, gpuManagement);
            return;
        }

        void subInPlaceVectorWithMultiple(TElementType multiple, const LightVectorND_CUDA& other) {
            assert(size() == other.size());
            applyKernelToEvaluateAppendItemsWithMultiplier(components, other.components, componentsCount, -multiple, gpuManagement);
            return;
        }

        /* Append element wise other items of vector v thread safely
        * @param v other vector
        * @return reference to itself
        * @remark non-blocking implementation
        */
        LightVectorND_CUDA& addAnotherVectrorNonBlockingMulthithread(const LightVectorND_CUDA& v)
        {
            assert(size() == v.size());
            applyKernelToEvaluateAppendItems(components, v.components, componentsCount, gpuManagement, OperationFlags::eMakeOperationAtomic);
            return *this;
        }

        /* Append element wise other items of vector v thread safely
        * @param v other vector
        * @return reference to itself
        * @remark non-blocking implementation
        * @tparam highContentionIsPossible - please specify this into the true, if you know that a lot of threads are going to make update on the same memory
        */
        template<bool highContentionIsPossible = true>
        LightVectorND_CUDA& multiplyByScalarNonBlockingMulthithread(const TElementType multiplier)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, 
                                                      componentsCount, 
                                                      TElementType(multiplier), 
                                                      gpuManagement, 
                                                      OperationFlags::eMakeOperationAtomic);
            return *this;
        }

        /* Remove element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        LightVectorND_CUDA& operator -= (const LightVectorND_CUDA& v) {
            assert(size() == v.size());
            applyKernelToEvaluateSubItems(components, v.components, componentsCount, gpuManagement);
            return *this;
        }

        /* Multiply element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        LightVectorND_CUDA& operator *= (const LightVectorND_CUDA& v) {
            assert(size() == v.size());
            applyKernelToEvaluateMultiplyItems(components, v.components, componentsCount, gpuManagement);
            return *this;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        LightVectorND_CUDA& operator *= (TFactorType factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Multiply element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        LightVectorND_CUDA& operator *= (double factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Divide element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        LightVectorND_CUDA& operator /= (TFactorType factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(1.0 / factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Divide element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        LightVectorND_CUDA& operator /= (double factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(1.0 / factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Evaluate dot product on two vectors
        * @param rhs other vector
        * @return result dot product
        */
        TElementType operator & (const LightVectorND_CUDA& rhs) const
        {
            assert(size() == rhs.size());
            return reducedDotProduct(*this, rhs, 0, size());
        }

        /* Evaluate dot product on two vectors for which there is a garantee of propery memory alignment of the start
        * @param rhs other vector
        * @return result dot product
        */
        TElementType dotProductForAlignedMemory(const LightVectorND_CUDA& rhs) const
        {
            assert(size() == rhs.size());
            return reducedDotProduct(*this, rhs, 0, size());
        }

        /* Evaluate dot product on slice of two vectors which correspond to indices [start:end)
        * @param a first vector
        * @param b second vector
        * @param start start index for making slice of vector "a" and vector "b"
        * @param end end index for making slice of vector "a" and vector "b"
        * @return result dot product
        */
        static TElementType reducedDotProduct(const LightVectorND_CUDA& a, const LightVectorND_CUDA& b, size_t start, size_t end)
        {
            if (start == end)
                return TElementType();

            assert(start < a.size());
            assert(start < b.size());
            assert(end <= a.size());
            assert(end <= b.size());
            assert(start <= end);
            assert(start <= end);

            TElementType res = TElementType();
            GpuManagement& gpuManagement = const_cast<GpuManagement&>(a.gpuManagement);
            TElementType* devPtr = (TElementType*)gpuManagement.allocateBytesInDevice(sizeof(TElementType));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToReducedDotProduct(devPtr,
                                           a.components + start,
                                           b.components + start,
                                           end - start,
                                           gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /* Evaluate L2 norm square of vector
        * @return result
        */
        TElementType vectorL2NormSquare() const
        {
            return vectorLPNormPowerP(2);
        }

        /* Evaluate L2 norm square of vector
        * @return result
        */
        TElementType vectorL2NormSquareForAlignedMemory() const
        {
            return vectorLPNormPowerP(2);
        }

        /* Evaluate L2 norm of vector
        * @return result
        */
        TElementType vectorL2Norm() const {
            return ::sqrt(vectorL2NormSquare());
        }

        /*Evaluate LP norm to the power of P
        * @return result
        */
        TElementType vectorLPNormPowerP(uint32_t p) const
        {
            TElementType res = TElementType();
            TElementType* devPtr = (TElementType*)gpuManagement.allocateBytesInDevice(sizeof(TElementType));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateLpNormToPowerP(devPtr, components, componentsCount, gpuManagement, p);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /* Evaluate LP norm to the power of P
        * @return result
        */
        TElementType vectorLPNorm(uint32_t p) const 
        {
            size_t dim = size();
            TElementType res = dopt::powerReal(vectorLPNormPowerP(p), 1.0 / double(p));

            return res;
        }

        /* Evaluate L1 norm of vector
        * @return result
        */
        TElementType vectorL1Norm() const {
            return vectorLPNormPowerP(1);
        }

        /* Evaluate Linf norm of vector
        * @return result
        */
        TElementType vectorLinfNorm() const 
        {
            TElementType res = TElementType();
            TElementType* devPtr = (TElementType*)gpuManagement.allocateBytesInDevice(sizeof(TElementType));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateLInfNorm(devPtr, components, componentsCount, gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /* Check if two vectors are equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator == (const LightVectorND_CUDA& v) const
        {
            if (size() != v.size())
                return false;

            Vec tmp;
            hardcopy(tmp);
            
            LightVectorND_CUDA tmpView(tmp, 0);
            tmpView -= v;

            return tmpView.isNull();
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const LightVectorND_CUDA& v) const 
        {
            return !(*this == v);
        }

        /* Obtain raw pointer to underlying data
        * @return raw pointer
        */
        TElementType* rawDevData() {
            if (componentsCount == 0)
                return nullptr;
            else
                return components;
        }

        /* Obtain raw const pointer to underlying data
        * @return raw pointer
        */
        const TElementType* rawDevData() const {
            if (componentsCount == 0)
                return nullptr;
            else
                return components;
        }

        /* Check if two vectors are equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator == (const Vec& v) const
        {
            LightVectorND_CUDA v_view(const_cast<Vec&>(v), 0);
            return (*this) == (v_view);
        }

        /** Concatenate two vectors a and b => [a, b]
        * @param a first vector
        * @param b second vector
        * @return result vector
        */
        static Vec concat(const LightVectorND_CUDA& a, const LightVectorND_CUDA& b)
        {
            GpuManagement& gpuManagement = const_cast<GpuManagement&>(a.gpuManagement);

            size_t asize = a.size();
            size_t bsize = b.size();

            Vec res(asize + bsize);
            gpuManagement.copyDevice2DeviceSync(res.rawDevData(), a.components, asize);
            gpuManagement.copyDevice2DeviceSync(res.rawDevData() + asize, b.components, bsize);
            return res;
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const Vec& v) const {
            return !(*this == v);
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return reference to *this
        */
        LightVectorND_CUDA& clamp(const LightVectorND_CUDA& lower, const LightVectorND_CUDA& upper)
        {
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                const TElementType& ai = get(i);

                if ( ai <= lower[i] )
                    set(i, lower[i]);
                else if ( ai >= upper[i] )
                    set(i, upper[i]);
            }
            return *this;
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return reference to *this
        */
        LightVectorND_CUDA& clamp(const TElementType lower, const TElementType upper)
        {
            applyKernelToClampItems(components, componentsCount, lower, upper, gpuManagement);
            return *this;
        }

        /** For current vector view add vector "v" with multiple "multiplier"
        * @param v vector to be added
        * @param multiplier the multiplication factor near vector "v" which should be added
        * @return reference to *this
        */
        LightVectorND_CUDA& addWithVectorMultiple(const LightVectorND_CUDA& v, const TElementType multiplier)
        {
            assert(size() == v.size());
            applyKernelToEvaluateAppendItemsWithMultiplier(components, 
                                                           v.components, componentsCount, 
                                                           multiplier, gpuManagement);
            return *this;
        }
        
        /** For current vector view assign vector "v" with multiple "multiplier"
        * @param v vector to be added
        * @param multiplier the multiplication factor near vector "v" which should be added
        * @return reference to *this
        * TODO
        */
        LightVectorND_CUDA& assignWithVectorMultiple(const LightVectorND_CUDA& v, TElementType multiplier)
        {
            assert(size() == v.size());
            applyKernelToEvaluateAssignItemsWithMultiplier(components,
                                                           v.components, componentsCount,
                                                           multiplier, gpuManagement);
            return *this;
        }

        /** For current vector view assign value of vector "a" - "b" 
        * @param a first vector
        * @param b second vector
        * @return reference to *this
        */
        LightVectorND_CUDA& assignWithVectorDifference(const LightVectorND_CUDA& a, const LightVectorND_CUDA& b)
        {
            return assignWithVectorDifferenceAligned(a, b);
        }

        /** For current vector view assign value of vector "a" - "b"
        * @param a first vector
        * @param b second vector
        * @return reference to *this
        */
        LightVectorND_CUDA& assignWithVectorDifferenceAligned(const LightVectorND_CUDA& a, const LightVectorND_CUDA& b)
        {
            assert(size() == a.size());
            assert(size() == b.size());
            gpuManagement.copyDevice2DeviceSync(components, a.components, componentsCount);
            *this -= b;
            
            return *this;
        }

        /* Copy values from one light view array "r" to current.
        * @param r other light vector
        * @return reference to itself
        */
        LightVectorND_CUDA& assignAllValues(const LightVectorND_CUDA& r)
        {
            assert(size() == r.size());
            assert(this != &r);
            gpuManagement.copyDevice2DeviceSync(components, r.components, componentsCount);
            return *this;
        }

        /** Create vector in form [start, start + 1, start + 2,...]
        * @param dimension dimension of the vector
        * @tparam start initial value for sequence
        * @return result vector
        */
        LightVectorND_CUDA& assignIncreasingSequence(TElementType initialValue)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                set(i, initialValue + i);
            }

            return *this;
        }
    };
}
