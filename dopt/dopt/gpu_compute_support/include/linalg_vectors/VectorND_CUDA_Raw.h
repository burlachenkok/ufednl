/** @file
* C++ cross-platform implementation of mathematical vector, elements of which are stored in dynamically allocated memory
*/

#pragma once

//#include "dopt/copylocal/include/Copier.h"
//#include "dopt/math_routines/include/SimpleMathRoutines.h"
//#include "dopt/system/include/MemoryPool.h"

#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/VectorND_CUDA_Raw_kernels.h"
#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"

#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include <vector>
#include <initializer_list>
#include <type_traits>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <map>

#include <cstdlib>

#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include <iostream>

namespace dopt
{
    template <typename TVec>
    class LightVectorND_CUDA;

    /** Allocated memory enough for store specific number of items
    * @param nBytes number of bytes to allocate
    * @return pointer to allocated memory
    */
    forceinline_ext void* memAllocationCUDA(size_t nBytes, dopt::GpuManagement& gpuManagement)
    {
        if (nBytes == 0)
            return nullptr;

        void* rawDevPointer = gpuManagement.allocateBytesInDevice(nBytes);       
        return rawDevPointer;
    }

    /** Free memory previously allocated via memAllocation
    * @param rawPointer pointer previously allocated with memAllocation
    * @param nBytes number of bytes to allocate memory
    */
    forceinline_ext void memFreeCUDA(void* rawPointer, size_t nBytes, dopt::GpuManagement& gpuManagement)
    {
        if (nBytes == 0 || rawPointer == nullptr)
            return;
                
        gpuManagement.freeMemoryInDevice(rawPointer);
    }
    
    typedef void* (*TMemoryAllocationCUDACallback)(size_t nBytes, dopt::GpuManagement& gpuManagement);
    typedef void (*TMemoryFreeCUDACallback)(void* rawPointer, size_t nBytes, dopt::GpuManagement& gpuManagement);

    /** Vector with arbitrarily dimension
    * @tparam type of elements inside the vector
    * @tparam memAllocation callback used for allocate memory
    * @tparam memFree callback used for free memory
    * @sa VectorNDStd
    */
    template <typename T>
    class VectorND_CUDA_Raw
    {
    private:
        const static TMemoryAllocationCUDACallback memAllocation;
        const static TMemoryFreeCUDACallback memFree;

    public:
        using TElementType = T;                   ///< Typedef for elements types
        static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");

    private:
        /** Helper function to declare number of bytes which is needed to store specified number of components in flat array
        * @param components number of components in the array
        * @return number of bytes in flat array
        */
        static size_t components2bytes(size_t components) {
            return components * sizeof(T);
        }

        size_t componentsCount;                                  ///< Components count
        T* components;                                           ///< Components of the vector
        mutable GpuManagement gpuManagement;                     ///< Device in which data is allocated
        
    public:
        /** Debug print with debug representation of the vector
        * @param out output steam for which operator << (const char* str) and  operator << (const T&) is defined.
        * @param variableName used variable name
        * @param delimiter used delimiter during printing values
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out, const char* variableName = "x=", const char* delimiter = ",") const
        {
            out << variableName << "[";

            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
            {
                if (i != 0)
                    out << delimiter;
                out << get(i);
            }
            out << "];\n";
        }

        /** Create empty vector
        */
        VectorND_CUDA_Raw() noexcept
        : components(0)
        , componentsCount(0)
        , gpuManagement(dopt::GpuManagement::defaultGPUDevice())
        {}

        /** Copy ctor
        */
        VectorND_CUDA_Raw(const VectorND_CUDA_Raw& rhs) noexcept
        : VectorND_CUDA_Raw(eAllocNotInit, rhs.componentsCount, rhs.gpuManagement)
        {
            gpuManagement.copyDevice2DeviceSync(components, rhs.components, rhs.size());
        }

        /** Create vector with specified dimension
        * @param dimension specified dimension
        * @remark all component are initialization with default ctor or zero.
        */
        VectorND_CUDA_Raw(size_t dimension) noexcept
        : VectorND_CUDA_Raw(eAllocAndSetToZero, dimension, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()))
        {
        }

        /** Create vector with specified dimension
       * @param dimension specified dimension
       * @remark all component are initialization with default ctor or zero.
       */
        VectorND_CUDA_Raw(size_t dimension, GpuManagement device) noexcept
        : VectorND_CUDA_Raw(eAllocAndSetToZero, dimension, device)
        {
        }

        /** Destructor.
        */
        ~VectorND_CUDA_Raw() noexcept
        {
            if (!components)
                return;

            size_t sz = size();
            gpuManagement.freeMemoryInDevice(components);

            components = nullptr;
            componentsCount = 0;
        }

        /** Assignment move operator
        * @param rhs xvalue expression from which we perform move
        */
        VectorND_CUDA_Raw& operator = (VectorND_CUDA_Raw&& rhs) noexcept
        {
            this->~VectorND_CUDA_Raw();
            
            componentsCount = rhs.componentsCount;
            components = rhs.components;
            gpuManagement = rhs.gpuManagement;
            
            rhs.components = nullptr;
            rhs.componentsCount = 0;
            
            return *this;
        }

        /** Copy move operator
        */
        VectorND_CUDA_Raw(VectorND_CUDA_Raw&& rhs) noexcept
        {
            componentsCount = rhs.componentsCount;
            components = rhs.components;
            gpuManagement = rhs.gpuManagement;
            
            rhs.components = nullptr;
            rhs.componentsCount = 0;
        }

        /** Usual assignment operator
        */
        VectorND_CUDA_Raw& operator = (const VectorND_CUDA_Raw& rhs) noexcept
        {
            if (this == &rhs)
                return *this;

            if (componentsCount != rhs.componentsCount)
            {
                this->~VectorND_CUDA_Raw();

                componentsCount = rhs.componentsCount;

                if (componentsCount > 0)
                {
                    components = static_cast<T*>( memAllocation(components2bytes(componentsCount), gpuManagement) );
                }
            }
            
            gpuManagement.copyDevice2DeviceSync(components, rhs.components, rhs.size());
            return *this;
        }

        /** Create vector in form [start, start + 1, start + 2,...]
        * @param dimension dimension of the vector
        * @tparam start initial value for sequence
        * @return result vector
        */
        template<int start>
        static VectorND_CUDA_Raw sequence(size_t dimension)
        {
            VectorND_CUDA_Raw res(eAllocNotInit,
                                  dimension, 
                                  dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()));
            T value = (T)start;

            for (size_t i = 0; i < dimension; ++i, value += T(1))
            {
                res.set(i, start + i);
            }
            return res;
        }

        /** Create vector in form [1,1,1,...,1]
        * @param dimension dimension of the vector
        */
        static VectorND_CUDA_Raw eye(size_t dimension)
        {
            VectorND_CUDA_Raw res(eAllocNotInit, dimension, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()));
            res.setAll(T(1));
            return res;
        }

        /** Get uninitialized vector
        * @param dimension dimension of the vector
        * @remark Please call it only if you understand what you're doing.
        */
        static VectorND_CUDA_Raw getUninitializedVector(size_t dimension)
        {
            VectorND_CUDA_Raw res(eAllocNotInit, dimension, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()));
            return res;
        }

        /** Create vector in form [value,value,value,...,value]
        * @param dimension dimension of the vector
        * @param value used value to initialize vector
        */
        static VectorND_CUDA_Raw eye(size_t dimension, T value)
        {
            VectorND_CUDA_Raw res(eAllocNotInit, dimension, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()));
            res.setAll(value);
            return res;
        }
        
        /** Create vector via using specified values from initialize list
        * @param objects init. list in form {v1, v2,...vn}
        */
        VectorND_CUDA_Raw(std::initializer_list<T> objects)
        : VectorND_CUDA_Raw(eAllocNotInit, objects.size(), dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()))
        {
            size_t sz = objects.size();
            const T* item = objects.begin();
            gpuManagement.copyHost2DeviceSync(components, item, sz);
        }

        /** Create vector via using specified values from initialize list
        * @param objects init. list in form {v1, v2,...vn}
        * @remark it's due to lack of poor support of std::initializer list for temporary objects in Microsoft Visual Studio compiler
        */
        static VectorND_CUDA_Raw init(std::initializer_list<T> objects)
        {
            VectorND_CUDA_Raw res = objects;
            return res;
        }

        /** Create vector via using specified values from raw array
        * @param srcItems start of the array used to initialize vector
        * @param srcNumberOfItems number of items used from srcItems array
        */
        VectorND_CUDA_Raw(const T* srcItems, size_t srcNumberOfItems)
        : VectorND_CUDA_Raw(eAllocNotInit, srcNumberOfItems, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()))
        {
            gpuManagement.copyHost2DeviceSync(components, srcItems, srcNumberOfItems);           
        }

        /** Dump vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        * @param startRow first row
        * @param endRow last row
        */
        void dumpByRows(T* out, size_t startRow, size_t endRow) const
        {
            for (size_t i = startRow; i <= endRow; ++i)
                *(out++) = get(i);
        }

        /** Dump all vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        */
        void dumpByRows(T* out) const
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                *(out++) = get(i);
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
        * @param vecA first vector
        * @param vecB second vector
        * @remark vecA and vecB are different objects in memory
        */
        template <bool swappedVectorsHasTheSameDimension = false>
        static void swapDifferentVectors(VectorND_CUDA_Raw& vecA, VectorND_CUDA_Raw& vecB) noexcept 
        {
            {
                auto tmp = vecA.components;
                vecA.components = vecB.components;
                vecB.components = tmp;
            }
            {
                auto tmp = vecA.gpuManagement;
                vecA.gpuManagement = vecB.gpuManagement;
                vecB.gpuManagement = tmp;
            }

            if constexpr (swappedVectorsHasTheSameDimension)
            {
                // Do nothing
            }
            else
            {
                auto tmp = vecA.componentsCount;
                vecA.componentsCount = vecB.componentsCount;
                vecB.componentsCount = tmp;
            }
        }

        /** Size of the vector in terms of number of items in it
        * @return size of the vector
        */
        size_t size() const {
            return componentsCount;
        }

        /** Size of the vector in terms of bytes occupied by it
        * @return size of the vector in bytes
        */
        size_t sizeInBytes() const {
            return components2bytes(componentsCount);
        }

        /** Memory reserved for this vector
        * @return real number of elements reserved for the vector content
        */
        size_t capacity() const {
            return componentsCount;
        }

        /** Resize vector for new size
        * @param newSize new size of the vector
        * @return reference to itself
        * @remark this call can invalidate previously obtained pointers via data() and dataConst()
        */
        VectorND_CUDA_Raw& resize(size_t newSize)
        {
            T* mewComponents = static_cast<T*>( memAllocationCUDA(components2bytes(newSize), gpuManagement) );
            gpuManagement.setDeviceMemoryToZero(mewComponents, newSize);
            size_t newComponentsCount = newSize;

            if (!mewComponents) [[unlikely]]
            {
                assert(!"Allocation problems in resize() ");
                return *this;
            }
                
            gpuManagement.copyDevice2DeviceSync(mewComponents, components, componentsCount);

            if (components) [[likely]]
            {
                size_t sz = componentsCount;
                gpuManagement.freeMemoryInDevice(components);
                components = nullptr;
            }
                
            components = mewComponents;
            componentsCount = newComponentsCount;
            return *this;
        }

        /** Get maximum element in the vector
        * @return maximum element
        */
        T maxItem() const
        {
            T res = std::numeric_limits<T>::min();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateMax(devPtr, components, componentsCount, gpuManagement);
            
            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /** Get minimum element value in the vector
        * @return minimum element value in the vector
        */
        T minItem() const
        {
            T res = std::numeric_limits<T>::max();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);
            
            applyKernelToEvaluateMin(devPtr, components, componentsCount, gpuManagement);
            
            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /** Current vector of elements is interpretated as a dense vector with margins for each example
        * @return return unweighted logistic loss value for margins
        * @remark Margin is quantity used in Machine Learning are relative areas to denote dot product of features by datapoints value times sign of example.
        */
        template <class TAccumulator = T>
        static TAccumulator logisticUnweightedLossFromMargin(const VectorND_CUDA_Raw& margin)
        {
            T res = T();
            T* devPtr = (T*)(margin.gpuManagement.allocateBytesInDevice(sizeof(T)));
            margin.gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateLogisticLossFromMarginSum(devPtr, margin.components, margin.componentsCount, margin.gpuManagement);

            margin.gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            margin.gpuManagement.freeMemoryInDevice(devPtr);

            res *= T(1.0) / T(margin.componentsCount);
            return TAccumulator(res);
        }

        template <class TAccumulator = T>
        static TAccumulator logisticUnweightedLossFromMarginSigmoid(const VectorND_CUDA_Raw& classificationMarginSigmoid)
        {
            GpuManagement& gpuManagement = classificationMarginSigmoid.gpuManagement;
            
            T res = T();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(devPtr, 
                                                                  classificationMarginSigmoid.components, 
                                                                  classificationMarginSigmoid.componentsCount, gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            res *= T(1.0) / T(classificationMarginSigmoid.componentsCount);
            return TAccumulator(res);
        } 
        
        /** Evaluate sum of all elements
        * @return result sum
        * @remark Implementation contains naive way. More robust way in terms of numerics is firstly sort array.
        */
        template <class TAccumulator = T>
        TAccumulator sum() const
        {
            T res = T();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);
            
            applyKernelToEvaluateSum(devPtr, components, componentsCount, gpuManagement);
            
            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            TAccumulator resAccum = TAccumulator(res);
            return resAccum;
        }
        
        /** Create vector which element wise contains absolute values of current (*this) vector
        * @return vector with element wise positive values
        */
        VectorND_CUDA_Raw abs() const
        {
            VectorND_CUDA_Raw res(size());
            if (res.size() > 0)
                applyKernelToEvaluateAbsItems(res.components, components, size(), gpuManagement);
            return res;
        }

        /** Get new vector which store sign's of vec input
        * @param input vector with positive/negative elements
        * @param posSignValue value which is set for positive items
        * @param negSignValue value which is set for negative items
        * @return result vector which element wise contain -1/+1 depend on sign
        */
        static VectorND_CUDA_Raw sign(const VectorND_CUDA_Raw& vec, const T posSignValue = T(+1), const T negSignValue = T(-1))
        {
            VectorND_CUDA_Raw res(vec.size());
            if (res.size() > 0)
                applyKernelToEvaluateSignOfItems(res.components, vec.components, vec.componentsCount, posSignValue, negSignValue, res.gpuManagement);
            return res;
        }
        
        /** Get non-const reference item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        * @remark This a bit dangerous method. Use it only if you know what you are doing. It is a pointer to GPU memory.
        */
        T& getRaw(size_t i) {
            T& result = components[i];
            return result;
        }

        /** Get const reference item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        * @remark This a bit dangerous method. Use it only if you know what you are doing. It is a pointer to GPU memory.
        */
        const T& getRaw(size_t i) const {
            T& result = components[i];
            return result;
        }
        
        /** Get copy of item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        */
        T get(size_t i) const 
        {
            T result = T();
            gpuManagement.copyDevice2HostSync(& result, components + i, 1);            
            return result;
        }
        
        /** Get subvector of vector (*this)
        * @param theStartIndex startIndex of slice
        * @param theCount count number of items in sliced vector
        * @return constructed vector
        */
        VectorND_CUDA_Raw get(size_t theStartIndex, size_t theCount) const
        {
            if (theStartIndex + theCount > size())
            {
                assert(!"Not enough items to get subvector");
                return VectorND_CUDA_Raw();
            }
            
            VectorND_CUDA_Raw result(theCount);
            gpuManagement.copyDevice2DeviceSync(result.components, components + theStartIndex, theCount);
            return result;
        }


        /** Load values into GPU vector
        * @param valuesCopyFrom values to copy into vector stored in GPU memory
        * @param itemsToCopy number of items to copy
        * @return reference vector
        */
        VectorND_CUDA_Raw& load(const T* valuesCopyFrom, size_t itemsToCopy)
        {
            assert(componentsCount >= itemsToCopy);
            gpuManagement.copyHost2DeviceSync(components, valuesCopyFrom, itemsToCopy);
            return *this;
        }

        /** Store(or dump) values from current vector into CPU/Virtual memory
        * @param valuesCopyTo storage in which values from vector will be copied into
        * @param itemsToCopy number of items to copy
        * @return this vector
        */
        VectorND_CUDA_Raw& store(T* valuesCopyTo, size_t itemsToCopy)
        {
            assert(componentsCount >= itemsToCopy);
            gpuManagement.copyDevice2HostSync(valuesCopyTo, components, itemsToCopy);
            return *this;
        }

        /** Load values into GPU vector
        * @param valuesCopy2vec values to copy into vector stored in GPU memory
        * @return reference vector
        */
        VectorND_CUDA_Raw& load(const VectorNDRaw<T>& valuesCopyFrom)
        {
            assert(componentsCount == valuesCopyFrom.size());
            gpuManagement.copyHost2DeviceSync(components, valuesCopyFrom.dataConst(), componentsCount);
            return *this;
        }

        /** Store(or dump) values from current vector into CPU/Virtual memory
        * @param valuesCopyTo storage in which values from vector will be copied into
        * @return this vector
        */
        VectorND_CUDA_Raw& store(VectorNDRaw<T>& valuesCopyTo)
        {
            if (componentsCount != valuesCopyTo.size())
                valuesCopyTo = VectorNDRaw<T>(componentsCount);

            gpuManagement.copyDevice2HostSync(valuesCopyTo.data(), components, componentsCount);
            return *this;
        }

        /** Load values into GPU vector
        * @param valuesCopy2vec values to copy into vector stored in GPU memory
        * @return reference vector
        */        
        VectorND_CUDA_Raw& load(const VectorNDStd<T>& valuesCopyFrom)
        {
            assert(componentsCount == valuesCopyFrom.size());
            gpuManagement.copyHost2DeviceSync(components, valuesCopyFrom.dataConst(), componentsCount);
            return *this;
        }

        /** Store(or dump) values from current vector into CPU/Virtual memory
        * @param valuesCopyTo storage in which values from vector will be copied into
        * @return this vector
        */
        VectorND_CUDA_Raw& store(VectorNDStd<T>& valuesCopyTo)
        {
            if (componentsCount != valuesCopyTo.size())
                valuesCopyTo = VectorNDStdDRaw<T>(componentsCount);

            gpuManagement.copyDevice2HostSync(valuesCopyTo.data(), components, componentsCount);
            return *this;
        }

        /** Set it-the item to value
        * @param i index of setup
        * @param value for which item should be setuped
        * @return reference to itself
        */
        VectorND_CUDA_Raw& set(size_t i, T value) 
        {
            gpuManagement.copyHost2DeviceSync(components + i, &value, 1);
            return *this;
        }
        /** Set all items of vector to default value
        * @return reference to itself
        */
        VectorND_CUDA_Raw& setAllToDefault() {
            return setAll(T());
        }


        /** Set all items of vector to specific value
        * @param value for which all items should be setted
        * @return reference to itself
        */
        VectorND_CUDA_Raw& setAll(T value) 
        {
            applyKernelToSetAllItemsToValue(components, componentsCount, value, gpuManagement);
            return *this;
        }

        /** Set all randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        template<class Generator>
        VectorND_CUDA_Raw& setAllRandomly(Generator& generator)
        {
            dopt::VectorNDRaw<TElementType> hostVec(componentsCount);
            hostVec.setAllRandomly(generator);
            gpuManagement.copyHost2DeviceSync(components, hostVec.data(), componentsCount);
            return *this;
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return const-reference to item
        */
        const T operator [] (size_t index) const 
        {
            T res = T();
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

            T nullItem = T();
            return get(i) == nullItem;
        }

        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        uint64_t nnz() const
        {
            T res = T();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateNnz(devPtr, components, componentsCount, gpuManagement);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return (uint64_t)res;
        }
        
        /** Return vector of possibly non-empty indices
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
        VectorND_CUDA_Raw& clean()
        {
            setAllToDefault();
            return *this;
        }

        /* Unary operator +. Returns hard copy of vector.
        * @return copy of object
        */
        VectorND_CUDA_Raw operator + () const
        {
            return *this;
        }

        /* Unary operator -. Returns new vector where all elements has a flipped sign.
        * @return copy of object with all items reverse their sign
        */
        VectorND_CUDA_Raw operator - () const
        {
            VectorND_CUDA_Raw res(eAllocNotInit, componentsCount, dopt::GpuManagement(dopt::GpuManagement::defaultGPUDevice()));
            applyKernelToEvaluateNegItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /* Append element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorND_CUDA_Raw& operator += (const VectorND_CUDA_Raw& v) 
        {
            assert(size() == v.size());
            applyKernelToEvaluateAppendItems(components, v.components, componentsCount, gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /** Add to current vector [this] the muplipler of another vector [other] in a way that:
        *  [this] := this + (multiple) * other
        * @param multiple muplitple of vector to add
        * @param other another vector to add with specific multiplicative factor
        */
        void addInPlaceVectorWithMultiple(TElementType multiple, const VectorND_CUDA_Raw& other)
        {
            assert(size() == other.size());
            applyKernelToEvaluateAppendItemsWithMultiplier(components, other.components, componentsCount, multiple, gpuManagement);
            return;
        }

        void subInPlaceVectorWithMultiple(TElementType multiple, const VectorND_CUDA_Raw& other)
        {
            assert(size() == other.size());
            applyKernelToEvaluateAppendItemsWithMultiplier(components, other.components, componentsCount, -multiple, gpuManagement);
            return;
        }
        
        /* Remove element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorND_CUDA_Raw& operator -= (const VectorND_CUDA_Raw& v) {
            assert(size() == v.size());
            applyKernelToEvaluateSubItems(components, v.components, componentsCount, gpuManagement);
            return *this;
        }

        /* Multiply element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorND_CUDA_Raw& operator *= (const VectorND_CUDA_Raw& v) {
            assert(size() == v.size());
            applyKernelToEvaluateMultiplyItems(components, v.components, componentsCount, gpuManagement);
            return *this;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        VectorND_CUDA_Raw& operator *= (TFactorType factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }
        
        /* Multiply element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        VectorND_CUDA_Raw& operator *= (double factor)
        {
            applyKernelToEvaluateMutiplyItemsByFactor(components, componentsCount, TElementType(factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Divide element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        VectorND_CUDA_Raw& operator /= (TFactorType factor)
        {
            applyKernelToEvaluateDivItemsByFactor(components, componentsCount, factor, gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Divide element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        VectorND_CUDA_Raw& operator /= (double factor)
        {
            applyKernelToEvaluateDivItemsByFactor(components, componentsCount, TElementType(factor), gpuManagement, OperationFlags::eOperationNone);
            return *this;
        }

        /* Add two vectors element wise and return new vector
        * @param rhs other vector
        * @return vector with result
        */
        VectorND_CUDA_Raw<T> operator + (const VectorND_CUDA_Raw<T>& rhs) const
        {
            VectorND_CUDA_Raw<T> res(*this);
            res += rhs;
            return res;
        }

        /*  Evaluate (a-b) for two vectors
        * @param rhs other vector
        * @return copy of vector with result
        */
        VectorND_CUDA_Raw<T> operator - (const VectorND_CUDA_Raw<T>& rhs) const
        {
            VectorND_CUDA_Raw<T> res(*this);
            res -= rhs;
            return res;
        }

        /*  Evaluate (a[1,...,1] - rhs) * multipler
        * @param a scalar which is used to scale vector consisted elementwise of all "1"
        * @param rhs other vector
        * @param multiplier final multiplier
        */
        static VectorND_CUDA_Raw scaledDifferenceWithEye(TElementType a, const VectorND_CUDA_Raw& rhs, TElementType multiplier)
        {
            VectorND_CUDA_Raw res(rhs);
            applyKernelToScaledDifferenceWithEye(rhs.components, rhs.componentsCount, a, multiplier, rhs.gpuManagement);
            return res;
        }

        /* Divide element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        template <typename TFactorType>
        VectorND_CUDA_Raw<T> operator / (TFactorType factor) const {
            VectorND_CUDA_Raw<T> res(*this);
            res /= factor;
            return res;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        template <typename TFactorType>
        VectorND_CUDA_Raw<T> operator * (TFactorType factor) const {
            VectorND_CUDA_Raw<T> res(*this);
            res *= factor;
            return res;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        VectorND_CUDA_Raw<T> operator * (double factor) const {
            VectorND_CUDA_Raw<T> res(*this);
            res *= factor;
            return res;
        }

        /* Evaluate do product on two vectors
        * @param rhs other vector
        * @return result dot product
        */
        T operator & (const VectorND_CUDA_Raw& rhs) const {
            assert(size() == rhs.size());
            return reducedDotProduct(*this, rhs, 0, size());
        }

        /* Evaluate do product on slice of two vectors which correspond to indices [start:end)
        * @param a first vector
        * @param b second vector
        * @param start start index for making slice of vector "a" and vector "b"
        * @param end end index for making slice of vector "a" and vector "b"
        * @return result dot product
        */
        static T reducedDotProduct(const VectorND_CUDA_Raw& a, const VectorND_CUDA_Raw& b, size_t start, size_t end) {
            if (start == end)
                return T();

            assert(start < a.size());
            assert(start < b.size());
            assert(end <= a.size());
            assert(end <= b.size());
            assert(start <= end);
            assert(start <= end);

            T res = T();
            GpuManagement& gpuManagement = const_cast<GpuManagement&>(a.gpuManagement);
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
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
        T vectorL2NormSquare() const {
            return vectorLPNormPowerP(2);
        }

        /* Evaluate L2 norm of vector
        * @return result
        */
        T vectorL2Norm() const {
            return ::sqrt(vectorL2NormSquare());
        }

        /* Evaluate LP norm to the power of P
        * @return result
        */
        T vectorLPNormPowerP(uint32_t p) const 
        {
            T res = T();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
            gpuManagement.copyHost2DeviceSync(devPtr, &res, 1);

            applyKernelToEvaluateLpNormToPowerP(devPtr, components, componentsCount, gpuManagement, p);

            gpuManagement.copyDevice2HostSync(&res, devPtr, 1);
            gpuManagement.freeMemoryInDevice(devPtr);

            return res;
        }

        /* Evaluate LP norm to the power of P
        * @return result
        */
        T vectorLPNorm(uint32_t p) const {
            size_t dim = size();
            T res = dopt::powerReal(vectorLPNormPowerP(p), 1.0 / double(p));

            return res;
        }

        /* Evaluate L1 norm of vector
        * @return result
        */
        T vectorL1Norm() const {
            return vectorLPNormPowerP(1);
        }

        /* Evaluate Linf norm of vector
        * @return result
        */
        T vectorLinfNorm() const
        {
            T res = T();
            T* devPtr = (T*)gpuManagement.allocateBytesInDevice(sizeof(T));
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
        bool operator == (const VectorND_CUDA_Raw& v) const 
        {
            VectorND_CUDA_Raw res = *this - v;
            return res.isNull();
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const VectorND_CUDA_Raw& v) const {
            return !(*this == v);
        }

        /* Obtain raw pointer to underlying data
        * @return raw pointer
        */
        T* rawDevData() {
            if (componentsCount == 0) [[unlikely]]
                return nullptr;
            else
                return components;
        }

        /* Obtain raw const pointer to underlying data
        * @return raw pointer
        */
        const T* rawDevData() const {
            if (componentsCount == 0) [[unlikely]]
                return nullptr;
            else
                return components;
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return vector with results
        */
        VectorND_CUDA_Raw& clamp(const VectorND_CUDA_Raw& lower, const VectorND_CUDA_Raw& upper)
        {
            applyKernelToClampItemsVectorized(components, componentsCount, lower.components, upper.components, gpuManagement);
            return *this;
        }


        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return vector with results
        */
        VectorND_CUDA_Raw& clamp(const T lower, const T upper)
        {
            applyKernelToClampItems(components, componentsCount, lower, upper, gpuManagement);
            return *this;
        }


        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        VectorND_CUDA_Raw& zeroOutItems(T eps)
        {
            applyKernelToZeroOutItems(components, componentsCount, eps, gpuManagement);
            return *this;
        }
        
        /** Apply element wise function exp() to elements of the vector
        * @return vector with results
        */
        VectorND_CUDA_Raw exp() const
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateExpItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Apply element wise function log() which is natural logarithm to elements of the vector
        * @return vector with results
        */
        VectorND_CUDA_Raw log()
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateLogItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Apply element wise function inv(x)=1/x
        * @return vector with results
        */
        VectorND_CUDA_Raw inv()
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateInvItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Apply element wise square function
        * @return vector with results
        */
        VectorND_CUDA_Raw square()
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateSquareItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Apply element wise square root function
        * @return vector with results
        */
        VectorND_CUDA_Raw sqrt()
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateSqrtItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Apply element wise function inv(x)=1/(x*x)
        * @return vector with results
        */
        VectorND_CUDA_Raw invSquare()
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateInvSquareItems(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Concatenate two vectors a and b => [a, b]
        * @param a first vector
        * @param b second vector
        * @return result vector
        */
        static VectorND_CUDA_Raw concat(const VectorND_CUDA_Raw& a, const VectorND_CUDA_Raw& b)
        {
            GpuManagement& gpuManagement = const_cast<GpuManagement&>(a.gpuManagement);

            size_t asize = a.size();
            size_t bsize = b.size();

            VectorND_CUDA_Raw<T> res(asize + bsize);
            gpuManagement.copyDevice2DeviceSync(res.components, a.components, asize);
            gpuManagement.copyDevice2DeviceSync(res.components + asize, b.components, bsize);
            return res;
        }
        
        /** Interpretate vector as vector which consist of classification margins. I.e. element wise it contains -b_i <a_i, x>
        * @return value of (unweghted) logistic loss
        */
        T logisticLoss() const {
            return logisticUnweightedLossFromMargin<T>(*this);
        }

        /** Compute element wise sigmoid(x_i), where sigmoid(x)=1.0/(1.0 + e^(-x))
        * @return value of (unweghted) logistic loss
        */
        VectorND_CUDA_Raw elementwiseSigmoid() const
        {
            VectorND_CUDA_Raw res(size());
            applyKernelToEvaluateSigmoidFn(res.components, components, componentsCount, gpuManagement);
            return res;
        }

        /** Compute approximate version of element wise sigmoid(x_i), where sigmoid(x)=1.0/(1.0 + e^(-x))
        * @return value of (unweghted) logistic loss
        * @remark this method designed via Taylor expansio of exp(-x) near zero and it uses symmetry of original logistic loss
        */
        VectorND_CUDA_Raw elementwiseSigmoidApproximate() const
        {
            return elementwiseSigmoid();
        }

        /** Compute: (a) difference between current vector and rhs; (b) L2 norm of the result
        * @param rhs vector from which difference is computed
        * @param l2NormOfDifference reference to value which will store the result L2 norm of the result
        * @remark Function is presented for compute optimization
        */
        void computeDiffAndComputeL2Norm(const VectorND_CUDA_Raw& rhs, TElementType& restrict_ext l2NormOfDifference) {
            *this -= rhs;
            l2NormOfDifference = vectorL2Norm();
        }

    public:
        GpuManagement& device() {
            return gpuManagement;
        }
        const GpuManagement& device() const {
            return gpuManagement;
        }
        
    public:

        enum InitPolicyForStorage {
            eNotAllocate = 0,        ///< Warning: Not allocate underlying storage. Use it if you understand what you are doing.
            eAllocNotInit = 1,       ///< Warning: Allocate underlying storage, but not initialized. Use it if you understand what you are doing.
            eAllocAndSetToZero = 2   ///< Allocate underlying storage and initialize to zero
        };

        /** Construct vector with using specific initialization policy
        * @param policy initialization policy
        * @param dimension dimension of dense vector to be allocated
        * @param theGpuManagement reference to GPU management object
        * @remark Use only if you understand what you're doing.
        * @remark Be very careful with uninitilized vectors
        * @remark For std::vector it's tricky to create std::vector with uninitialized values
        */
        VectorND_CUDA_Raw(InitPolicyForStorage policy, size_t dimension, const GpuManagement& theGpuManagement)
        : components(nullptr)
        , componentsCount(dimension)
        , gpuManagement(theGpuManagement)
        {
            if (componentsCount == 0)
                return;

            switch (policy)
            {
                case eNotAllocate:
                {
                    break;
                }
                case eAllocNotInit:
                {
                    components = static_cast<T*>(memAllocationCUDA(components2bytes(componentsCount), gpuManagement));
                    break;
                }
                case eAllocAndSetToZero:
                {                
                    components = static_cast<T*>(memAllocationCUDA(components2bytes(componentsCount), gpuManagement));
                    gpuManagement.setDeviceMemoryToZero(components, componentsCount);
                }
            }
        }
    };

    /** Helper function when factor multiplied by vector from the right
    * @param factor factor for multiply
    * @param v vector which will be multiplied by factor
    * @return result vector
    */
    template <typename T, typename TFactorType>
    inline VectorND_CUDA_Raw<T> operator * (TFactorType factor, const VectorND_CUDA_Raw<T>& v) {
        return v * factor;
    }

    using VectorND_CUDA_Raw_i = VectorND_CUDA_Raw<int32_t>;
    using VectorND_CUDA_Raw_ui = VectorND_CUDA_Raw<uint32_t>;
    using VectorND_CUDA_Raw_f = VectorND_CUDA_Raw<float>;
    using VectorND_CUDA_Raw_d = VectorND_CUDA_Raw<double>;
}

namespace dopt
{
    template <class T>
    const TMemoryAllocationCUDACallback VectorND_CUDA_Raw<T>::memAllocation = memAllocationCUDA;
    
    template <class T>
    const TMemoryFreeCUDACallback VectorND_CUDA_Raw<T>::memFree = memFreeCUDA;
}
