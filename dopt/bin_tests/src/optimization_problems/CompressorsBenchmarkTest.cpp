#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"
#include "dopt/random/include/RandomGenXorShiftInteger.h"
#include "dopt/random/include/RandomGenMersenne.h"

#include "dopt/random/include/RandomGenRealLinear.h"

#include "dopt/random/include/Shuffle.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"

#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>

#include <assert.h>

template<class RndGenerator>
void testIndK_Implementation_One_CPU(RndGenerator& generator, size_t K, size_t iterations,
    const dopt::VectorNDRaw_d& input, dopt::VectorNDRaw_d& out, dopt::HighPrecisionTimer* timer = 0)
{
    size_t D = input.size();

    assert(input.size() == out.size());
    assert(D >= K);

    // Initialization phase
    dopt::VectorNDRaw_i indicies(D);
    size_t writePos = 0;

    double pi = double(K) / double(D);
    // In Expectation pi * D coordinates will be selected

    int inv_pi = int(1.0 / pi);

    // 1 / pi = INT
    //   = > If roll an honest dice and take{ 0,...INT - 1 } then the specific number will be taken 
    //       w.p. 1 / INT = 1 / (1 / pi) = pi
    for (size_t i = 0; i < iterations; ++i)
    {
        writePos = 0;

        for (size_t i = 0; i < D; ++i)
        {
            uint32_t choice = generator.generateInteger() % inv_pi;
            if (choice == 0)
            {
                indicies[writePos++] = i;
                out[i] = input[i];
            }
        }
    }
    // std::cout << " >> writePos in: " << writePos << ", K:" << K << '\n';
}

template<class RndGenerator>
void testRandK_Implementation_One_CPU(RndGenerator& generator, size_t K, size_t iterations, 
                                      const dopt::VectorNDRaw_d& input, dopt::VectorNDRaw_d& out, dopt::HighPrecisionTimer* timer = 0)
{
    size_t D = input.size();

    assert(input.size() == out.size());
    assert(D >= K);

    // Initialization phase
    using IndiciesVector = std::vector<size_t>;
    using IndexType = IndiciesVector::value_type;

    IndiciesVector indiciesForVector;
    indiciesForVector.reserve(D);
    for (IndexType i = 0; i < D; ++i)
        indiciesForVector.push_back(i);

    for (size_t i = 0; i < iterations; ++i)
    {
        IndiciesVector indicies(indiciesForVector);
        indicies = indiciesForVector;
        // Indicies generation phases
        dopt::shuffle(indicies, K, generator);
        indicies.resize(K);

        // Fetching and and filling output phase
        for (size_t j = 0; j < K; ++j)
        {
            out[indicies[j]] = input[indicies[j]];
        }
    }
}

template<class RndGenerator>
void testRandK_Implementation_One_CPU_with_known_ind(RndGenerator& generator, size_t K, size_t iterations,
                                                    const dopt::VectorNDRaw_d& input, dopt::VectorNDRaw_d& out, dopt::HighPrecisionTimer* timer = 0)
{
    size_t D = input.size();

    assert(input.size() == out.size());
    assert(D >= K);

    // Initialization phase
    using IndiciesVector = std::vector<size_t>;
    using IndexType = IndiciesVector::value_type;

    timer->pause();
    IndiciesVector indiciesForVector;
    indiciesForVector.reserve(D);
    for (IndexType i = 0; i < D; ++i)
        indiciesForVector.push_back(i);
    timer->resume();

    for (size_t i = 0; i < iterations; ++i)
    {
        timer->pause();
        IndiciesVector indicies(indiciesForVector);
        indicies = indiciesForVector;
        // Indicies generation phases
        dopt::shuffle(indicies, K, generator);
        indicies.resize(K);
        timer->resume();

        // Fetching and and filling output phase
        for (size_t j = 0; j < K; ++j)
        {
            out[indicies[j]] = input[indicies[j]];
        }
    }
}

template<class RndGenerator>
void testSeqK_Implementation_One_CPU(RndGenerator& generator, size_t K, size_t iterations,
                                     const dopt::VectorNDRaw_d& input, dopt::VectorNDRaw_d& out, dopt::HighPrecisionTimer* timer = 0)
{
    size_t D = input.size();

    assert(input.size() == out.size());
    assert(D >= K);

    // Initialization phase

    for (size_t i = 0; i < iterations; ++i)
    {
        // No indicies generation phases is needed

        size_t pPos = size_t(generator.generateInteger() % D);
        size_t first_part_len = std::min(K, D - pPos);
        size_t second_part_len = K - first_part_len;

        dopt::VectorNDRaw_d::TElementType* outData = out.data();
        const dopt::VectorNDRaw_d::TElementType* inData = input.dataConst();

        dopt::CopyHelpers::copy(outData + pPos, inData + pPos, first_part_len);
        dopt::CopyHelpers::copy(outData, inData, second_part_len);
    }
}

template<class RndGenerator>
void testSelectionK_Implementation_One_CPU(RndGenerator& generator, size_t K, size_t iterations, const dopt::VectorNDRaw_d& input, dopt::VectorNDRaw_d& out, dopt::HighPrecisionTimer* timer = 0)
{
    size_t D = input.size();
    size_t kCoordinatesPerBucket = K;
    size_t kBuckets = D / K;

    assert(D % K == 0);
    assert(input.size() == out.size());
    assert(D >= K);

    // Initialization phase
    for (size_t i = 0; i < iterations; ++i)
    {
        // No indicies generation phases is needed
        size_t pBucket = size_t(generator.generateInteger() % kBuckets);
        size_t pPos = pBucket * kCoordinatesPerBucket;

        dopt::VectorNDRaw_d::TElementType* outData = out.data();
        const dopt::VectorNDRaw_d::TElementType* inData = input.dataConst();

        dopt::CopyHelpers::copy(outData + pPos, inData + pPos, kCoordinatesPerBucket);
    }
}

TEST(dopt, CompressorsCheckUnbiasedGTest)
{
    size_t D = 50;
    size_t K = 10;
    size_t trials = 100000;
    uint32_t seed = 123;


    dopt::VectorNDRaw_d output(D);
    dopt::VectorNDRaw_d input(D);
    dopt::VectorNDRaw_d sum(D);

    dopt::RandomGenRealLinear generatorReal;
    generatorReal.setSeed(seed);

    dopt::RandomGenIntegerLinear generator;
    generator.setSeed(seed);

    {
        input.setAllRandomly(generatorReal);
        sum.setAllToDefault();
        for (size_t i = 0; i < trials; ++i)
        {
            output.setAllToDefault();
            testRandK_Implementation_One_CPU(generator, K, 1, input, output);

            output *= (double(D) / double(K));                                  // scaling            
            sum = sum * double(i) / (double(i) + 1.0) + output / (double(i) + 1.0); // rank-one summation
        }

        double err1 = (sum - input).vectorLinfNorm();
        double err2 = input.vectorLinfNorm();
        
        EXPECT_TRUE(err1 < (0.9 * err2)) << "testRandK_Implementation_One_CPU -- IS UNBIASED TEST";        
    }

    {
        input.setAllRandomly(generatorReal);
        sum.setAllToDefault();
        for (size_t i = 0; i < trials; ++i)
        {
            output.setAllToDefault();
            testSeqK_Implementation_One_CPU(generator, K, 1, input, output);

            output *= (double(D) / double(K));                                      // scaling            
            sum = sum * double(i) / (double(i) + 1.0) + output / (double(i) + 1.0); // rank-one summation
        }
        double err1 = (sum - input).vectorLinfNorm();
        double err2 = input.vectorLinfNorm();
        EXPECT_TRUE(err1 < (0.1 * err2)) << "testSeqK_Implementation_One_CPU -- IS UNBIASED TEST";
    }

    {
        input.setAllRandomly(generatorReal);
        sum.setAllToDefault();
        for (size_t i = 0; i < trials; ++i)
        {
            output.setAllToDefault();
            testSelectionK_Implementation_One_CPU(generator, K, 1, input, output);

            output *= (double(D) / double(K));                                      // scaling            
            sum = sum * double(i) / (double(i) + 1.0) + output / (double(i) + 1.0); // rank-one summation
        }
        double err1 = (sum - input).vectorLinfNorm();
        double err2 = input.vectorLinfNorm();
        EXPECT_TRUE(err1 < (0.1 * err2)) << "testSelectionK_Implementation_One_CPU -- IS UNBIASED TEST";        
    }

    {
        input.setAllRandomly(generatorReal);
        sum.setAllToDefault();
        for (size_t i = 0; i < trials; ++i)
        {
            output.setAllToDefault();
            testIndK_Implementation_One_CPU(generator, K, 1, input, output);

            output *= (double(D) / double(K));                                      // scaling            
            sum = sum * double(i) / (double(i) + 1.0) + output / (double(i) + 1.0); // rank-one summation
        }
        double err1 = (sum - input).vectorLinfNorm();
        double err2 = input.vectorLinfNorm();

        EXPECT_TRUE(err1 < (0.1 * err2)) << "testIndK_Implementation_One_CPU -- IS UNBIASED TEST";
    }
}

template<class RndGenerator>
void makeCompressorsBenchmarkTest(size_t D, size_t K, size_t trials, const char* rndGeneratorType)
{
    uint32_t seed = 123;

    dopt::VectorNDRaw_d output(D);
    dopt::VectorNDRaw_d input(D);
    dopt::HighPrecisionTimer tm1;

    RndGenerator generator;

    generator.setSeed(seed);
    testRandK_Implementation_One_CPU(generator, K, 1, input, output, &tm1);

    tm1.reset();
    testRandK_Implementation_One_CPU(generator, K, trials, input, output, &tm1);
    double eclapsedTimeRandK = tm1.getTimeSec();

    tm1.reset();
    testRandK_Implementation_One_CPU_with_known_ind(generator, K, trials, input, output, &tm1);
    double eclapsedTimeRandKWithInd = tm1.getTimeSec();

    generator.setSeed(seed);
    testSeqK_Implementation_One_CPU(generator, K, 1, input, output, &tm1);

    tm1.reset();
    testSeqK_Implementation_One_CPU(generator, K, trials, input, output, &tm1);
    double eclapsedTimeSeqK = tm1.getTimeSec();

    generator.setSeed(seed);
    testSelectionK_Implementation_One_CPU(generator, K, 1, input, output, &tm1);

    tm1.reset();
    testSelectionK_Implementation_One_CPU(generator, K, trials, input, output, &tm1);
    double eclapsedTimeSelectionK = tm1.getTimeSec();

    generator.setSeed(seed);
    testIndK_Implementation_One_CPU(generator, K, 1, input, output, &tm1);
    tm1.reset();
    testIndK_Implementation_One_CPU(generator, K, trials, input, output, &tm1);
    double eclapsedTimeIndK = tm1.getTimeSec();

    std::cout << "D:" << D << ",K:" << K << ",trials = " << trials << '\n';
    std::cout << '\n';
    std::cout << "Used Random Number Generator: " << rndGeneratorType << '\n';
    std::cout << '\n';
    std::cout << "testRandK_Implementation_One_CPU_with_known_ind      -- eclapsedTime = " << eclapsedTimeRandKWithInd << " seconds\n";
    std::cout << "testRandK_Implementation_One_CPU      -- eclapsedTime = " << eclapsedTimeRandK << " seconds\n";
    std::cout << "testSeqK_Implementation_One_CPU       -- eclapsedTime = " << eclapsedTimeSeqK << " seconds\n";
    std::cout << "testSelectionK_Implementation_One_CPU -- eclapsedTime = " << eclapsedTimeSelectionK << " seconds\n";
    std::cout << "testIndK_Implementation_One_CPU -- eclapsedTime = " << eclapsedTimeIndK << " seconds\n";
    std::cout << '\n';
}

TEST(dopt, CompressorsGPerf)
{
    {
        size_t D = 100 * 10000;
        size_t K = 10 * 10000;
        size_t trials = 20;

        makeCompressorsBenchmarkTest<dopt::RandomGenIntegerLinear>(D, K, trials, "Congruent Linear Generator");
        makeCompressorsBenchmarkTest<dopt::RandomGenXorShiftInteger>(D, K, trials, "Xorshift* Algorithm");
        makeCompressorsBenchmarkTest<dopt::RandomGenMersenne>(D, K, trials, "Mersenne Twister Algorithm");
    }

    {
        size_t D = 100 * 10000;
        size_t K = 20 * 10000;
        size_t trials = 20;

        makeCompressorsBenchmarkTest<dopt::RandomGenIntegerLinear>(D, K, trials, "Congruent Linear Generator");
        makeCompressorsBenchmarkTest<dopt::RandomGenXorShiftInteger>(D, K, trials, "Xorshift* Algorithm");
        makeCompressorsBenchmarkTest<dopt::RandomGenMersenne>(D, K, trials, "Mersenne Twister Algorithm");
    }

    {
        size_t D = 100 * 10000;
        size_t K = 50 * 10000;
        size_t trials = 20;

        makeCompressorsBenchmarkTest<dopt::RandomGenIntegerLinear>(D, K, trials, "Congruent Linear Generator");
        makeCompressorsBenchmarkTest<dopt::RandomGenXorShiftInteger>(D, K, trials, "Xorshift* Algorithm");
        makeCompressorsBenchmarkTest<dopt::RandomGenMersenne>(D, K, trials, "Mersenne Twister Algorithm");
    }
}
