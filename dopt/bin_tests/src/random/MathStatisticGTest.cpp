#include "dopt/random/include/MathStatistics.h"
#include "gtest/gtest.h"

#include <vector>

TEST(dopt, MathStatisticGTest)
{
    int oneElement[] = {4};

    // Basic tests for common aggregations
    {
        int a[] = {1, -1, 3, 2};
        int b[] = {2, 4, 6};
        auto zz = dopt::mathstats::sum(a, sizeof(a) / sizeof(a[0]));

        EXPECT_EQ(5,dopt::mathstats::sum(a, sizeof(a)/sizeof(a[0])));
        EXPECT_EQ(4,dopt::mathstats::sum(oneElement, sizeof(oneElement)/sizeof(oneElement[0])));
        EXPECT_EQ(4,dopt::mathstats::mx(oneElement, sizeof(oneElement)/sizeof(oneElement[0])));

        EXPECT_EQ(4,dopt::mathstats::mx(b, sizeof(b)/sizeof(b[0])));
        EXPECT_EQ(0,dopt::mathstats::minIndex(b, sizeof(b)/sizeof(b[0])));
        EXPECT_EQ(1,dopt::mathstats::minIndex(a, sizeof(a)/sizeof(a[0])));
        EXPECT_EQ(0,dopt::mathstats::minIndex(oneElement, sizeof(oneElement)/sizeof(oneElement[0])));

        EXPECT_EQ(2,dopt::mathstats::maxIndex(b, sizeof(b)/sizeof(b[0])));
        EXPECT_EQ(2,dopt::mathstats::maxIndex(a, sizeof(a)/sizeof(a[0])));
        EXPECT_EQ(0,dopt::mathstats::maxIndex(oneElement, sizeof(oneElement)/sizeof(oneElement[0])));

        EXPECT_EQ(6,dopt::mathstats::maxElement<int>(b, sizeof(b)/sizeof(b[0])));
        EXPECT_EQ(-1,dopt::mathstats::minElement<int>(a, sizeof(a)/sizeof(a[0])));
    }

    {
        int q[]={1, 2, 3, 0, 1, 20, 1};
        size_t minIndex = 11, maxIndex = 11;
        dopt::mathstats::maxAndMinIndex(minIndex, maxIndex, q, sizeof(q)/sizeof(q[0]));
        EXPECT_EQ(minIndex, 3);
        EXPECT_EQ(maxIndex, 5);

        std::vector<int> qvec = { 1, 2, 3, 0, 1, 20, 1 };

        EXPECT_EQ(dopt::mathstats::sum(qvec.begin(), qvec.size()),
                  dopt::mathstats::sum(&qvec[0], qvec.size())
                 );
    }
    {
        size_t minIndex = 11, maxIndex = 11;
        dopt::mathstats::maxAndMinIndex(minIndex, maxIndex, oneElement, sizeof(oneElement)/sizeof(oneElement[0]));
        EXPECT_EQ(minIndex, 0);
        EXPECT_EQ(maxIndex, 0);
    }
    {
        int q[] = { 1, 3, -1 };
        size_t minIndex = 11, maxIndex = 11;
       dopt::mathstats::maxAndMinIndex(minIndex, maxIndex, q, sizeof(q)/sizeof(q[0]));
        EXPECT_EQ(minIndex, 2);
        EXPECT_EQ(maxIndex, 1);
    }

    {
        EXPECT_EQ(12, dopt::mathstats::minValue(12, 14));
        EXPECT_EQ(15, dopt::mathstats::maxValue(15, 14));
    }

    // basic point-wise measuring
    {
        int similarValues[] = {3,3,3,3};
        EXPECT_EQ(3,dopt::mathstats::mx(similarValues, sizeof(similarValues)/sizeof(similarValues[0])));
        EXPECT_EQ(0,dopt::mathstats::dx(similarValues, sizeof(similarValues)/sizeof(similarValues[0])));
	}

	// test correlation
	{
		double a[] = {3.0*1.0, 4.0*1.0, 4.0*1.0, 12.0*1.0};
		double b[] = {3.0*3.0, 4.0*3.0, 4.0*3.0, 12.0*3.0};
		double c[] = {3.0*-2.0+1.0, 4.0*-2.0+1.0, 4.0*-2.0 + 1.0, 12.0*-2.0 + 1.0};
		double d[] = {1.0, 2.0, 3.0, 4.0};
		
		EXPECT_DOUBLE_EQ(1,dopt::mathstats::correlation(a, a, sizeof(a)/sizeof(a[0])));
		EXPECT_DOUBLE_EQ(1,dopt::mathstats::correlation(a, b, sizeof(a)/sizeof(a[0])));
		EXPECT_DOUBLE_EQ(-1,dopt::mathstats::correlation(a, c, sizeof(a)/sizeof(a[0])));
		EXPECT_LE(dopt::mathstats::correlation(a, d, sizeof(d)/sizeof(d[0])), 1);
		EXPECT_GE(dopt::mathstats::correlation(a, d, sizeof(d)/sizeof(d[0])), -1);
	}
}
