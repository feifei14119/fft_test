#include <cassert>
#include <complex>
#include <iostream>
#include <vector>
#include <map>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime_api.h>

#include "hip/hip_runtime.h"
#include "rocfft.h"

#define TEST_LENGTH (100)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_WORK_GROUP_SIZE 1024
#define TWO_PI (-6.283185307179586476925286766559)
#define TWO_PI_FP32 (6.283185307179586476925286766559F)
#define TWO_PI_FP64 (6.283185307179586476925286766559)

/* radix table: tell the FFT algorithms for size <= 4096 ; required by twiddle, passes, and kernel*/
struct SpecRecord
{
    size_t length;
    size_t workGroupSize;
    size_t numTransforms;
    size_t numPasses;
    size_t radices[12]; // Setting upper limit of number of passes to 12
};
inline const std::vector<SpecRecord>& GetRecord()
{
    static const std::vector<SpecRecord> specRecord = 
	{
        //  Length, WorkGroupSize (thread block size), NumTransforms , NumPasses,
        //  Radices
        //  vector<size_t> radices; NUmPasses = radices.size();
        //  Tuned for single precision on OpenCL stack; double precsion use the
        //  same table as single
        {4096, 256,  1, 3, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // pow2
        {2048, 256,  1, 4,  8,  8,  8, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        {1024, 128,  1, 4,  8,  8,  4, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        { 512,  64,  1, 3,  8,  8,  8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 256,  64,  1, 4,  4,  4,  4, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        { 128,  64,  4, 3,  8,  4,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  64,  64,  4, 3,  4,  4,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  32,  64, 16, 2,  8,  4,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  16,  64, 16, 2,  4,  4,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   8,  64, 32, 2,  4,  2,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   4,  64, 32, 2,  2,  2,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   2,  64, 64, 1,  2,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    return specRecord;
}
inline void DetermineSizes(const size_t& length, size_t& workGroupSize, size_t& numTrans)
{
    assert(MAX_WORK_GROUP_SIZE >= 64);

    if(length == 1) // special case
    {
        workGroupSize = 64;
        numTrans      = 64;
        return;
    }

    size_t baseRadix[]   = {13, 11, 7, 5, 3, 2}; // list only supported primes
    size_t baseRadixSize = sizeof(baseRadix) / sizeof(baseRadix[0]);

    size_t                   l = length;
    std::map<size_t, size_t> primeFactorsExpanded;
    for(size_t r = 0; r < baseRadixSize; r++)
    {
        size_t rad = baseRadix[r];
        size_t e   = 1;
        while(!(l % rad))
        {
            l /= rad;
            e *= rad;
        }

        primeFactorsExpanded[rad] = e;
    }

    assert(l == 1); // Makes sure the number is composed of only supported primes

    if(primeFactorsExpanded[2] == length) // Length is pure power of 2
    {
        if(length >= 1024)
        {
            workGroupSize = (MAX_WORK_GROUP_SIZE >= 256) ? 256 : MAX_WORK_GROUP_SIZE;
            numTrans      = 1;
        }
        else if(length == 512)
        {
            workGroupSize = 64;
            numTrans      = 1;
        }
        else if(length >= 16)
        {
            workGroupSize = 64;
            numTrans      = 256 / length;
        }
        else
        {
            workGroupSize = 64;
            numTrans      = 128 / length;
        }
    }
    else if(primeFactorsExpanded[3] == length) // Length is pure power of 3
    {
        workGroupSize = (MAX_WORK_GROUP_SIZE >= 256) ? 243 : 27;
        numTrans      = length >= 3 * workGroupSize ? 1 : (3 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[5] == length) // Length is pure power of 5
    {
        workGroupSize = (MAX_WORK_GROUP_SIZE >= 128) ? 125 : 25;
        numTrans      = length >= 5 * workGroupSize ? 1 : (5 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[7] == length) // Length is pure power of 7
    {
        workGroupSize = 49;
        numTrans      = length >= 7 * workGroupSize ? 1 : (7 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[11] == length) // Length is pure power of 11
    {
        workGroupSize = 121;
        numTrans      = length >= 11 * workGroupSize ? 1 : (11 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[13] == length) // Length is pure power of 13
    {
        workGroupSize = 169;
        numTrans      = length >= 13 * workGroupSize ? 1 : (13 * workGroupSize) / length;
    }
    else
    {
        size_t leastNumPerWI    = 1; // least number of elements in one work item
        size_t maxWorkGroupSize = MAX_WORK_GROUP_SIZE; // maximum work group size desired

        if(primeFactorsExpanded[2] * primeFactorsExpanded[3] == length)
        {
            if(length % 12 == 0)
            {
                leastNumPerWI    = 12;
                maxWorkGroupSize = 128;
            }
            else
            {
                leastNumPerWI    = 6;
                maxWorkGroupSize = 256;
            }
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[5] == length)
        {
            if(length % 20 == 0)
            {
                leastNumPerWI    = 20;
                maxWorkGroupSize = 64;
            }
            else
            {
                leastNumPerWI    = 10;
                maxWorkGroupSize = 128;
            }
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 14;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[5] == length)
        {
            leastNumPerWI    = 15;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 21;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[5] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 35;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[5] == length)
        {
            leastNumPerWI    = 30;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 42;
            maxWorkGroupSize = 60;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[5] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 70;
            maxWorkGroupSize = 36;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[5] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 105;
            maxWorkGroupSize = 24;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[11] == length)
        {
            leastNumPerWI    = 22;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[13] == length)
        {
            leastNumPerWI    = 26;
            maxWorkGroupSize = 128;
        }
        else
        {
            leastNumPerWI    = 210;
            maxWorkGroupSize = 12;
        }

        if(maxWorkGroupSize > MAX_WORK_GROUP_SIZE)
            maxWorkGroupSize = MAX_WORK_GROUP_SIZE;
        assert(leastNumPerWI > 0 && length % leastNumPerWI == 0);

        for(size_t lnpi = leastNumPerWI; lnpi <= length; lnpi += leastNumPerWI)
        {
            if(length % lnpi != 0)
                continue;

            if(length / lnpi <= MAX_WORK_GROUP_SIZE)
            {
                leastNumPerWI = lnpi;
                break;
            }
        }

        numTrans      = maxWorkGroupSize / (length / leastNumPerWI);
        numTrans      = numTrans < 1 ? 1 : numTrans;
        workGroupSize = numTrans * (length / leastNumPerWI);
    }

    assert(workGroupSize <= MAX_WORK_GROUP_SIZE);
}
std::vector<size_t> GetRadices(size_t length)
{
    std::vector<size_t> radices;

    // get number of items in this table
    std::vector<SpecRecord> specRecord  = GetRecord();
    size_t tableLength = specRecord.size();

    printf("tableLength=%zu\n", tableLength);
    for(int i = 0; i < tableLength; i++)
    {
        if(length == specRecord[i].length)
        { // if find the matched size

            size_t numPasses = specRecord[i].numPasses;
            printf("numPasses=%zu, table item %d \n", numPasses, i);
            for(int j = 0; j < numPasses; j++)
            {
                radices.push_back((specRecord[i].radices)[j]);
            }
            break;
        }
    }

    // if not in the table, then generate the radice order with the algorithm.
    if(radices.size() == 0)
    {
        size_t R = length;

        // Possible radices
        size_t cRad[]   = {13, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1}; // Must be in descending order
        size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

        size_t workGroupSize;
        size_t numTrans;
        // need to know workGroupSize and numTrans
        DetermineSizes(length, workGroupSize, numTrans);
        size_t cnPerWI = (numTrans * length) / workGroupSize;

        // Generate the radix and pass objects
        while(true)
        {
            size_t rad;

            // Picks the radices in descending order (biggest radix first) performanceCC
            // purpose
            for(size_t r = 0; r < cRadSize; r++)
            {

                rad = cRad[r];
                if((rad > cnPerWI) || (cnPerWI % rad))
                    continue;

                if(!(R % rad)) // if not a multiple of rad, then exit
                    break;
            }

            assert((cnPerWI % rad) == 0);

            R /= rad;
            radices.push_back(rad);

            assert(R >= 1);
            if(R == 1)
                break;

        } // end while
    } // end if(radices == empty)

	printf("radices size = %zu:\n",radices.size());
	for(std::vector<size_t>::const_iterator i = radices.begin(); i != radices.end(); i++)
		printf("\t%lu\n",*i);
	
    return radices;
}
class TwiddleTable
{
    size_t N;
    float2 * wc;

public:
    TwiddleTable(size_t length) : N(length)    {  wc = new float2[N];    }

    ~TwiddleTable()    { delete[] wc;    }

    float2 * GenerateTwiddleTable(const std::vector<size_t>& radices)
    {
		size_t radix;
        size_t L  = 1;
        size_t nt = 0;
		size_t idxcnt = 0;
        for(std::vector<size_t>::const_iterator i = radices.begin(); i != radices.end(); i++)
        {
            radix = *i;
            L *= radix;
			
			printf("radix = %zu, L = %zu\n", radix, L);

            // Twiddle factors
            for(size_t k = 0; k < (L / radix); k++)
            {
                double theta = TWO_PI * (k) / (L);
				printf("	[k/L] = [%zu/%zu], theta = %.3f\n", k, L, theta);
				
				{
					size_t j = 0;
					double c = cos((j)*theta);
					double s = sin ((j)*theta);
					printf("		j = %zu, alph = %.3f, c = %.3f, s = %.3f\n", j, j*theta, c,s);
				}
					
                for(size_t j = 1; j < radix; j++)
                {
                    double c = cos((j)*theta);
                    double s = sin ((j)*theta);
					printf("[%zu]		j = %zu, alph = %.3f, c = %.3f, s = %.3f\n", idxcnt, j, j*theta, c,s);
					idxcnt++;

                    wc[nt].x = c;
                    wc[nt].y = s;					
                    nt++;
                }
            }
        }

		printf("\n");
        return wc;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define C5QA 0.30901699437494742410229341718282f
#define C5QB 0.95105651629515357211643933337938f
#define C5QC 0.50000000000000000000000000000000f
#define C5QD 0.58778525229247312916870595463907f
#define C5QE 0.80901699437494742410229341718282f

__device__ void FwdRad10B1(float2* R0, float2* R1, float2* R2, float2* R3, float2* R4, float2* R5, float2* R6, float2* R7, float2* R8, float2* R9)
{
    float TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5, TR6, TI6, TR7, TI7, TR8, TI8, TR9, TI9;

    TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;
    TR2 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) + C5QB * ((*R2).y - (*R8).y) + C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR8 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) - C5QB * ((*R2).y - (*R8).y) - C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) - C5QB * ((*R4).y - (*R6).y) + C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));
    TR6 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) + C5QB * ((*R4).y - (*R6).y) - C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));

    TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;
    TI2 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) - C5QB * ((*R2).x - (*R8).x) - C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI8 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) + C5QB * ((*R2).x - (*R8).x) + C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) + C5QB * ((*R4).x - (*R6).x) - C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));
    TI6 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) - C5QB * ((*R4).x - (*R6).x) + C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));

    TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;
    TR3 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) + C5QB * ((*R3).y - (*R9).y) + C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR9 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) - C5QB * ((*R3).y - (*R9).y) - C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR5 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) - C5QB * ((*R5).y - (*R7).y) + C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));
    TR7 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) + C5QB * ((*R5).y - (*R7).y) - C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));

    TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;
    TI3 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) - C5QB * ((*R3).x - (*R9).x) - C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI9 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) + C5QB * ((*R3).x - (*R9).x) + C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI5 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) + C5QB * ((*R5).x - (*R7).x) - C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));
    TI7 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) - C5QB * ((*R5).x - (*R7).x) + C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C5QE * TR3 + C5QD * TI3);
    (*R2).x = TR4 + (C5QA * TR5 + C5QB * TI5);
    (*R3).x = TR6 + (-C5QA * TR7 + C5QB * TI7);
    (*R4).x = TR8 + (-C5QE * TR9 + C5QD * TI9);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (-C5QD * TR3 + C5QE * TI3);
    (*R2).y = TI4 + (-C5QB * TR5 + C5QA * TI5);
    (*R3).y = TI6 + (-C5QB * TR7 - C5QA * TI7);
    (*R4).y = TI8 + (-C5QD * TR9 - C5QE * TI9);

    (*R5).x = TR0 - TR1;
    (*R6).x = TR2 - (C5QE * TR3 + C5QD * TI3);
    (*R7).x = TR4 - (C5QA * TR5 + C5QB * TI5);
    (*R8).x = TR6 - (-C5QA * TR7 + C5QB * TI7);
    (*R9).x = TR8 - (-C5QE * TR9 + C5QD * TI9);

    (*R5).y = TI0 - TI1;
    (*R6).y = TI2 - (-C5QD * TR3 + C5QE * TI3);
    (*R7).y = TI4 - (-C5QB * TR5 + C5QA * TI5);
    (*R8).y = TI6 - (-C5QB * TR7 - C5QA * TI7);
    (*R9).y = TI8 - (-C5QD * TR9 - C5QE * TI9);
}
__device__ void FwdPass0_len100(const float2 *twiddles, 
																const size_t stride_in, const size_t stride_out, 
																unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, 
																float2 *bufIn, 
																float *bufOutRe, float*bufOutIm, 
																float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7, float2 *R8, float2 *R9)
{
	if(rw)
	{
		(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*stride_in];
		(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 10 )*stride_in];
		(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 20 )*stride_in];
		(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 30 )*stride_in];
		(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 40 )*stride_in];
		(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 50 )*stride_in];
		(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 60 )*stride_in];
		(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 70 )*stride_in];
		(*R8) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*stride_in];
		(*R9) = bufIn[inOffset + ( 0 + me*1 + 0 + 90 )*stride_in];
	}

	FwdRad10B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9);	

	if(rw)
	{
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 1 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 2 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 3 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 4 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 5 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 6 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 7 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 8 ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 9 ) ] = (*R9).x;		
		__syncthreads();

		(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 10 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 20 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 30 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 40 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 50 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 60 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 70 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 80 ) ];
		(*R9).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 90 ) ];
		__syncthreads();
	
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 1 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 2 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 3 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 4 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 5 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 6 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 7 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 8 ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*10 + (1*me + 0)%1 + 9 ) ] = (*R9).y;	
		__syncthreads();
	
		(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 10 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 20 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 30 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 40 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 50 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 60 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 70 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 80 ) ];
		(*R9).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 90 ) ];
	}
	
	__syncthreads();
}
__device__ void FwdPass1_len100(const float2  *twiddles, 
																const size_t stride_in, const size_t stride_out, 
																unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, 
																float *bufInRe, float *bufInIm, 
																float2  *bufOut, 
																float2  *R0, float2  *R1, float2  *R2, float2  *R3, float2  *R4, float2  *R5, float2  *R6, float2  *R7, float2  *R8, float2  *R9)
{
	{
		//float2  W;
		float TR, TI;
		float wx, wy, rx, ry;
		
		double phase;
		float phase_fp;
		uint32_t k = me;//j=Wn;
		uint32_t L = 100;

		 //float2 W1;// = twiddles[9 + 9*((1*me + 0)%10) + 0];
		 //float2 W2;// = twiddles[9 + 9*((1*me + 0)%10) + 1];
		 //float2 W3;// = twiddles[9 + 9*((1*me + 0)%10) + 2];
		 //float2 W4;// = twiddles[9 + 9*((1*me + 0)%10) + 3];
		 //float2 W5;// = twiddles[9 + 9*((1*me + 0)%10) + 4];
		 //float2 W6;// = twiddles[9 + 9*((1*me + 0)%10) + 5];
		 //float2 W7;// = twiddles[9 + 9*((1*me + 0)%10) + 6];
		 //float2 W8;// = twiddles[9 + 9*((1*me + 0)%10) + 7];
		 //float2 W9;// = twiddles[9 + 9*((1*me + 0)%10) + 8];

		phase = -1.0/L * 1 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		//wx = cosf(float(phase * TWO_PI_FP64)); 
		//wy = sinf(float(phase * TWO_PI_FP64)); 
		float2 W1 = twiddles[9 + 9*((1*me + 0)%10) + 0];
		wx = W1.x; wy = W1.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		phase = -1.0/L * 2 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		//wx = cosf(float(phase * TWO_PI_FP64)); 
		//wy = sinf(float(phase * TWO_PI_FP64)); 
		float2 W2 = twiddles[9 + 9*((1*me + 0)%10) + 1];
		wx = W2.x; wy = W2.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		phase = -1.0/L * 3 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W3 = twiddles[9 + 9*((1*me + 0)%10) + 2];
		wx = W3.x; wy = W3.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		phase = -1.0/L * 4 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W4 = twiddles[9 + 9*((1*me + 0)%10) + 3];
		wx = W4.x; wy = W4.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		phase = -1.0/L * 5 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W5 = twiddles[9 + 9*((1*me + 0)%10) + 4];
		wx = W5.x; wy = W5.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		phase = -1.0/L * 6 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W6 = twiddles[9 + 9*((1*me + 0)%10) + 5];
		wx = W6.x; wy = W6.y;
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
		
		phase = -1.0/L * 7 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W7 = twiddles[9 + 9*((1*me + 0)%10) + 6];
		wx = W7.x; wy = W7.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		phase = -1.0/L * 8 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W8 = twiddles[9 + 9*((1*me + 0)%10) + 7];
		wx = W8.x; wy = W8.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
		
		phase = -1.0/L * 9 * k;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		float2 W9 = twiddles[9 + 9*((1*me + 0)%10) + 8];
		wx = W9.x; wy = W9.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	FwdRad10B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9);	

	if(rw)
	{
		bufOut[outOffset + ( 1*me + 0 + 0  )*stride_out] = (*R0);
		bufOut[outOffset + ( 1*me + 0 + 10 )*stride_out] = (*R1);
		bufOut[outOffset + ( 1*me + 0 + 20 )*stride_out] = (*R2);
		bufOut[outOffset + ( 1*me + 0 + 30 )*stride_out] = (*R3);
		bufOut[outOffset + ( 1*me + 0 + 40 )*stride_out] = (*R4);
		bufOut[outOffset + ( 1*me + 0 + 50 )*stride_out] = (*R5);
		bufOut[outOffset + ( 1*me + 0 + 60 )*stride_out] = (*R6);
		bufOut[outOffset + ( 1*me + 0 + 70 )*stride_out] = (*R7);
		bufOut[outOffset + ( 1*me + 0 + 80 )*stride_out] = (*R8);
		bufOut[outOffset + ( 1*me + 0 + 90 )*stride_out] = (*R9);
	}	
}

__device__ void fwd_len100_device(const float2  *twiddles, 
																const size_t stride_in, const size_t stride_out, 
																unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, 
																float2  *lwbIn, float2  *lwbOut, 
																float  *lds)
{
	float2  R0, R1, R2, R3, R4, R5, R6, R7, R8, R9;
	FwdPass0_len100(twiddles, stride_in, stride_out,       rw, b, me, 0, ldsOffset,     lwbIn, lds, lds,         &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9);
	FwdPass1_len100(twiddles, stride_in, stride_out,       rw, b, me, ldsOffset, 0,     lds, lds,  lwbOut,      &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9);
}
__global__ void fft_fwd_op_len100( const float2  * twiddles, float2  * gbIn, float2  * gbOut)
{
	__shared__ float  lds[1200];
	
	size_t dim = 1;
	size_t lengths[3];
	size_t stride_in[4];
	size_t stride_out[4];
	size_t batch_count = 1;
	lengths[0] = 100; lengths[1] = 1; lengths[2] = 1;
	stride_in[0] = 1;stride_in[1] = lengths[0];stride_in[2] = lengths[0]*lengths[1];stride_in[3] = lengths[0]*lengths[1]*lengths[2];
	stride_out[0] = 1;stride_out[1] = lengths[0];stride_out[2] = lengths[0]*lengths[1];stride_out[3] = lengths[0]*lengths[1]*lengths[2];
	
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2  *lwbIn;
	float2  *lwbOut;

	unsigned int upper_count = batch_count;
	
	for(int i=1; i<dim; i++)
	{
		upper_count *= lengths[i];
	}
	// do signed math to guard against underflow
	unsigned int rw = (static_cast<int>(me) < (static_cast<int>(upper_count)  - static_cast<int>(batch)*12)*10) ? 1 : 0;
	unsigned int b = 0;

	size_t counter_mod = (batch*12 + (me/10));
	if(dim == 1)
	{
		iOffset += counter_mod*stride_in[1];
		oOffset += counter_mod*stride_out[1];
	}
	else if(dim == 2)
	{
		int counter_1 = counter_mod / lengths[1];
		int counter_mod_1 = counter_mod % lengths[1];
		iOffset += counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	else if(dim == 3)
	{
		int counter_2 = counter_mod / (lengths[1] * lengths[2]);
		int counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);
		int counter_1 = counter_mod_2 / lengths[1];
		int counter_mod_1 = counter_mod_2 % lengths[1];
		iOffset += counter_2*stride_in[3] + counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_2*stride_out[3] + counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fwd_len100_device(twiddles, stride_in[0], stride_out[0],    rw, b, me%10, (me/10)*100,    lwbIn, lwbOut, lds);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void test_gpu(float2 * x, float2 * y)
{	
	const size_t Nx = TEST_LENGTH;
	
	printf("twiddles_create_pr \n");
	std::vector<size_t> radices;
	radices = GetRadices(Nx);

	float2 * twtc;
	float2 * dtw = NULL;
	TwiddleTable twTable(Nx);
	twtc = twTable.GenerateTwiddleTable(radices);
	hipMalloc(&dtw, Nx * sizeof(float2));
	hipMemcpy(dtw, twtc, Nx * sizeof(float2), hipMemcpyHostToDevice);	

	std::cout << "gpu test\n";
	hipLaunchKernelGGL(fft_fwd_op_len100, dim3(1, 1), dim3(10, 1), 
					0, 0, 
					dtw, x, y);
					
	if(1)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			hipLaunchKernelGGL(fft_fwd_op_len100, dim3(1, 1), dim3(10, 1), 0, 0, dtw, x, y);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
}

int main(int argc, char* argv[])
{		
	printf("\n***************************************************\n");
	std::cout << "rocFFT complex 1d FFT example";
	printf("\n***************************************************\n");

	// The problem size
	const size_t Nx = (argc < 2) ? TEST_LENGTH : atoi(argv[1]);
	const size_t batch = 1;
	const size_t dimension = 1;
	std::cout << "Nx: " << Nx  << std::endl;

	std::vector<std::complex<float>> cx(Nx*batch);
	std::vector<std::complex<float>> cy(Nx*batch);	
	std::vector<std::complex<float>> mycy(Nx*batch);	
    std::vector<std::complex<float>> backx(cx.size());
	for(size_t i = 0; i < Nx; ++i)
	{
		cx[i] = std::complex<float>(1.0f*i, -0.1f*i);
		cy[i] = std::complex<float>(0,0);
	}
	if(0)
	{
		std::cout << "Input:\n";
		for(size_t i = 0; i < Nx; ++i)
			std::cout << real(cx[i]) << ", " << imag(cx[i]) << "\n";
		std::cout << "\n";
	}

	// Create HIP device objects:
	float2 * x = NULL;
	float2 * y = NULL;
	float2 * myy = NULL;
	hipMalloc(&x, cx.size() * sizeof(float2));
	hipMalloc(&y, cy.size() * sizeof(float2));
	hipMalloc(&myy, cy.size() * sizeof(float2));
	hipMemcpy(x, cx.data(), cx.size() * sizeof(float2), hipMemcpyHostToDevice);

	// Length are in reverse order because rocfft is column-major.
	const size_t lengths[1] = {Nx};
	rocfft_status status = rocfft_status_success;
	

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed:\n";
	// Create plans
	rocfft_plan forward = NULL;
	status = rocfft_plan_create(&forward,
                                rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                dimension,
                                lengths,
                                batch,
                                NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info forwardinfo = NULL;
	status = rocfft_execution_info_create(&forwardinfo); 									assert(status == rocfft_status_success);
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(forward, &fbuffersize);								assert(status == rocfft_status_success);
	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);    	assert(status == rocfft_status_success);

	// Execute the forward transform
	status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo); 					assert(status == rocfft_status_success);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(float2), hipMemcpyDeviceToHost);
	if(0)
	{
		std::cout << "Output:\n";
		for(size_t i = 0; i < 10; ++i)
			std::cout << real(cy[i]) << ", " << imag(cy[i]) << "\n";
	}
	
	if(0)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}

	std::cout << "\n";
	//return 0;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	test_gpu(x, myy);
	hipMemcpy(mycy.data(), myy, mycy.size() * sizeof(float2), hipMemcpyDeviceToHost);
	
	if(0)
	{
		std::cout << "My output:\n";
		for(size_t i = 0; i < 10; ++i)
			std::cout << real(mycy[i]) << ", " << imag(mycy[i]) << "\n";
	}
	
	float errormy = 0.0f;
	for(size_t i = 0; i < Nx; i++)
	{
		float diffx = std::abs(real(mycy[i]) - real(cy[i]));
		float diffy = std::abs(imag(mycy[i]) - imag(cy[i]));
		float diff = diffx + diffy;
		if(diff > errormy)
			errormy = diff;
		/*if(diff > 1.0f)
		{
			std::cout << "[" << i << "]";
			std::cout << real(cy[i]) << ", " << imag(cy[i]) << ";\t";
			std::cout << real(mycy[i]) << ", " << imag(mycy[i]) << "\n";
		}*/
	}
	std::cout << "Maximum error: " << errormy << "\n";
	return 0;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed back:\n";
	// Create plans
	rocfft_plan backward = NULL;
	status = rocfft_plan_create(&backward,
                                rocfft_placement_notinplace,
                                rocfft_transform_type_complex_inverse,
                                rocfft_precision_single,
                                dimension,
                                lengths,
                                batch,
                                NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info backwardinfo = NULL;
	status = rocfft_execution_info_create(&backwardinfo);  									assert(status == rocfft_status_success);
	size_t bbuffersize = 0;
	status = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);    					assert(status == rocfft_status_success);
	void* bbuffer = NULL;
	hipMalloc(&bbuffer, bbuffersize);
	status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);    	assert(status == rocfft_status_success);
	
	// Execute the backward transform
	status = rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);     			assert(status == rocfft_status_success);
	
	hipMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
	if(1)
	{
		std::cout << "Output:\n";
		for(size_t i = 0; i < Nx; ++i)
			std::cout << real(backx[i])/Nx << ", " << imag(backx[i])/Nx << "\n";
	}

	if(0)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
	//return 0;
	std::cout << "\n";	
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	float error = 0.0f;
	for(size_t i = 0; i < Nx; i++)
	{
		float diff = std::abs(real(backx[i]) / Nx - real(cx[i]));
		if(diff > error)
			error = diff;
	}
	std::cout << "Maximum error: " << error << "\n";	
	
	hipFree(x);
	hipFree(y);
	hipFree(fbuffer);
	hipFree(bbuffer);

	// Destroy plans
	rocfft_plan_destroy(forward);
	rocfft_plan_destroy(backward);

	rocfft_cleanup();
}

