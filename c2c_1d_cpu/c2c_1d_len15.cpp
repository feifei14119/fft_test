#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <vector>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime_api.h>

#include "rocfft.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define TEST_LENGTH (15)

// butterfly radix-3 constants
#define C3QA 0.50000000000000000000000000000000F
#define C3QB 0.86602540378443864676372317075294F
// butterfly radix-5 constants
#define C5QA 0.30901699437494742410229341718282F
#define C5QB 0.95105651629515357211643933337938F
#define C5QC 0.50000000000000000000000000000000F
#define C5QD 0.58778525229247312916870595463907F
#define C5QE 0.80901699437494742410229341718282F

#define MAX_WORK_GROUP_SIZE 1024
#define TWO_PI (-6.283185307179586476925286766559)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    // printf("tableLength=%d\n", tableLength);
    for(int i = 0; i < tableLength; i++)
    {
        if(length == specRecord[i].length)
        { // if find the matched size

            size_t numPasses = specRecord[i].numPasses;
            // printf("numPasses=%d, table item %d \n", numPasses, i);
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
					printf("		j = %zu, alph = %.3f, c = %.3f, s = %.3f\n", j, j*theta, c,s);

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
void fwd_cpu_len3(float2* R0, float2* R1, float2* R2)
{
    float TR0, TR1, TR2;
    float TI0,   TI1,  TI2;

    TR0 = (*R0).x + (*R1).x + (*R2).x;
    TR1 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) + C3QB * ((*R1).y - (*R2).y);
    TR2 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) - C3QB * ((*R1).y - (*R2).y);

    TI0 = (*R0).y + (*R1).y + (*R2).y;
    TI1 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) - C3QB * ((*R1).x - (*R2).x);
    TI2 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) + C3QB * ((*R1).x - (*R2).x);

    ((*R0).x) = TR0;    ((*R0).y) = TI0;
    ((*R1).x) = TR1;    ((*R1).y) = TI1;
    ((*R2).x) = TR2;    ((*R2).y) = TI2;
}
void fwd_cpu_len5(float2* R0, float2* R1, float2* R2, float2* R3, float2* R4)
{
    float TR0, TR1, TR2, TR3, TR4;	
    float TI0,   TI1,   TI2,   TI3,   TI4;

    TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;
    TR1 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) + C5QB * ((*R1).y - (*R4).y) + C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) - C5QB * ((*R1).y - (*R4).y) - C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR2 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) - C5QB * ((*R2).y - (*R3).y) + C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));
    TR3 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) + C5QB * ((*R2).y - (*R3).y) - C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));

    TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;
    TI1 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) - C5QB * ((*R1).x - (*R4).x) - C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) + C5QB * ((*R1).x - (*R4).x) + C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI2 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) + C5QB * ((*R2).x - (*R3).x) - C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));
    TI3 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) - C5QB * ((*R2).x - (*R3).x) + C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));

    ((*R0).x) = TR0;    ((*R0).y) = TI0;
    ((*R1).x) = TR1;    ((*R1).y) = TI1;
    ((*R2).x) = TR2;    ((*R2).y) = TI2;
    ((*R3).x) = TR3;    ((*R3).y) = TI3;
    ((*R4).x) = TR4;    ((*R4).y) = TI4;
}
void fwd_cpu_len15_pass0(const float2 * twiddles, 
		unsigned int rw, unsigned int b, unsigned int me, 
		unsigned int inOffset, unsigned int outOffset, 
		float2 *bufIn, 
		float * bufOutRe, float * bufOutIm, 
		float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7, float2 *R8, float2 *R9, float2 *R10, float2 *R11, float2 *R12, float2 *R13, float2 *R14)
{
	printf("\n===================================================\n");
	printf("fwd_cpu_len15_pass0");
	printf("\n===================================================\n");
	printf("rw = %d\n", rw);
	printf("b = %d\n", b);
	printf("me = %d\n", me);
	printf("inOffset = %d\n", inOffset);
	printf("outOffset = %d\n", outOffset);	
	
	if(rw)
	{	
		(*R0)   = bufIn[inOffset + ( 0 + me*3 + 0 + 0   )]; printf("R0 id  = %d\n",  inOffset + ( 0 + me*3 + 0 + 0  ));
		(*R1)   = bufIn[inOffset + ( 0 + me*3 + 0 + 3   )]; printf("R1 id  = %d\n",  inOffset + ( 0 + me*3 + 0 + 3  ));
		(*R2)   = bufIn[inOffset + ( 0 + me*3 + 0 + 6   )]; printf("R2 id  = %d\n",  inOffset + ( 0 + me*3 + 0 + 6  ));
		(*R3)   = bufIn[inOffset + ( 0 + me*3 + 0 + 9   )]; printf("R3 id  = %d\n",  inOffset + ( 0 + me*3 + 0 + 9  ));
		(*R4)   = bufIn[inOffset + ( 0 + me*3 + 0 + 12 )]; printf("R4 id  = %d\n",  inOffset + ( 0 + me*3 + 0 + 12));
		(*R5)   = bufIn[inOffset + ( 0 + me*3 + 1 + 0   )]; printf("R5 id  = %d\n",  inOffset + ( 0 + me*3 + 1 + 0  ));
		(*R6)   = bufIn[inOffset + ( 0 + me*3 + 1 + 3   )]; printf("R6 id  = %d\n",  inOffset + ( 0 + me*3 + 1 + 3  ));
		(*R7)   = bufIn[inOffset + ( 0 + me*3 + 1 + 6   )]; printf("R7 id  = %d\n",  inOffset + ( 0 + me*3 + 1 + 6  ));
		(*R8)   = bufIn[inOffset + ( 0 + me*3 + 1 + 9   )]; printf("R8 id  = %d\n",  inOffset + ( 0 + me*3 + 1 + 9  ));
		(*R9)   = bufIn[inOffset + ( 0 + me*3 + 1 + 12 )]; printf("R9 id  = %d\n",  inOffset + ( 0 + me*3 + 1 + 12));
		(*R10) = bufIn[inOffset + ( 0 + me*3 + 2 + 0   )]; printf("R10 id = %d\n", inOffset + ( 0 + me*3 + 2 + 0  ));
		(*R11) = bufIn[inOffset + ( 0 + me*3 + 2 + 3   )]; printf("R11 id = %d\n", inOffset + ( 0 + me*3 + 2 + 3  ));
		(*R12) = bufIn[inOffset + ( 0 + me*3 + 2 + 6   )]; printf("R12 id = %d\n", inOffset + ( 0 + me*3 + 2 + 6  ));
		(*R13) = bufIn[inOffset + ( 0 + me*3 + 2 + 9   )]; printf("R13 id = %d\n", inOffset + ( 0 + me*3 + 2 + 9  ));
		(*R14) = bufIn[inOffset + ( 0 + me*3 + 2 + 12 )]; printf("R14 id = %d\n", inOffset + ( 0 + me*3 + 2 + 12));
		printf("-----------\n");
	}

	fwd_cpu_len5(R0, R1, R2, R3, R4);
	fwd_cpu_len5(R5, R6, R7, R8, R9);
	fwd_cpu_len5(R10, R11, R12, R13, R14);

	if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 0 )] = (*R0).x;   printf("real id 0  = %d\n",  outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 0 ) );
		bufOutRe[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 1 )] = (*R1).x;   printf("real id 1  = %d\n",  outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 1 ) );
		bufOutRe[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 2 )] = (*R2).x;   printf("real id 2  = %d\n",  outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 2 ) );
		bufOutRe[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 3 )] = (*R3).x;   printf("real id 3  = %d\n",  outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 3 ) );
		bufOutRe[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 4 )] = (*R4).x;   printf("real id 4  = %d\n",  outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 4 ) );
		bufOutRe[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 0 )] = (*R5).x;   printf("real id 5  = %d\n",  outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 0 ) );
		bufOutRe[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 1 )] = (*R6).x;   printf("real id 6  = %d\n",  outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 1 ) );
		bufOutRe[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 2 )] = (*R7).x;   printf("real id 7  = %d\n",  outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 2 ) );
		bufOutRe[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 3 )] = (*R8).x;   printf("real id 8  = %d\n",  outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 3 ) );
		bufOutRe[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 4 )] = (*R9).x;   printf("real id 9  = %d\n",  outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 4 ) );
		bufOutRe[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 0 )] = (*R10).x; printf("real id 10 = %d\n", outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 0 ) );
		bufOutRe[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 1 )] = (*R11).x; printf("real id 11 = %d\n", outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 1 ) );
		bufOutRe[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 2 )] = (*R12).x; printf("real id 12 = %d\n", outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 2 ) );
		bufOutRe[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 3 )] = (*R13).x; printf("real id 13 = %d\n", outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 3 ) );
		bufOutRe[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 4 )] = (*R14).x; printf("real id 14 = %d\n", outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 4 ) );
		printf("-----------\n");
		
		(*R0).x   = bufOutRe[outOffset + ( 0 + me*5 + 0 + 0 ) ];   printf("R0.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 0 + 0   )));
		(*R1).x   = bufOutRe[outOffset + ( 0 + me*5 + 0 + 5 ) ];   printf("R1.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 0 + 5   )));
		(*R2).x   = bufOutRe[outOffset + ( 0 + me*5 + 0 + 10 ) ]; printf("R2.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 0 + 10 )));
		(*R3).x   = bufOutRe[outOffset + ( 0 + me*5 + 1 + 0 ) ];   printf("R3.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 1 + 0   )));
		(*R4).x   = bufOutRe[outOffset + ( 0 + me*5 + 1 + 5 ) ];   printf("R4.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 1 + 5   )));
		(*R5).x   = bufOutRe[outOffset + ( 0 + me*5 + 1 + 10 ) ]; printf("R5.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 1 + 10 )));
		(*R6).x   = bufOutRe[outOffset + ( 0 + me*5 + 2 + 0 ) ];   printf("R6.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 2 + 0   )));
		(*R7).x   = bufOutRe[outOffset + ( 0 + me*5 + 2 + 5 ) ];   printf("R7.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 2 + 5   )));
		(*R8).x   = bufOutRe[outOffset + ( 0 + me*5 + 2 + 10 ) ]; printf("R8.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 2 + 10 )));
		(*R9).x   = bufOutRe[outOffset + ( 0 + me*5 + 3 + 0 ) ];   printf("R9.x   = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 3+ 0    )));
		(*R10).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 5 ) ];   printf("R10.x  = %d\n",  outOffset + (outOffset + ( 0 + me*5 + 3 + 5   )));
		(*R11).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 10 ) ]; printf("R11.x  = %d\n",  outOffset + (outOffset + ( 0 + me*5 + 3 + 10 )));
		(*R12).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 0 ) ];   printf("R12.x  = %d\n",  outOffset + (outOffset + ( 0 + me*5 + 4 + 0   )));
		(*R13).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 5 ) ];   printf("R13.x  = %d\n",  outOffset + (outOffset + ( 0 + me*5 + 4 + 5   )));
		(*R14).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 10 ) ]; printf("R14.x  = %d\n",  outOffset + (outOffset + ( 0 + me*5 + 4 + 10 )));
		printf("-----------\n");
		
		bufOutIm[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 0 ) ] = (*R0).y;   printf("imag id 0   = %d\n",   outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 0 ) );
		bufOutIm[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 1 ) ] = (*R1).y;   printf("imag id 1   = %d\n",   outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 1 ) );
		bufOutIm[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 2 ) ] = (*R2).y;   printf("imag id 2   = %d\n",   outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 2 ) );
		bufOutIm[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 3 ) ] = (*R3).y;   printf("imag id 3   = %d\n",   outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 3 ) );
		bufOutIm[outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 4 ) ] = (*R4).y;   printf("imag id 4   = %d\n",   outOffset + ( ((3*me + 0)/1)*5 + (3*me + 0)%1 + 4 ) );
		bufOutIm[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 0 ) ] = (*R5).y;   printf("imag id 5   = %d\n",   outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 0 ) );
		bufOutIm[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 1 ) ] = (*R6).y;   printf("imag id 6   = %d\n",   outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 1 ) );
		bufOutIm[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 2 ) ] = (*R7).y;   printf("imag id 7   = %d\n",   outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 2 ) );
		bufOutIm[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 3 ) ] = (*R8).y;   printf("imag id 8   = %d\n",   outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 3 ) );
		bufOutIm[outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 4 ) ] = (*R9).y;   printf("imag id 9   = %d\n",   outOffset + ( ((3*me + 1)/1)*5 + (3*me + 1)%1 + 4 ) );
		bufOutIm[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 0 ) ] = (*R10).y; printf("imag id 10  = %d\n",  outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 0 ) );
		bufOutIm[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 1 ) ] = (*R11).y; printf("imag id 11  = %d\n",  outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 1 ) );
		bufOutIm[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 2 ) ] = (*R12).y; printf("imag id 12  = %d\n",  outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 2 ) );
		bufOutIm[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 3 ) ] = (*R13).y; printf("imag id 13  = %d\n",  outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 3 ) );
		bufOutIm[outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 4 ) ] = (*R14).y; printf("imag id 14  = %d\n",  outOffset + ( ((3*me + 2)/1)*5 + (3*me + 2)%1 + 4 ) );
		printf("-----------\n");
		
		(*R0).y   = bufOutIm[outOffset + ( 0 + me*5 + 0 + 0 ) ];   printf("R0.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 0 + 0   )));
		(*R1).y   = bufOutIm[outOffset + ( 0 + me*5 + 0 + 5 ) ];   printf("R1.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 0 + 5   )));
		(*R2).y   = bufOutIm[outOffset + ( 0 + me*5 + 0 + 10)];   printf("R2.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 0 + 10)));
		(*R3).y   = bufOutIm[outOffset + ( 0 + me*5 + 1 + 0 ) ];   printf("R3.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 1 + 0   )));
		(*R4).y   = bufOutIm[outOffset + ( 0 + me*5 + 1 + 5 ) ];   printf("R4.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 1 + 5   )));
		(*R5).y   = bufOutIm[outOffset + ( 0 + me*5 + 1 + 10)];   printf("R5.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 1 + 10)));
		(*R6).y   = bufOutIm[outOffset + ( 0 + me*5 + 2 + 0 ) ];   printf("R6.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 2 + 0   )));
		(*R7).y   = bufOutIm[outOffset + ( 0 + me*5 + 2 + 5 ) ];   printf("R7.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 2 + 5   )));
		(*R8).y   = bufOutIm[outOffset + ( 0 + me*5 + 2 + 10)];   printf("R8.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 2 + 10)));
		(*R9).y   = bufOutIm[outOffset + ( 0 + me*5 + 3 + 0 ) ];   printf("R9.y   = %d\n",    outOffset + (outOffset + ( 0 + me*5 + 3 + 0   )));
		(*R10).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 5 ) ];   printf("R10.y  = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 3 + 5   )));
		(*R11).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 10)];   printf("R11.y  = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 3 + 10)));
		(*R12).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 0 ) ];   printf("R12.y  = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 4 + 0   )));
		(*R13).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 5 ) ];   printf("R13.y  = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 4 + 5  )));
		(*R14).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 10)];   printf("R14.y  = %d\n",   outOffset + (outOffset + ( 0 + me*5 + 4 + 10)));
		printf("-----------\n");
	}
}
void fwd_cpu_len15_pass1(const float2 *twiddles, 
		unsigned int rw, unsigned int b, unsigned int me, 
		unsigned int inOffset, unsigned int outOffset, 
		float *bufInRe, float *bufInIm, 
		float2 *bufOut, 
		float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7, float2 *R8, float2 *R9, float2 *R10, float2 *R11, float2 *R12, float2 *R13, float2 *R14)
{
	printf("\n===================================================\n");
	printf("fwd_cpu_len15_pass1");
	printf("\n===================================================\n");
	
	float2 W;
	float wx, wy, rx, ry;
	
	{
		W = twiddles[4 + 2*((5*me + 0)%5) + 0];  printf("W1 id  = %d\n", 4 + 2*((5*me + 0)%5) + 0);
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		(*R1).x = wx * rx - wy * ry;
		(*R1).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 0)%5) + 1];  printf("W2 id  = %d\n", 4 + 2*((5*me + 0)%5) + 1);
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		(*R2).x = wx * rx - wy * ry;
		(*R2).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 1)%5) + 0];  printf("W4 id  = %d\n", 4 + 2*((5*me + 1)%5) + 0);
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		(*R4).x = wx * rx - wy * ry;
		(*R4).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 1)%5) + 1];  printf("W5 id  = %d\n", 4 + 2*((5*me + 1)%5) + 1);
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		(*R5).x = wx * rx - wy * ry;
		(*R5).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 2)%5) + 0];  printf("W7 id  = %d\n", 4 + 2*((5*me + 2)%5) + 0);
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		(*R7).x = wx * rx - wy * ry;
		(*R7).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 2)%5) + 1];  printf("W8 id  = %d\n", 4 + 2*((5*me + 2)%5) + 1);
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		(*R8).x = wx * rx - wy * ry;
		(*R8).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 3)%5) + 0];  printf("W10 id = %d\n", 4 + 2*((5*me + 3)%5) + 0);
		wx = W.x; wy = W.y;
		rx = (*R10).x; ry = (*R10).y;
		(*R10).x = wx * rx - wy * ry;
		(*R10).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 3)%5) + 1];  printf("W11 id = %d\n", 4 + 2*((5*me + 3)%5) + 1);
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		(*R11).x = wx * rx - wy * ry;
		(*R11).y = wy * rx + wx * ry;
		
		W = twiddles[4 + 2*((5*me + 4)%5) + 0];  printf("W13 id = %d\n", 4 + 2*((5*me + 4)%5) + 0);
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		(*R13).x = wx * rx - wy * ry;
		(*R13).y = wy * rx + wx * ry;	
		
		W = twiddles[4 + 2*((5*me + 4)%5) + 1];  printf("W14 id = %d\n", 4 + 2*((5*me + 4)%5) + 1);
		wx = W.x; wy = W.y;
		rx = (*R14).x; ry = (*R14).y;
		(*R14).x = wx * rx - wy * ry;
		(*R14).y = wy * rx + wx * ry;	
		
		printf("-----------\n");
	}

	fwd_cpu_len3(R0, R1, R2);
	fwd_cpu_len3(R3, R4, R5);
	fwd_cpu_len3(R6, R7, R8);
	fwd_cpu_len3(R9, R10, R11);
	fwd_cpu_len3(R12, R13, R14);

	if(rw)
	{
		bufOut[outOffset + ( 5*me + 0 + 0 )]   = (*R0);   printf("R0 id  = %d\n",  outOffset + ( 5*me + 0 + 0 ));
		bufOut[outOffset + ( 5*me + 0 + 5 )]   = (*R1);   printf("R1 id  = %d\n",  outOffset + ( 5*me + 0 + 5 ));
		bufOut[outOffset + ( 5*me + 0 + 10 )] = (*R2);   printf("R2 id  = %d\n",  outOffset + ( 5*me + 0 + 10 ));
		bufOut[outOffset + ( 5*me + 1 + 0 )]   = (*R3);   printf("R3 id  = %d\n",  outOffset + ( 5*me + 1 + 0 ));
		bufOut[outOffset + ( 5*me + 1 + 5 )]   = (*R4);   printf("R4 id  = %d\n",  outOffset + ( 5*me + 1 + 5 ));
		bufOut[outOffset + ( 5*me + 1 + 10 )] = (*R5);   printf("R5 id  = %d\n",  outOffset + ( 5*me + 1 + 10 ));
		bufOut[outOffset + ( 5*me + 2 + 0 )]   = (*R6);   printf("R6 id  = %d\n",  outOffset + ( 5*me + 2 + 0 ));
		bufOut[outOffset + ( 5*me + 2 + 5 )]   = (*R7);   printf("R7 id  = %d\n",  outOffset + ( 5*me + 2 + 5 ));
		bufOut[outOffset + ( 5*me + 2 + 10 )] = (*R8);   printf("R8 id  = %d\n",  outOffset + ( 5*me + 2 + 10 ));
		bufOut[outOffset + ( 5*me + 3 + 0 )]   = (*R9);   printf("R9 id  = %d\n",  outOffset + ( 5*me + 3 + 0 ));
		bufOut[outOffset + ( 5*me + 3 + 5 )]   = (*R10); printf("R10 id = %d\n", outOffset + ( 5*me + 3 + 5 ));
		bufOut[outOffset + ( 5*me + 3 + 10 )] = (*R11); printf("R11 id = %d\n", outOffset + ( 5*me + 3 + 10 ));
		bufOut[outOffset + ( 5*me + 4 + 0 )]   = (*R12); printf("R12 id = %d\n", outOffset + ( 5*me + 4 + 0 ));
		bufOut[outOffset + ( 5*me + 4 + 5 )]   = (*R13); printf("R13 id = %d\n", outOffset + ( 5*me + 4 + 5 ));
		bufOut[outOffset + ( 5*me + 4 + 10 )] = (*R14); printf("R14 id = %d\n", outOffset + ( 5*me + 4 + 10 ));
		printf("-----------\n");
	}
}
void fwd_cpu_len15( const float2 * twiddles, const uint32_t length, float2 * gbIn, float2 * gbOut)
{
	float lds[1920];
	
	unsigned int me = 0;		//(unsigned int)hipThreadIdx_x;
	unsigned int batch = 0;	//(unsigned int)hipBlockIdx_x;
	unsigned int upper_count = 1;	
	// do signed math to guard against underflow
	unsigned int rw = (me < (upper_count  - batch*128)*1) ? 1 : 0; // 1
	unsigned int b = 0;
	size_t counter_mod = (batch*128 + (me/1)); // 0
	
	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	//if(dim == 1)
	{
		iOffset += 0;		//counter_mod*stride_in[1];
		oOffset += 0;	//counter_mod*stride_out[1];
	}
	float2 * lwbIn = gbIn + iOffset;
	float2 * lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds	
	float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14;
	fwd_cpu_len15_pass0(twiddles, rw, b, me%1, 0, (me/1)*15,    lwbIn, lds, lds,       &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14);
	fwd_cpu_len15_pass1(twiddles, rw, b, me%1, (me/1)*15, 0,    lds, lds,  lwbOut,    &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void test_cpu()
{	
	const size_t Nx = TEST_LENGTH;
	printf("twiddles_create_pr sizeof(T) = %lu, N = %lu\n",sizeof(float2), Nx);

	//std::vector<size_t> radices;
	//radices = GetRadices(Nx);
	std::vector<size_t> radices(2);
	radices[0] = 5;radices[1] = 3;

	float2 * twtc;
	TwiddleTable twTable(Nx);
	twtc = twTable.GenerateTwiddleTable(radices); // calculate twiddles and print
	
	//return;
	/////////////////////////////////////////////////////////////////////////////////////////	
    std::cout << "Input:\n";
    std::vector<float2> cx(Nx);
    std::vector<float2> cy(Nx);
    std::vector<float2> backx(cx.size());
    for(size_t i = 0; i < Nx; ++i)
    {
		cx[i] = float2(1.0f*i, -0.1f*i);
		cy[i] = float2(0,0);
		std::cout << cx[i].x << ", " << cx[i].y << "\n";
    }	
	
	fwd_cpu_len15(twtc, Nx, cx.data(), cy.data());
	
    std::cout << "\nOutput:\n";
    for(size_t i = 0; i < Nx; ++i)
		std::cout << cy[i].x << ", " << cy[i].y << "\n";
    std::cout << "\n";
}
int main(int argc, char* argv[])
{
	test_cpu();
	
	printf("\n***************************************************\n");
    std::cout << "rocFFT real/complex 3d FFT example";
	printf("\n***************************************************\n");

    // The problem size
	int iteration_times = 1000;
	double ElapsedMilliSec = 0;
	double ElapsedNanoSec = 0;
	double d_startTime;
	double d_currentTime;
	timespec startTime,stopTime;
    const size_t Nx = (argc < 2) ? TEST_LENGTH : atoi(argv[1]);
	const size_t batch = 1;
	const size_t dimension = 1;
    std::cout << "Nx: " << Nx  << std::endl;

    std::cout << "Input:\n";
    std::vector<std::complex<float>> cx(Nx*batch);
    std::vector<std::complex<float>> cy(Nx*batch);	
    for(size_t i = 0; i < Nx; ++i)
    {
		const size_t pos = i;
		cx[pos] = std::complex<float>(1.0f*i, -0.1f*i);
		cy[pos] = std::complex<float>(0,0);
    }
    for(size_t i = 0; i < Nx; ++i)
		std::cout << real(cx[i]) << ", " << imag(cx[i]) << "\n";
    std::cout << "\n";

    // Create HIP device objects:
    std::complex<float>* x = NULL;
	std::complex<float>* y = NULL;
    size_t malloc_size = cx.size() * sizeof(decltype(cx)::value_type);
    printf("hipMalloc size = %.3f(KB)\n",malloc_size / 1024.0);
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    // Length are in reverse order because rocfft is column-major.
    //const size_t lengths[3] = {Nz, Ny, Nx};
	const size_t lengths[1] = {Nx};
    rocfft_status status = rocfft_status_success;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Transformed:\n"; std::cout << "\n";
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

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info forwardinfo = NULL;
    status = rocfft_execution_info_create(&forwardinfo);    											assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    rocfft_plan_get_work_buffer_size(forward, &fbuffersize);    										assert(status == rocfft_status_success);
    void* fbuffer = NULL;
    hipMalloc(&fbuffer, fbuffersize);
    status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);    	assert(status == rocfft_status_success);

    // Execute the forward transform
    status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo);     					assert(status == rocfft_status_success);

    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    std::cout << "Output:\n";
    for(size_t i = 0; i < Nx; ++i)
		std::cout << real(cy[i]) << ", " << imag(cy[i]) << "\n";
    std::cout << "\n";
	
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
    status = rocfft_execution_info_create(&backwardinfo);    											assert(status == rocfft_status_success);
    size_t bbuffersize = 0;
    status = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);    						assert(status == rocfft_status_success);
    void* bbuffer = NULL;
    hipMalloc(&bbuffer, bbuffersize);
    status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);    assert(status == rocfft_status_success);
	
    // Execute the backward transform
    std::vector<std::complex<float>> backx(cx.size());
    status = rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);     				assert(status == rocfft_status_success);
	
    hipMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; ++i)
		std::cout << real(backx[i])/Nx << ", " << imag(backx[i])/Nx << "\n";
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

