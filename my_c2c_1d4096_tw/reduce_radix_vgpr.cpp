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

//#include "./common.h"
//#include "./butterfly_constant.h"
//#include "./rocfft_butterfly_template.h"
#include "./rocfft_kernel_4096.h"

#define TEST_LENGTH (4096)
#define BATCH_SIZE  (1)

size_t Nx;
size_t Batch;
uint32_t IsProf;
uint32_t Dimension;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_WORK_GROUP_SIZE 1024
#define TWO_PI      (-6.283185307179586476925286766559)
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
    double2 * dwc;

public:
    TwiddleTable(size_t length) : N(length)    {  wc = new float2[N];  dwc = new double2[N];  }

    ~TwiddleTable()    { delete[] wc;   delete[] dwc;  }

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
			
			//printf("radix = %zu, L = %zu\n", radix, L);

            // Twiddle factors
            for(size_t k = 0; k < (L / radix); k++)
            {
                double theta = TWO_PI * (k) / (L);
				//printf("	[k/L] = [%zu/%zu], theta = %.3f\n", k, L, theta);
				
				{
					size_t j = 0;
					double c = cos((j)*theta);
					double s = sin ((j)*theta);
					//printf("		j = %zu, alph = %.3f, c = %.3f, s = %.3f\n", j, j*theta, c,s);
				}
					
                for(size_t j = 1; j < radix; j++)
                {
                    double c = cos((j)*theta);
                    double s = sin ((j)*theta);
					//printf("[%zu]		j = %zu, alph = %.3f, c = %.3f, s = %.3f\n", idxcnt, j, j*theta, c,s);
					idxcnt++;

                    wc[nt].x = c; wc[nt].y = s;	
                    dwc[nt].x = c; dwc[nt].y = s;				
                    nt++;
                }
            }
        }

		printf("\n");
        return wc;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_gpu(float2 * x, float2 * y, std::vector<std::complex<float>> * cy)
{		
	printf("twiddles_create_pr \n");
	std::vector<size_t> radices;
	radices = GetRadices(Nx);
	TwiddleTable twTable(Nx);

    // float tw
	float2 * twtc;
	float2 * dtw = NULL;
	twtc = twTable.GenerateTwiddleTable(radices);
	hipMalloc(&dtw, Nx * sizeof(float2));
	hipMemcpy(dtw, twtc, Nx * sizeof(float2), hipMemcpyHostToDevice);
    // double tw
    double2 * dtwtc;
	double2 * ddtw = NULL;
    dtwtc = twTable.dwc;
	hipMalloc(&ddtw, Nx * sizeof(double2));
	hipMemcpy(ddtw, dtwtc, Nx * sizeof(double2), hipMemcpyHostToDevice);
    // debug
    const int dbg_len = 4096;
	double2 * hdebug = new double2[dbg_len];
	double2 * ddebug = NULL;
	hipMalloc(&ddebug, dbg_len * sizeof(double2));
    hipMemset(&ddebug, 0, dbg_len * sizeof(double2));

	const size_t lengths[1] = {Nx};
	const size_t stride_in[2] = {1, 4096};
	const size_t stride_out[2] = {1, 4096};
	size_t * dlen = NULL;
	size_t * dstrin = NULL;
	size_t * dstrout = NULL;
	hipMalloc(&dlen, 1 * sizeof(size_t));
	hipMalloc(&dstrin, 2 * sizeof(size_t));
	hipMalloc(&dstrout, 2 * sizeof(size_t));
	hipMemcpy(dlen, lengths, 1 * sizeof(size_t), hipMemcpyHostToDevice);
	hipMemcpy(dstrin, stride_in, 2 * sizeof(size_t), hipMemcpyHostToDevice);
	hipMemcpy(dstrout, stride_out, 2 * sizeof(size_t), hipMemcpyHostToDevice);

	std::cout << "gpu test\n";
	dim3 gp_sz = dim3(256);
	dim3 gp_num = dim3(Batch);
	hipLaunchKernelGGL(my_fft_fwd_op_len4096, gp_num, gp_sz, 
					0, 0, 
					dtw, ddtw, 1, dlen, dstrin, dstrout, Batch, x, y, ddebug);
	hipMemcpy(cy->data(), y, cy->size() * sizeof(float2), hipMemcpyDeviceToHost);

	hipMemcpy(hdebug, ddebug, dbg_len * sizeof(double2), hipMemcpyDeviceToHost);
    for(uint32_t i = 0; i < 250; i++)
    {
        //if((hdebug[i].x != 0)||(hdebug[i].y != 0))
        //printf("[%03d] = <%.5e, %.5e>\n", i, (double)(hdebug[i].x), (double)(hdebug[i].y));
    }
					
	if(IsProf)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			hipLaunchKernelGGL(my_fft_fwd_op_len4096, gp_num, gp_sz, 0, 0, dtw, ddtw, 1, dlen, dstrin, dstrout, Batch, x, y, ddebug);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}

	//exit(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fftw3.h>
void test_fftw(std::vector<std::complex<float>> * x, std::vector<std::complex<float>> * y)
{
	printf("fftw \n");
    fftwf_complex *in, *out;
    fftwf_plan p;
    
    in = (fftwf_complex*) (x->data());
    out = (fftwf_complex*) (y->data());

    p = fftwf_plan_dft_1d(Nx, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    
    fftwf_destroy_plan(p);
}

void test_fftwd(std::vector<std::complex<float>> * x, std::vector<std::complex<float>> * y)
{
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);

    double * pp = (double*)in;
    for(uint32_t i = 0; i < Nx; i++)
    {
        pp[i*2+0] = 1.0*i;
        pp[i*2+1] = -0.1f*i;
    }

    p = fftw_plan_dft_1d(Nx, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p); // 执行变换
    
    double *ip = (double*)out;
    float *op = (float*)(y->data());
    for(uint32_t i = 0; i < Nx; i++)
    {
        op[i*2+0] = (float)ip[i*2+0];
        op[i*2+1] = (float)ip[i*2+1];
    }

    fftw_destroy_plan(p);
    fftw_free(in); 
    fftw_free(out);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{		
	printf("\n***************************************************\n");
	std::cout << "rocFFT complex 1d FFT example";
	printf("\n***************************************************\n");

	// The problem size
	Nx = (argc < 2) ? TEST_LENGTH : atoi(argv[1]);
	Batch = (argc < 3) ? BATCH_SIZE : atoi(argv[2]);
	Nx = TEST_LENGTH;
	Batch = BATCH_SIZE;
	IsProf = (argc < 4) ? 0 : atoi(argv[3]);
	Dimension = 1;
	printf("Nx = %zu, Batch = %zu, IsProf = %d\n", Nx, Batch, IsProf);

	std::vector<std::complex<float>> cx(Nx*Batch);
	std::vector<std::complex<float>> cy(Nx*Batch);
	std::vector<std::complex<float>> mycy(Nx*Batch);
	std::vector<std::complex<float>> fftwcy(Nx*Batch);
    std::vector<std::complex<float>> backx(cx.size());
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			cx[pos] = std::complex<float>(1.0f*(i+k), -0.1f*(i+k));
			cy[pos] = std::complex<float>(0,0);
		}
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
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create plans
	rocfft_plan forward = NULL;
	status = rocfft_plan_create(&forward,
                                rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                Dimension,
                                lengths,
                                Batch,
                                NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info forwardinfo = NULL;
	status = rocfft_execution_info_create(&forwardinfo); 									assert(status == rocfft_status_success);
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(forward, &fbuffersize);								assert(status == rocfft_status_success);
	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);    	assert(status == rocfft_status_success);
	status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo); 					assert(status == rocfft_status_success);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(float2), hipMemcpyDeviceToHost);
	
	if(IsProf)
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
//    return 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	test_gpu(x, myy, &mycy);
    	
	float myerr = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			float diffx = std::abs(real(mycy[pos]) - real(cy[pos]));
			float diffy = std::abs(imag(mycy[pos]) - imag(cy[pos]));
			float diff = diffx + diffy;
			if(diff > myerr)
				myerr = diff;
		}
	}
	std::cout << "Maximum error: " << myerr << "\n";
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	test_fftwd(&cx, &fftwcy);
	float fftwerr = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			float diffx = std::abs(real(cy[pos]) - real(fftwcy[pos]));
			float diffy = std::abs(imag(cy[pos]) - imag(fftwcy[pos]));
			float diff = diffx + diffy;
			if(diff > fftwerr)
				fftwerr = diff;
		}
	}
	std::cout << "Maximum error: " << fftwerr << "\n";

    float ffterr;
    ffterr = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			float diffx = std::abs(real(cy[pos]) - real(fftwcy[pos]));
			float diffy = std::abs(imag(cy[pos]) - imag(fftwcy[pos]));
            ffterr += (diffx + diffy);
		}
	}
	std::cout << "roc-fftw sum error: " << ffterr << "\n";
    ffterr = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			float diffx = std::abs(real(mycy[pos]) - real(fftwcy[pos]));
			float diffy = std::abs(imag(mycy[pos]) - imag(fftwcy[pos]));
            ffterr += (diffx + diffy);
		}
	}
	std::cout << "fftw-my sum error: " << ffterr << "\n";
    ffterr = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			float diffx = std::abs(real(cy[pos]) - real(mycy[pos]));
			float diffy = std::abs(imag(cy[pos]) - imag(mycy[pos]));
            ffterr += (diffx + diffy);
		}
	}
	std::cout << "roc-my sum error: " << ffterr << "\n";
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed back:\n";
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create plans
	rocfft_plan backward = NULL;
	status = rocfft_plan_create(&backward,
                                rocfft_placement_notinplace,
                                rocfft_transform_type_complex_inverse,
                                rocfft_precision_single,
                                Dimension,
                                lengths,
                                Batch,
                                NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info backwardinfo = NULL;
	status = rocfft_execution_info_create(&backwardinfo);  									assert(status == rocfft_status_success);
	size_t bbuffersize = 0;
	status = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);    					assert(status == rocfft_status_success);
	void* bbuffer = NULL;
	hipMalloc(&bbuffer, bbuffersize);
	status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);    	assert(status == rocfft_status_success);
	status = rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);     			assert(status == rocfft_status_success);
	
	hipMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
	
	if(IsProf)
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
	
	std::cout << "\n";	
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	float error = 0.0f;
	for(size_t i = 0; i < Batch; i++)
	{
		for(size_t k = 0; k < Nx; k++)
		{
			size_t pos = i * Nx + k;
			double diffr = std::abs(real(backx[pos]) / Nx - real(cx[pos]));
			double diffi = std::abs(imag(backx[pos]) / Nx - imag(cx[pos]));
			double diff = diffr + diffi;
			if(diff > error)
				error = diff;
		}
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

