#include <stdlib.h>
#include <tuple>
#include <cassert>
#include <complex>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime_api.h>


#include "hip/hip_runtime.h"
#include "rocfft.h"
#include "common.h"

#define FFT_LEN 2039 // 2039,31
#define FFT_BATCH 1

size_t Nx;
size_t Batch;
uint32_t IsProf;
uint32_t Dimension;

inline size_t CeilPo2(size_t n)
{
	size_t v = 1, t = 0;
	while(v < n)
	{
		v <<= 1;
		t++;
	}

	return t;
}
inline size_t DivRoundingUp(size_t a, size_t b)
{
    return (a + (b - 1)) / b;
}

inline size_t FindBlue(size_t len)
{
    size_t p = 1;
    while(p < len)
        p <<= 1;
    return 2 * p;
}
class TwiddleTableLarge
{
    size_t N; // length
    size_t X, Y;
    size_t tableSize;
    float2* wc; // cosine, sine arrays
	#define TWIDDLE_DEE 8

public:
    TwiddleTableLarge(size_t length) : N(length)
    {
        X         = size_t(1) << TWIDDLE_DEE; // 2*8 = 256
        Y         = DivRoundingUp(CeilPo2(N), TWIDDLE_DEE);
        tableSize = X * Y;

        // Allocate memory for the tables
        wc = new float2[tableSize];
		printf("X = %zu\n",X);
		printf("Y = %zu\n",Y);
		printf("tableSize = %zu\n",tableSize);
    }

    ~TwiddleTableLarge(){ delete[] wc; }

    std::tuple<size_t, float2*> GenerateTwiddleTable()
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt  = 0;
        double phi = TWO_PI / double(N);
		printf("N = %zu, phase = %.3f\n", N, phi);
        for(size_t iY = 0; iY < Y; ++iY)
        {
            size_t i = size_t(1) << (iY * TWIDDLE_DEE);
			printf("------- i = %zu, iY = %zu ---------\n", i, iY);
            for(size_t iX = 0; iX < X; ++iX)
            {
                size_t j = i * iX;

                double c = cos(phi * j);
                double s = sin(phi * j);
				//printf("iX = %zu, j = %zu:  c = %.3f, s = %.3f\n", iX, j, c, s);

                // if (fabs(c) < 1.0E-12)	c = 0.0;
                // if (fabs(s) < 1.0E-12)	s = 0.0;

                wc[nt].x = c;
                wc[nt].y = s;
                nt++;
            }
        } // end of for

        return std::make_tuple(tableSize, wc);
    }
};

__global__ void chirp_device(const size_t N, 
							const size_t M, 
							float2* output, 
							float2* twiddles_large, 
							const int twl, 
							const int dir)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    float2 val = lib_make_vector2<float2>(0, 0);

    if(twl == 1)		val = TWLstep1(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 2)	val = TWLstep2(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 3)	val = TWLstep3(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 4)	val = TWLstep4(twiddles_large, (tx * tx) % (2 * N));

    val.y *= (real_type_t<float2>)(dir);

    if(tx == 0)
    {
        output[tx]     = val;
        output[tx + M] = val;
    }
    else if(tx < N)
    {
        output[tx]     = val;
        output[tx + M] = val;

        output[M - tx]     = val;
        output[M - tx + M] = val;
    }
    else if(tx <= (M - N))
    {
        output[tx]     = lib_make_vector2<float2>(0, 0);
        output[tx + M] = lib_make_vector2<float2>(0, 0);
    }
}
__global__ void mul_device(const size_t  numof,
                           const size_t  totalWI,
                           const size_t  N,
                           const size_t  M,
                           const float2* input,
                           float2*       output,
                           const size_t  dim,
                           const size_t* lengths,
                           const size_t* stride_in,   
                           const size_t* stride_out, 
                           const int     scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    /*for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }*/
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx          = tx % numof;
    size_t iIdx = tx * stride_in[0];
    size_t oIdx = tx * stride_out[0];
    if(scheme == 0)
    {
        output += oOffset;

        float2 out     = output[oIdx];
        output[oIdx].x = input[iIdx].x * out.x - input[iIdx].y * out.y;
        output[oIdx].y = input[iIdx].x * out.y + input[iIdx].y * out.x;
    }
    else if(scheme == 1)
    {
        float2* chirp = output;

        input += iOffset;

        output += M;
        output += oOffset;

        if(tx < N)
        {
            output[oIdx].x = input[iIdx].x * chirp[tx].x + input[iIdx].y * chirp[tx].y;
            output[oIdx].y = -input[iIdx].x * chirp[tx].y + input[iIdx].y * chirp[tx].x;
        }
        else
        {
            output[oIdx] = lib_make_vector2<float2>(0, 0);
        }
    }
    else if(scheme == 2)
    {
        const float2* chirp = input;

        input += 2 * M;
        input += iOffset;

        output += oOffset;

        real_type_t<float2> MI = 1.0 / (real_type_t<float2>)M;
        output[oIdx].x    = MI * (input[iIdx].x * chirp[tx].x + input[iIdx].y * chirp[tx].y);
        output[oIdx].y    = MI * (-input[iIdx].x * chirp[tx].y + input[iIdx].y * chirp[tx].x);
    }
}
void Blustein()
{
	printf("\n***************************************************\n");
	std::cout << "rocFFT complex 1d FFT example";
	printf("\n***************************************************\n");

    size_t lengthBlue = FindBlue(Nx);
	size_t large1D    = 2 * Nx;
	printf("Nx = %zu, lengthBlue = %zu, large1D = %zu\n", Nx, lengthBlue, large1D);

	printf("========== GenerateTwiddleTable -==========\n");
    float2 * twts; // device side // gpubuf.data()
    float2 * twtc; // host side
    size_t TableLen = 0; // table size
	TwiddleTableLarge twTable(Nx); // does not generate radices
	std::tie(TableLen, twtc) = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

	printf("TableLen = %zu\n", TableLen);
	hipMalloc(&twts, TableLen * sizeof(float));
	hipMemcpy(twts, twtc, TableLen * sizeof(float), hipMemcpyHostToDevice);

	printf("========== chirp_device ==========\n");
    int twl = 0;
    if(large1D > (size_t)256 * 256 * 256 * 256)	printf("large1D twiddle size too large error");
    else if(large1D > (size_t)256 * 256 * 256)	twl = 4;
    else if(large1D > (size_t)256 * 256)		twl = 3;
    else if(large1D > (size_t)256)        		twl = 2;
    else        								twl = 1;

    size_t N = Nx; // (DeviceCallIn*)data_p->node->length[0];
    size_t M = lengthBlue;
    dim3 grid((M - N) / 64 + 1); 
    dim3 threads(64);
	printf("N = %zu, M = %zu, twl = %d\n", N, M, twl);
	printf("grid = %zu, group = %d\n", ((M - N) / 64 + 1), 64);
	
	float2 * dB;
	hipMalloc(&dB, TableLen * sizeof(float2));
    hipLaunchKernelGGL(chirp_device,
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       M,
                       dB,
                       twts,
                       twl,
                       1);
					   
	float2 * hB;
	hB = (float2 *)malloc(TableLen * sizeof(float2));
	hipMemcpy(hB, dB, TableLen * sizeof(float2), hipMemcpyDeviceToHost);
	for(uint32_t i = 0; i < TableLen; i++)
	{
		//printf("%d = <%.3e, %.3e>\n", i, hB[i].x, hB[i].y);
	}

	printf("========== CS_KERNEL_FFT_MUL ==========\n");
	{
		size_t N = Nx;
		size_t M = lengthBlue;
		int scheme = 0;
		size_t numof = M;
    	size_t count = Batch;
    	//for(size_t i = 1; i < Dimension; i++)
        //	count *= length[i];
    	count *= numof;
		dim3 grid((count - 1) / 64 + 1);
		dim3 threads(64);
		printf("N = %zu, M = %zu, numof = %zu, count = %zu\n", N, M, numof, count);
		printf("grid = %zu, group = %d\n", ((count - 1) / 64 + 1), 64);	
		
		size_t h_len[2];h_len[0] = Nx; h_len[1] = 1;
		size_t h_strIn[2];h_strIn[0] = 1;h_strIn[1] = Nx;
		size_t h_strOut[2];h_strOut[0] = 1;h_strOut[1] = Nx;		
		size_t * d_len;hipMalloc(&d_len, 2 * sizeof(size_t));
		size_t * d_strIn;hipMalloc(&d_strIn, 2 * sizeof(size_t));
		size_t * d_strOut;hipMalloc(&d_strOut, 2 * sizeof(size_t));
		hipMemcpy(d_len, h_len, 2 * sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_strIn, h_strIn, 2 * sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(d_strOut, h_strOut, 2 * sizeof(float), hipMemcpyHostToDevice);

		float2 * h_in = (float2 *)malloc(1024 * sizeof(float2));
		float2 * d_in;	hipMalloc(&d_in, 1024 * sizeof(float2));
		float2 * h_out = (float2 *)malloc(1024 * sizeof(float2));
		float2 * d_out;	hipMalloc(&d_out, 1024 * sizeof(float2));
		hipLaunchKernelGGL(mul_device,
							grid,
							threads,
							0,
							0,
							numof,
							count,
							N,
							M,
							(const float2*)d_in,
							(float2*)d_out,
							Dimension,
							d_len,
							d_strIn,
							d_strOut,
							scheme);
		hipMemcpy(h_out, d_out, 1024 * sizeof(float2), hipMemcpyDeviceToHost);
	}
	printf("========== CS_KERNEL_PAD_MUL ==========\n");
	{
		size_t N = lengthBlue;
		size_t M = lengthBlue;
		int scheme = 1;
		size_t numof = M;
    	size_t count = Batch;
    	count *= numof;
		dim3 grid((count - 1) / 64 + 1);
		dim3 threads(64);
		printf("N = %zu, M = %zu, numof = %zu, count = %zu\n", N, M, numof, count);
		printf("grid = %zu, group = %d\n", ((count - 1) / 64 + 1), 64);
	}
	printf("========== CS_KERNEL_RES_MUL ==========\n");
	{
		size_t N = Nx;
		size_t M = lengthBlue;
		int scheme = 2;
		size_t numof = N;
    	size_t count = Batch;
    	count *= numof;
		dim3 grid((count - 1) / 64 + 1);
		dim3 threads(64);
		printf("N = %zu, M = %zu, numof = %zu, count = %zu\n", N, M, numof, count);
		printf("grid = %zu, group = %d\n", ((count - 1) / 64 + 1), 64);
	}

	//hipStreamCreate(&stream)
	//hipStreamSynchronize(stream)
	//hipStreamDestroy(stream)
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{	
	printf("\n***************************************************\n");
	std::cout << "rocFFT complex 1d FFT example";
	printf("\n***************************************************\n");

    Nx = (argc < 2) ? FFT_LEN : atoi(argv[1]);
    Batch = (argc < 3) ? FFT_BATCH : atoi(argv[2]);
    IsProf = (argc < 4) ? 0 : atoi(argv[3]);
	Dimension = 1;
	printf("N = %zu, Batch = %zu, IsProf = %d\n", Nx, Batch, IsProf);

	std::vector<std::complex<float>> cx(Nx*Batch);
	std::vector<std::complex<float>> cy(Nx*Batch);	
    std::vector<std::complex<float>> backx(cx.size());
    for(size_t i = 0; i < Batch; ++i)
    {
		for(size_t k = 0; k < Nx; ++k)
		{
			size_t pos = i * Nx + k;
			cx[i] = std::complex<float>(1.0f*i, -0.1f*i);
			cy[i] = std::complex<float>(0,0);
		}
	}
	//std::cout << "Input:\n";
	//for(size_t i = 0; i < Nx; ++i)
	//	std::cout << real(cx[i]) << ", " << imag(cx[i]) << "\n";
	std::cout << "\n";

	// Create HIP device objects:
	std::complex<float>* x = NULL;
	std::complex<float>* y = NULL;
	hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
	hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

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
                                Dimension,
                                lengths,
                                Batch,
                                NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info forwardinfo = NULL;
	status = rocfft_execution_info_create(&forwardinfo); 
	//hipStream_t stream;
	//hipError_t err = hipStreamCreate(&stream);														assert(err == 0);
	//status = rocfft_execution_info_set_stream(forwardinfo, stream);									assert(status == rocfft_status_success);
	size_t fbuffersize = 0;
	status = rocfft_plan_get_work_buffer_size(forward, &fbuffersize);						assert(status == rocfft_status_success);
	printf("buffer size = %zu (float)\n", fbuffersize / sizeof(float));
	void* fbuffer = NULL;	hipMalloc(&fbuffer, fbuffersize);
	status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);    	assert(status == rocfft_status_success);
	printf("x = 0x%08X\n", x);
	printf("fbuffer = 0x%08X\n", fbuffer);
	printf("y = 0x%08X\n", y);
	// Execute the forward transform
	status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo); 					assert(status == rocfft_status_success);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
	//std::cout << "Output:\n";
	//for(size_t i = 0; i < Nx; ++i)
	//	std::cout << real(cy[i]) << ", " << imag(cy[i]) << "\n";

	//Blustein();
	if(1)
	{

		int iteration_times = 1000;
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		double d_startTime;
		double d_currentTime;
		timespec startTime,stopTime;
		
		ElapsedMilliSec = 0;
		ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("Forward elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
			
	}
	return 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed Inverse:\n";
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
	
	// Execute the backward transform
	status = rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);     			assert(status == rocfft_status_success);
	
	hipMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
	//std::cout << "Output:\n";
	//for(size_t i = 0; i < Nx; ++i)
	//	std::cout << real(backx[i])/Nx << ", " << imag(backx[i])/Nx << "\n";
	
	float error = 0.0f;
    for(size_t i = 0; i < Batch; i++)
    {
		for(size_t k = 0; k < Nx; k++)
		{
			const size_t pos = i * Nx + k;
			double diffx = std::abs(real(backx[pos]) / (Nx) - real(cx[pos]));
			double diffy = std::abs(imag(backx[pos]) / (Nx) - imag(cx[pos]));
			double diff = diffx + diffy;
			
			if(diff > error)
				error = diff;
		}
	}
	std::cout << "Maximum error: " << error << "\n";
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if(IsProf > 0)
	{
		int iteration_times = 1000;
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		double d_startTime;
		double d_currentTime;
		timespec startTime,stopTime;
		
		ElapsedMilliSec = 0;
		ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("Forward elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
			
		ElapsedMilliSec = 0;
		ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);
		hipDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("Inverse elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
	std::cout << "\n";
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	hipFree(x);
	hipFree(y);
	hipFree(fbuffer);
	hipFree(bbuffer);

	// Destroy plans
	rocfft_plan_destroy(forward);
	rocfft_plan_destroy(backward);

	rocfft_cleanup();
}

