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

#define TEST_LENGTH (10)

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
__device__ void FwdPass0_len10(unsigned int rw, unsigned int me, unsigned int inOffset, unsigned int outOffset, float2 *bufIn, float2 *bufOut,  float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7, float2 *R8, float2 *R9)
{
	if(rw)
	{
		(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*1];
		(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 1 )*1];
		(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 2 )*1];
		(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 3 )*1];
		(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 4 )*1];
		(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 5 )*1];
		(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 6 )*1];
		(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 7 )*1];
		(*R8) = bufIn[inOffset + ( 0 + me*1 + 0 + 8 )*1];
		(*R9) = bufIn[inOffset + ( 0 + me*1 + 0 + 9 )*1];
	}

	FwdRad10B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9);

	if(rw)
	{
		bufOut[outOffset + ( 1*me + 0 + 0 )*1] = (*R0);
		bufOut[outOffset + ( 1*me + 0 + 1 )*1] = (*R1);
		bufOut[outOffset + ( 1*me + 0 + 2 )*1] = (*R2);
		bufOut[outOffset + ( 1*me + 0 + 3 )*1] = (*R3);
		bufOut[outOffset + ( 1*me + 0 + 4 )*1] = (*R4);
		bufOut[outOffset + ( 1*me + 0 + 5 )*1] = (*R5);
		bufOut[outOffset + ( 1*me + 0 + 6 )*1] = (*R6);
		bufOut[outOffset + ( 1*me + 0 + 7 )*1] = (*R7);
		bufOut[outOffset + ( 1*me + 0 + 8 )*1] = (*R8);
		bufOut[outOffset + ( 1*me + 0 + 9 )*1] = (*R9);
	}
}

__device__ void fwd_len10_device (unsigned int rw, unsigned int me, unsigned int ldsOffset, float2 *lwbIn, float2 *lwbOut)
{
	float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9;
	FwdPass0_len10(rw, me, 0, 0, lwbIn, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9);
}
__global__ void fft_fwd_len10( float2 * gbIn, float2 * gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;

	unsigned int ioOffset = 0;
	unsigned int batch = 0;
	float2 *lwbIn = gbIn;
	float2 *lwbOut = gbOut;

	unsigned int upper_count = 1;
	unsigned int rw = me < upper_count  ? 1 : 0;

	unsigned int b = 0;

	size_t counter_mod = (batch*128 + (me/1));

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len10_device(rw, me%1, (me/1)*10, lwbIn, lwbOut);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void test_gpu()
{
	printf("\n***************************************************\n");
	std::cout << "gpu test";
	printf("\n***************************************************\n");
	
	const size_t Nx = TEST_LENGTH;
	const size_t batch = 1;
	const size_t dimension = 1;
	std::cout << "Nx: " << Nx  << std::endl;

	std::vector<std::complex<float>> cx(Nx*batch);
	std::vector<std::complex<float>> cy(Nx*batch);	
    std::vector<std::complex<float>> backx(cx.size());
	for(size_t i = 0; i < Nx; ++i)
	{
		cx[i] = std::complex<float>(1.0f*i, -0.1f*i);
		cy[i] = std::complex<float>(0,0);
	}
	
	// Create HIP device objects:
	float2 * x = NULL;
	float2 * y = NULL;
	hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
	hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

	hipLaunchKernelGGL(fft_fwd_len10, dim3(1, 1), dim3(1, 1), 
					0, 0, 
					x,y);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
	if(1)
	{
		std::cout << "Output:\n";
		for(size_t i = 0; i < Nx; ++i)
			std::cout << real(cy[i]) << ", " << imag(cy[i]) << "\n";
	}
	std::cout << "\n";	
}

int main(int argc, char* argv[])
{	
	test_gpu(); 
	
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
	std::complex<float>* x = NULL;
	std::complex<float>* y = NULL;
	size_t malloc_size = cx.size() * sizeof(decltype(cx)::value_type);
	printf("hipMalloc size = %.3f(KB)\n",malloc_size / 1024.0);
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

	hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
	if(1)
	{
		std::cout << "Output:\n";
		for(size_t i = 0; i < Nx; ++i)
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
	return 0;
	std::cout << "\n";	
	
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

