#include <cassert>
#include <complex>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime_api.h>

#include "rocfft.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define TEST_LENGTH (100)

int main(int argc, char* argv[])
{
	printf("\n***************************************************\n");
	std::cout << "rocFFT real/complex 3d FFT example";
	printf("\n***************************************************\n");

	// The problem size
	const size_t batch = 1;
	const size_t dimension = 3;
	const size_t Nx = (argc < 2) ? 100 : atoi(argv[1]);
	const size_t Ny = (argc < 3) ? 100 : atoi(argv[2]);
	const size_t Nz = (argc < 4) ? 100 : atoi(argv[3]);
	std::cout << "Nx: " << Nx << "\tNy: " << Ny << "\tNz: " << Nz << std::endl;

	std::vector<float> cx(Nx * Ny * Nz);
	std::vector<std::complex<float>> cy(Nx * Ny * Nz);
	std::vector<float> backx(cx.size());
	for(size_t i = 0; i < Nx; ++i)
	{
		for(size_t j = 0; j < Ny; ++j)
		{
			for(size_t k = 0; k < Nz; ++k)
			{
				const size_t pos = i * Ny * Nz + j * Nz + k;
				cx[pos] = i + j + k;
			}
		}
	}
	
	/*std::cout << "Input:\n";
	for(size_t i = 0; i < Nx; ++i)
	{
		for(size_t j = 0; j < Ny; ++j)
		{
			for(size_t k = 0; k < Nz; ++k)
			{
				const size_t pos = i * Ny * Nz + j * Nz + k;
				std::cout << cx[pos] << "  ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	// Create HIP device objects:
	float* x = NULL;
	float2* y = NULL;
	size_t malloc_size = cx.size() * sizeof(decltype(cx)::value_type);
	printf("hipMalloc size = %.3f(KB)\n",malloc_size / 1024.0);
	hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
	hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

	// Length are in reverse order because rocfft is column-major.
	const size_t lengths[3] = {Nz, Ny, Nx};
	rocfft_status status = rocfft_status_success;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed:\n";
	// Create plans
	rocfft_plan forward = NULL;
	status = rocfft_plan_create(&forward,
								rocfft_placement_notinplace,
								rocfft_transform_type_real_forward,
								rocfft_precision_single,
								dimension,
								lengths,
								batch,
								NULL);
	assert(status == rocfft_status_success);

	// The real-to-complex transform uses work memory, which is passed
	// via a rocfft_execution_info struct.
	rocfft_execution_info forwardinfo = NULL;
	status = rocfft_execution_info_create(&forwardinfo);								assert(status == rocfft_status_success);
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(forward, &fbuffersize);							assert(status == rocfft_status_success);
	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);	assert(status == rocfft_status_success);

	// Execute the forward transform
	status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo);				assert(status == rocfft_status_success);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
	
	/*std::cout << "Output:\n";
	for(size_t i = 0; i < Nx; i++)
	{
		for(size_t j = 0; j < Ny; ++j)
		{
			for(size_t k = 0; k < Nzcomplex; k++)
			{
				const size_t pos = (i * Ny + j) * Nzcomplex + k;
				std::cout << cy[pos] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/
	
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
	//return 0;
	std::cout << "\n";
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Transformed back:\n";
	// Create plans
	rocfft_plan backward = NULL;
	status = rocfft_plan_create(&backward,
								rocfft_placement_notinplace,
								rocfft_transform_type_real_inverse,
								rocfft_precision_single,
								dimension,
								lengths,
								batch,
								NULL);
	assert(status == rocfft_status_success);

	rocfft_execution_info backwardinfo = NULL;
	status = rocfft_execution_info_create(&backwardinfo);								assert(status == rocfft_status_success);
	size_t bbuffersize = 0;
	status = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);					assert(status == rocfft_status_success);
	void* bbuffer = NULL;
	hipMalloc(&bbuffer, bbuffersize);
	status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);	assert(status == rocfft_status_success);

	// Execute the backward transform
	status = rocfft_execute(backward, (void**)&y, (void**)&x, backwardinfo);			assert(status == rocfft_status_success);

	hipMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
	
	/*std::cout << "Output:\n";
	for(size_t i = 0; i < Nx; ++i)
	{
		for(size_t j = 0; j < Ny; ++j)
		{
			for(size_t k = 0; k < Nz; ++k)
			{
				const size_t pos = i * Ny * Nz + j * Nz + k;
				std::cout << backx[pos] << "  ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/
	
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
	const float overN = 1.0f / (Nx * Ny * Nz);
	float       error = 0.0f;
	for(size_t i = 0; i < Nx; i++)
	{
		for(size_t j = 0; j < Ny; j++)
		{
			for(size_t k = 0; k < Nz; k++)
			{
				float diff = std::abs(backx[i] * overN - cx[i]);
				if(diff > error)
				{
					error = diff;
				}
			}
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
