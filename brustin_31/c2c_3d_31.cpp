#include <cassert>
#include <complex>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime_api.h>

#include "rocfft.h"

#define FFT_LEN 31
#define FFT_BATCH 240

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{	
	printf("\n***************************************************\n");
	std::cout << "rocFFT complex 1d FFT example";
	printf("\n***************************************************\n");

    const size_t Nx = (argc < 2) ? FFT_LEN : atoi(argv[1]);
    const size_t Ny = Nx;
    const size_t Nz = Nx;
    const size_t Batch = (argc < 3) ? FFT_BATCH : atoi(argv[2]);
    const unsigned int IsProf = (argc < 4) ? 0 : atoi(argv[3]);
	const size_t Dimension = 3;
	printf("N = %zu, Batch = %zu, IsProf = %d\n", Nx, Ny, Nz, Batch, IsProf);

	std::vector<std::complex<float>> cx(Nx*Ny*Nz*Batch);
	std::vector<std::complex<float>> cy(Nx*Ny*Nz*Batch);	
    std::vector<std::complex<float>> backx(cx.size());
    for(size_t k3 = 0; k3 < Batch; ++k3)
    {
		for(size_t k2 = 0; k2 < Nz; ++k2)
		{
			for(size_t k1 = 0; k1 < Ny; ++k1)
			{
				for(size_t k = 0; k < Nx; ++k)
				{
					const size_t pos = (Nx*Ny*Nz)*k3 + (Nx*Ny)*k2 + Nx*k1 + k;
					cx[pos] = std::complex<float>(1.0f*k, -0.1f*k1);
					cy[pos] = std::complex<float>(k3*1.0f,-0.1f*k2);
				}
			}
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
	const size_t lengths[3] = {Nx,Ny,Nz};
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
	status = rocfft_execution_info_create(&forwardinfo); 									assert(status == rocfft_status_success);
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(forward, &fbuffersize);								assert(status == rocfft_status_success);
	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);    	assert(status == rocfft_status_success);

	// Execute the forward transform
	status = rocfft_execute(forward, (void**)&x, (void**)&y, forwardinfo); 					assert(status == rocfft_status_success);

	hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
	//std::cout << "Output:\n";
	//for(size_t i = 0; i < Nx; ++i)
	//	std::cout << real(cy[i]) << ", " << imag(cy[i]) << "\n";

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

