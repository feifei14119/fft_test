
/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<complex>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>


////////////////////////////////////////////////////////////////////////////////
#define T float2
#define T2 float2
void runTest(int argc, char **argv) 
{
	printf("[simpleCUFFT] is starting...\n");

	findCudaDevice(argc, (const char **)argv);
  
    const size_t Nx      = (argc < 2) ? 100 : atoi(argv[1]);
	const size_t Ny      = 1;
	const size_t Nz      = 1;

    std::vector<T> cx(Nx * Ny * Nz);
    std::vector<T> backx(cx.size());
    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nz; ++k)
            {
                const size_t pos = i * Ny * Nz + j * Nz + k;
                cx[pos].x = i + j + k;
                cx[pos].y = -0.1f * (i + j + k);
            }
        }
    }

    // Output buffer
    std::vector<std::complex<T>> cy(Nx * Ny * Nz);
  
    // Create HIP device objects:
    T* x = NULL;
    T2 * y = NULL;
    cudaMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	cudaMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    cudaMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), cudaMemcpyHostToDevice);


    // Create plans
	cufftHandle plan;
	cufftHandle plan2;
	checkCudaErrors(cufftPlan1d(&plan, Nx*Ny*Nz, CUFFT_C2C, 1));		checkCudaErrors(cufftExecC2C(plan, x, reinterpret_cast<cufftComplex *>(y), CUFFT_FORWARD));
	checkCudaErrors(cufftPlan1d(&plan2, Nx*Ny*Nz, CUFFT_C2C, 1));		checkCudaErrors(cufftExecC2C(plan2, reinterpret_cast<cufftComplex *>(y), x, CUFFT_INVERSE));   
    cudaMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), cudaMemcpyDeviceToHost);
  

    double       error = 0.0f;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                double diffx = std::abs(backx[i].x / (Nx * Ny * Nz) - cx[i].x);
                double diffy = std::abs(backx[i].y / (Nx * Ny * Nz) - cx[i].y);
				double diff = diffx + diffy;
                if(diff > error)
                    error = diff;
            }
        }
    }
    std::cout << "Maximum error: " << error << "\n";
	
	if(1)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			cufftExecC2C(plan, x, reinterpret_cast<cufftComplex *>(y), CUFFT_FORWARD);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
	
	//delete cx;
	//free(backx);
	//checkCudaErrors(cudaFree(x));
	//checkCudaErrors(cudaFree(y));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(plan2));
	
}

int main(int argc, char **argv) { runTest(argc, argv); }
