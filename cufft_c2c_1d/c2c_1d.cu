
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
void runTest(int argc, char **argv) 
{
	printf("[simpleCUFFT] is starting...\n");

	findCudaDevice(argc, (const char **)argv);
  
    const size_t Nx = (argc < 2) ? 100 : atoi(argv[1]);
    const size_t Batch = (argc < 3) ? 100 : atoi(argv[2]);
    const unsigned int IsProf = (argc < 4) ? 0 : atoi(argv[3]);
	printf("N = %zu, Batch = %zu, IsProf = %d\n", Nx, Batch, IsProf);

    std::vector<float2> cx(Nx * Batch);
    std::vector<float2> cy(Nx * Batch);
    std::vector<float2> backx(cx.size());
    for(size_t i = 0; i < Batch; ++i)
    {
		for(size_t k = 0; k < Nx; ++k)
		{
			const size_t pos = i * Nx + k;
			cx[pos].x = (i + k) * 1.0f;
			cx[pos].y = (i + k) * -0.1f;
		}
    }
  
    // Create HIP device objects:
    cufftComplex * x = NULL;
    cufftComplex * y = NULL;
    cudaMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	cudaMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    cudaMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), cudaMemcpyHostToDevice);


    // Create plans
	cufftHandle plan;
	cufftHandle plan2;
	checkCudaErrors(cufftPlan1d(&plan,  Nx, CUFFT_C2C, Batch));		checkCudaErrors(cufftExecC2C(plan,  x, y, CUFFT_FORWARD));
	checkCudaErrors(cufftPlan1d(&plan2, Nx, CUFFT_C2C, Batch));		checkCudaErrors(cufftExecC2C(plan2, y, x, CUFFT_INVERSE));   
    cudaMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), cudaMemcpyDeviceToHost);
  

    double error = 0.0f;
    for(size_t i = 0; i < Batch; i++)
    {
		for(size_t k = 0; k < Nx; k++)
		{
			const size_t pos = i * Nx + k;
			double diffx = std::abs(backx[pos].x / (Nx) - cx[pos].x);
			double diffy = std::abs(backx[pos].y / (Nx) - cx[pos].y);
			double diff = diffx + diffy;
			if(diff > error)
				error = diff;
		}
    }
    std::cout << "Maximum error: " << error << "\n";
	
	if(IsProf > 0)
	{
		int iteration_times = 1000;
		timespec startTime,stopTime;	
		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
		clock_gettime(CLOCK_MONOTONIC, &startTime);
		for(int i = 0;i<iteration_times;i++)
			cufftExecC2C(plan, x, y, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
	
	//delete cx;
	//delete backx;
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(y));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(plan2));
	
}

int main(int argc, char **argv) { runTest(argc, argv); }
