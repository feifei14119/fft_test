
/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
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
	const size_t Ny = (argc < 3) ? 100 : atoi(argv[2]);
	const size_t Nz = (argc < 4) ? 100 : atoi(argv[3]);
    const unsigned int IsProf = (argc < 5) ? 0 : atoi(argv[4]);
	printf("Nx = %zu, Ny = %zu, Nz = %zu, IsProf = %d\n", Nx, Ny, Nz, IsProf);

    std::vector<float> cx(Nx * Ny * Nz);
    std::vector<float> backx(cx.size());
    std::vector<float2> cy(Nx * Ny * Nz);
	for(size_t i = 0; i < Nx; ++i)
	{
		for(size_t j = 0; j < Ny; ++j)
		{
			for(size_t k = 0; k < Nz; ++k)
			{
				const size_t pos = i * Ny * Nz + j * Nz + k;
				cx[pos] = (i + j + k) * 1.0f;
			}
		}
	}
  
    // Create HIP device objects:
    cufftReal * x = NULL;
    cufftComplex * y = NULL;
    cudaMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
	cudaMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    cudaMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), cudaMemcpyHostToDevice);

    // Create plans
	cufftHandle plan;
	cufftHandle plan2;
	checkCudaErrors(cufftPlan3d(&plan,  Nx, Ny, Nz, CUFFT_R2C));	checkCudaErrors(cufftExecR2C(plan,  x, y));
	checkCudaErrors(cufftPlan3d(&plan2, Nx, Ny, Nz, CUFFT_C2R));	checkCudaErrors(cufftExecC2R(plan2, y, x));  
	//checkCudaErrors(cufftPlan3d(&plan, Nx * Ny * Nz, CUFFT_D2Z));	checkCudaErrors(cufftExecD2Z(plan, x, reinterpret_cast<cufftDoubleComplex *>(y)));
	//checkCudaErrors(cufftPlan3d(&plan2, Nx * Ny * Nz, CUFFT_Z2D));checkCudaErrors(cufftExecZ2D(plan2, reinterpret_cast<cufftDoubleComplex *>(y), x));  
    cudaMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), cudaMemcpyDeviceToHost);
  

    double error = 0.0f;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
				const size_t pos = i * Ny * Nz + j * Nz + k;
				double diff = std::abs(backx[pos] / (Nx*Ny*Nz) - cx[pos]);
				if(diff > error)
					error = diff;
            }
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
			cufftExecR2C(plan, x, y);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
		printf("elapsed mill sec = %.3f(ms)\n", ElapsedMilliSec/iteration_times);
	}
	
	//free(cx);
	//free(backx);
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(y));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(plan2));
	
}

int main(int argc, char **argv) { runTest(argc, argv); }
