
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
#define T float
#define T2 float2
void runTest(int argc, char **argv) 
{
	printf("[simpleCUFFT] is starting...\n");

	findCudaDevice(argc, (const char **)argv);
  
	const size_t Nx      = 100;
	const size_t Ny      = 100;
	const size_t Nz      = 100;

    std::vector<T> cx(Nx * Ny * Nz);
    std::vector<T> backx(cx.size());
    std::fill(cx.begin(), cx.end(), 0.0);
    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            for(size_t k = 0; k < Nz; ++k)
            {
                const size_t pos = i * Ny * Nz + j * Nz + k;
                cx[pos]          = i + j + k;
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
	checkCudaErrors(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_R2C));			checkCudaErrors(cufftExecR2C(plan, x, reinterpret_cast<cufftComplex *>(y)));
	checkCudaErrors(cufftPlan3d(&plan2, Nx, Ny, Nz, CUFFT_C2R));		checkCudaErrors(cufftExecC2R(plan2, reinterpret_cast<cufftComplex *>(y), x));  
	//checkCudaErrors(cufftPlan1d(&plan, Nx * Ny * Nz, CUFFT_D2Z, 1));	checkCudaErrors(cufftExecD2Z(plan, x, reinterpret_cast<cufftDoubleComplex *>(y)));
	//checkCudaErrors(cufftPlan1d(&plan2, Nx * Ny * Nz, CUFFT_Z2D, 1));	checkCudaErrors(cufftExecZ2D(plan2, reinterpret_cast<cufftDoubleComplex *>(y), x));  
    cudaMemcpy(backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), cudaMemcpyDeviceToHost);
  

    double       error = 0.0f;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                double diff = std::abs(backx[i] / (Nx * Ny * Nz) - cx[i]);
                if(diff > error)
                    error = diff;
            }
        }
    }
    std::cout << "Maximum error: " << error << "\n";
	
	
	//free(cx);
	//free(backx);
	//checkCudaErrors(cudaFree(x));
	//checkCudaErrors(cudaFree(y));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(plan2));
	
}

int main(int argc, char **argv) { runTest(argc, argv); }
