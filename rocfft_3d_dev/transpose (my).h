// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// ipcc -c -target amdgcn-amd-amdhsa --genco  -mcpu=gfx906 -mno-code-object-v3  --save-temps -I/feifei/rocFFT/library/include -I/feifei/rocFFT/library/src/include -I/feifei/rocFFT/testmk/include -I/opt/rocm/include/ -I/usr/inc/ -D__HIP_PLATFORM_HCC__ -I/usr/include/c++/9/ -I/usr/include/x86_64-linux-gnu/c++/9/ -I/usr/include/x86_64-linux-gnu/ transpose.h


#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "array_format.h"
#include "common.h"
#include "rocfft_hip.h"

// WITH_TWL = false
// TWL = 0
// DIR = 0
#define TRANSPOSE_TWIDDLE_MUL()                                                                   \
    shared[tx1][ty1 + i] = tmp; // the transpose taking place here

// - transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
//   input, output are in device memory
//   shared memory of size DIM_X*DIM_X is allocated size_ternally as working space
// - Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
//   DIM_X is divisible by DIM_Y
template <typename T,typename T_I,typename T_O,size_t DIM_X,size_t DIM_Y,bool WITH_TWL,int TWL,int DIR,bool ALL,bool UNIT_STRIDE_0>
__device__ inline void transpose_tile_device(
    const T_I* input, T_O* output, 
    size_t in_offset, size_t out_offset, 
    const size_t m, const size_t n, 
    size_t gx, size_t gy, 
    size_t ld_in, size_t ld_out, 
    size_t stride_0_in, size_t stride_0_out, 
    T* twiddles_large)
{
    // DIM_X = 64
    // DIM_Y = 16
    // group_size = [64,16,1]
    // ty1 = hipThreadIdx_y
    // m = 64 or (100 - 64)
    // n = 64 or (51*100 - 79*64)
    // ld_in = 100
    // ld_out = 5100
    // i = 0,16,32,48

    size_t tx1,ty1;
    __shared__ T shared[DIM_X][DIM_X];

    size_t tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    tx1 = tid % DIM_X;
    ty1 = tid / DIM_X;
    
#if 1    
    for(size_t i = 0; i < m; i += DIM_Y)
    {
        if(tx1 < n && (ty1 + i) < m)
        {
            T tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            shared[tx1][ty1 + i] = tmp;
        }
    }
#endif

    /*for(size_t i = 0; i < m; i += DIM_Y)
    {
        if(tx1 < n && (ty1 + i) < m)
        {
            T tmp = Handler<T_I>::read(input, tx1 + (ty1 + i) * ld_in);
            shared[tx1][ty1 + i] = tmp;
        }
    }*/

    /*size_t dim_x = 32;
    size_t dim_y = 32;
    tx1 = tid % dim_x;
    ty1 = tid / dim_x;
    for(size_t i = 0; i < m; i += dim_y)
    {
        if(tx1 < n && (ty1 + i) < m)
        {
            cmplx_double * input_v4 = (cmplx_double *)input;
            cmplx_double * shared_v4 = (cmplx_double *)shared;

            size_t ld_in_v4 = ld_in >> 1;
            size_t idx_glb = (ty1 + i) * ld_in_v4 + tx1;
            size_t idx_lds = (ty1 + i) * dim_x + tx1;

            cmplx_double tmp_v4;
            tmp_v4 = input_v4[idx_glb];            
            shared_v4[idx_lds] = tmp_v4;
        }
    }*/

    /*for(size_t i = 0; i < m; i += DIM_Y)
    {
        if(tx1 < n && (ty1 + i) < m)
        {
            T tmp = Handler<T_I>::read(input, tx1 + (ty1 + i) * ld_in);
            size_t col_idx = (ty1 + i + tx1) % DIM_X;
            shared[tx1][col_idx] = tmp;
        }
    }*/

    /*size_t iii;
    T tmp1, tmp2, tmp3, tmp4;
    iii = 0;   if(tx1 < n && (ty1 + iii) < m)  tmp1 = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + iii) * ld_in);
    iii = 1;   if(tx1 < n && (ty1 + iii) < m)  tmp2 = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + iii) * ld_in);
    iii = 2;   if(tx1 < n && (ty1 + iii) < m)  tmp3 = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + iii) * ld_in);
    iii = 3;   if(tx1 < n && (ty1 + iii) < m)  tmp4 = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + iii) * ld_in);
    iii = 0;   shared[tx1][ty1 + iii] = tmp1;
    iii = 1;   shared[tx1][ty1 + iii] = tmp2;
    iii = 2;   shared[tx1][ty1 + iii] = tmp3;
    iii = 3;   shared[tx1][ty1 + iii] = tmp4;*/

    __syncthreads();

    for(size_t i = 0; i < n; i += DIM_Y)
    {
        // reconfigure the threads
        if(tx1 < m && (ty1 + i) < n)
        {
            size_t col_idx = (tx1 + ty1 + i) % DIM_X;
            Handler<T_O>::write(output,
                                out_offset + tx1 + (i + ty1) * ld_out,
                                shared[ty1 + i][tx1]);
        }
    }
}

template <typename T,typename T_I,typename T_O,size_t DIM_X,size_t DIM_Y,bool   WITH_TWL,int    TWL,int    DIR,bool   ALL,bool   UNIT_STRIDE_0,bool   DIAGONAL>
__global__ void transpose_kernel2(const T_I* input,T_O* output,T* twiddles_large,size_t* lengths,size_t* stride_in,size_t* stride_out){}

template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y, bool   ALL, bool   UNIT_STRIDE_0, bool   DIAGONAL>
__global__ void transpose_kernel2_scheme(const T_I* input,T_O* output,T* twiddles_large,size_t* lengths,size_t* stride_in,size_t* stride_out, const size_t scheme)
{
    //return;
    // ALL = false
    // UNIT_STRIDE_0 = true
    // DIAGONAL = false
    // lengths[0] = 100
    // lengths[1] = 100
    // lengths[2] = 51
    // stride_in[0]  = 1
    // stride_in[1]  = 100
    // stride_in[2]  = 5100
    // stride_out[0] = 1
    // stride_out[1] = 5100
    // stride_out[2] = 100
    // ld_in  = 100
    // ld_out = 5100
    // scheme = 2
    
    size_t ld_in  = scheme == 1 ? stride_in[2] : stride_in[1];
    size_t ld_out = scheme == 1 ? stride_out[1] : stride_out[2];

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = hipBlockIdx_z;

    iOffset += counter_mod * stride_in[3];
    oOffset += counter_mod * stride_out[3];

    size_t tileBlockIdx_x, tileBlockIdx_y;
    if(DIAGONAL) // diagonal reordering
    {
        //TODO: template and simplify index calc for square case if necessary
        size_t bid     = hipBlockIdx_x + gridDim.x * hipBlockIdx_y;
        tileBlockIdx_y = bid % hipGridDim_y;
        tileBlockIdx_x = ((bid / hipGridDim_y) + tileBlockIdx_y) % hipGridDim_x;
    }
    else
    {
        tileBlockIdx_x = hipBlockIdx_x;
        tileBlockIdx_y = hipBlockIdx_y;
    }

    iOffset += tileBlockIdx_x * DIM_X * stride_in[0] + tileBlockIdx_y * DIM_X * ld_in;
    oOffset += tileBlockIdx_x * DIM_X * ld_out + tileBlockIdx_y * DIM_X * stride_out[0];

    size_t m  = scheme == 1 ? lengths[2] : lengths[1] * lengths[2];
    size_t n  = scheme == 1 ? lengths[0] * lengths[1] : lengths[0];
    size_t mm = min(m - tileBlockIdx_y * DIM_X, DIM_X); // the corner case along m
    size_t nn = min(n - tileBlockIdx_x * DIM_X, DIM_X); // the corner case along n
    transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, false, 0, 0, ALL, UNIT_STRIDE_0>(
        input, output,
        iOffset, oOffset,
        mm, nn,
        hipBlockIdx_x * DIM_X, hipBlockIdx_y * DIM_X,
        ld_in, ld_out,
        stride_in[0], stride_out[0],
        twiddles_large);
}

template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y, bool   ALL, bool   UNIT_STRIDE_0, bool   DIAGONAL>
__global__ void transpose_kernel2_scheme111(const T_I* input,T_O* output,T* twiddles_large,size_t* lengths,size_t* stride_in,size_t* stride_out, const size_t scheme)
{    
    return;
    __shared__ T shared[DIM_X][DIM_X];

    uint32_t bid = hipBlockIdx_y  * hipGridDim_x  + hipBlockIdx_x;
    uint32_t tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    uint32_t tx1 = tid % DIM_X;
    uint32_t ty1 = tid / DIM_X;

    uint32_t ld_in  = scheme == 1 ? stride_in[2] : stride_in[1];
    uint32_t ld_out = scheme == 1 ? stride_out[1] : stride_out[2];

    uint32_t iOffset = hipBlockIdx_x * DIM_X * stride_in[0] + hipBlockIdx_y * DIM_X * ld_in;
    uint32_t oOffset = hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X * stride_out[0];

    uint32_t width  = scheme == 1 ? lengths[0] * lengths[1] : lengths[0];
    uint32_t height = scheme == 1 ? lengths[2] : lengths[1] * lengths[2];
    uint32_t lmt_width  = min(width - hipBlockIdx_x * DIM_X, DIM_X);
    uint32_t lmt_height = min(height - hipBlockIdx_y * DIM_X, DIM_X);

    uint32_t dim_x = DIM_X; 
    uint32_t dim_y = DIM_Y;
    uint32_t loop_num = lmt_height / DIM_Y;

    input += (iOffset + tx1);
    if(tx1 < lmt_width)
    {
        for(uint32_t loop_cnt = 0; loop_cnt < loop_num; loop_cnt ++)
        {
            uint32_t i = loop_cnt * DIM_Y;
            {
                T tmp = Handler<T_I>::read(input, (ty1 + i) * ld_in);
                shared[tx1][ty1 + i] = tmp;
            }
        }

        if((ty1 + loop_num * DIM_Y) < lmt_height)
        {
            T tmp = Handler<T_I>::read(input, (ty1 + loop_num* DIM_Y) * ld_in);
            shared[tx1][ty1 + loop_num* DIM_Y] = tmp;
        }
    }


    __syncthreads();

    for(uint32_t i = 0; i < lmt_width; i += DIM_Y)
    {
        // reconfigure the threads
        if(tx1 < lmt_height && (ty1 + i) < lmt_width)
        {
            Handler<T_O>::write(output, oOffset + tx1 + (i + ty1) * ld_out, shared[ty1 + i][tx1]);
        }
    }
    return;

}

#endif // TRANSPOSE_H
