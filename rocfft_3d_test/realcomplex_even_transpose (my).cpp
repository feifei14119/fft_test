// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "./kernels/array_format.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"

#include <numeric>

__device__ size_t output_row_base(size_t dim, size_t output_batch_start, const size_t* outStride, const size_t  col)
{
    if(dim == 2)        return output_batch_start + outStride[1] * col;
    else if(dim == 3)   return output_batch_start + outStride[2] * col;
    return 0;
}

#define BLOCK_ITER_SIZE (1)

// R2C post-process kernel, 2D and 3D, transposed output.
// lengths counts in complex elements
template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y>
__global__ static void real_post_process_kernel_transpose111(size_t        dim,
                                                          const T_I*    input0,
                                                          size_t        idist,
                                                          T_O*          output0,
                                                          size_t        odist,
                                                          const void*   twiddles0,
                                                          const size_t* lengths,
                                                          const size_t* inStride,
                                                          const size_t* outStride)
{
    //return;
    // idist1D = 50
    // odist1D = 100
    // idist = 500000
    // odist = 510000
    // len0 == 50
    // tile_size = DIM_X = 16
    // dim = 3
    // row_limit = 10000
    // lengths[0]   = 50; lengths[1]   = 100; lengths[2]   = 100;
    // inStride[0]  = 1;  inStride[1]  = 50;  inStride[2]  = 5000;
    // outStride[0] = 1;  outStride[1] = 100; outStride[2] = 10000;

    size_t idist1D            =  inStride[1];
    size_t odist1D            = outStride[1];
    size_t input_batch_start  = idist * blockIdx.z;
    size_t output_batch_start = odist * blockIdx.z;
    const T * twiddles        = static_cast<const T*>(twiddles0);

    // allocate 2 tiles so we can butterfly the values together.
    // left tile grabs values from towards the beginnings of the rows
    // right tile grabs values from towards the ends
    __shared__ T  leftTile[DIM_X][DIM_Y+1];
    __shared__ T rightTile[DIM_X][DIM_Y+1];

    // take fastest dimension and partition it into lengths that will go into each tile
    const size_t len0 = lengths[0];
    // size of a complete tile for this problem - ignore the first
    // element and middle element (if there is one).  those are
    // treated specially
    const size_t tile_size = (len0 - 1) / 2 < DIM_X ? (len0 - 1) / 2 : DIM_X;

    // first column to read into the left tile, offset by one because
    // first element is already handled
    const size_t left_col_start = blockIdx.x * tile_size + 1;
    const size_t middle         = (len0 + 1) / 2;

    // number of columns to actually read into the tile (can be less
    // than tile size if we're out of data)
    size_t cols_to_read = tile_size;
    if(left_col_start + tile_size >= middle)
        cols_to_read = middle - left_col_start;

    // maximum number of rows in the problem
    const size_t col_limit = lengths[0];
    const size_t row_limit = dim == 2 ? lengths[1] : lengths[1] * lengths[2];

    // start+end of range this thread will work on
    for(uint32_t itcnt = 0; itcnt<BLOCK_ITER_SIZE; itcnt++)
    {
        //const size_t row_start = blockIdx.y * DIM_Y;
        //size_t       row_end   = DIM_Y + row_start;
        const size_t row_start = blockIdx.y * DIM_Y*BLOCK_ITER_SIZE;
        size_t       row_end   = DIM_Y*BLOCK_ITER_SIZE + row_start;
        if(row_end > row_limit)
            row_end = row_limit;

        //const size_t lds_row = threadIdx.y;
        const size_t lds_row = threadIdx.y + DIM_Y*itcnt;
        const size_t lds_col = threadIdx.x;
        // TODO: currently assumes idist2D has no extra padding
        const size_t input_row_base = (row_start + lds_row) * idist1D;

        if(row_start + lds_row < row_end && lds_col < cols_to_read)
        {
            auto v = Handler<T_I>::read(input0, input_batch_start + input_row_base + left_col_start + lds_col);
            leftTile[lds_row][lds_col] = v;
            auto v2 = Handler<T_I>::read(input0, input_batch_start + input_row_base + (len0 - (left_col_start + cols_to_read - 1)) + lds_col);
            rightTile[lds_row][lds_col] = v2;
        }

        __syncthreads();

        // butterfly the two tiles we've collected (offset col by one
        // because first element is special)
        T tmp;
        T tmp2;
        if(row_start + lds_row < row_end && lds_col < cols_to_read)
        {
            size_t col = blockIdx.x * tile_size + 1 + threadIdx.x;

            const T p =  leftTile[lds_row][lds_col];
            const T q = rightTile[lds_row][cols_to_read - lds_col - 1];
            const T u = 0.5 * (p + q);
            const T v = 0.5 * (p - q);

            auto twd_p = twiddles[col];

            // write left side
            tmp.x = u.x + v.x * twd_p.y + u.y * twd_p.x;
            tmp.y = v.y + u.y * twd_p.y - v.x * twd_p.x;
            // write right side
            tmp2.x =  u.x - v.x * twd_p.y - u.y * twd_p.x;
            tmp2.y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////
        if(row_start + lds_row < row_end && lds_col < cols_to_read)
        {
            leftTile[lds_row][lds_col]  = tmp;
            rightTile[lds_row][lds_col] = tmp2;
            __syncthreads();
        }

        uint32_t lds_rd_row = threadIdx.x + DIM_Y*itcnt;
        uint32_t lds_rd_col = threadIdx.y;
        uint32_t glb_row = blockIdx.x * DIM_X + threadIdx.y + 1;
        uint32_t glb_col = blockIdx.y * DIM_Y*BLOCK_ITER_SIZE + threadIdx.x;

        if(lds_rd_col < cols_to_read && glb_col < row_end)
        {
            tmp  =  leftTile[lds_rd_row][lds_rd_col];
            tmp2 = rightTile[lds_rd_row][lds_rd_col];
            Handler<T_O>::write(output0, output_row_base(dim, output_batch_start, outStride, glb_row)        + glb_col, tmp);
            Handler<T_O>::write(output0, output_row_base(dim, output_batch_start, outStride, len0 - glb_row) + glb_col, tmp2);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////
        if(blockIdx.x != 0) return;
        // handle first + middle element (if there is a middle)
        // write first + middle        
        T first_elem;
        T middle_elem;
        uint32_t row_start2 = blockIdx.y * DIM_Y*BLOCK_ITER_SIZE;
        uint32_t lds_row2 = threadIdx.x + DIM_Y*itcnt;
        uint32_t input_row_base2 = (row_start2 + lds_row2) * idist1D;
        uint32_t glb_wr_col = row_start2 + lds_row2;

        if(threadIdx.y == 0 && row_start2 + lds_row2 < row_end)
        {
            first_elem = Handler<T_I>::read(input0, input_batch_start + input_row_base2);

            if(len0 % 2 == 0)
            {
                middle_elem = Handler<T_I>::read(input0, input_batch_start + input_row_base2 + len0 / 2);
            }        

            tmp.x  = first_elem.x - first_elem.y; tmp.y  = 0.0;
            Handler<T_O>::write(output0, output_row_base(dim, output_batch_start, outStride, len0)  + glb_wr_col, tmp);
            tmp2.x = first_elem.x + first_elem.y; tmp2.y = 0.0;                             
            Handler<T_O>::write(output0, output_row_base(dim, output_batch_start, outStride, 0)     + glb_wr_col, tmp2);

            if(len0 % 2 == 0)
            {
                tmp.x =  middle_elem.x; tmp.y = -middle_elem.y;
                Handler<T_O>::write(output0, output_row_base(dim, output_batch_start, outStride, middle) + glb_wr_col, tmp);
            }
        }

        //__syncthreads();
    }
}

// R2C post-process kernel, 2D and 3D, transposed output.
// lengths counts in complex elements
template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y>
__global__ static void real_post_process_kernel_transpose(size_t        dim,
                                                          const T_I*    input0,
                                                          size_t        idist,
                                                          T_O*          output0,
                                                          size_t        odist,
                                                          const void*   twiddles0,
                                                          const size_t* lengths,
                                                          const size_t* inStride,
                                                          const size_t* outStride)
{
    size_t idist1D            = inStride[1];
    size_t odist1D            = outStride[1];
    size_t input_batch_start  = idist * blockIdx.z;
    size_t output_batch_start = odist * blockIdx.z;
    auto   twiddles           = static_cast<const T*>(twiddles0);

    // allocate 2 tiles so we can butterfly the values together.
    // left tile grabs values from towards the beginnings of the rows
    // right tile grabs values from towards the ends
    __shared__ T leftTile[DIM_X][DIM_Y];
    __shared__ T rightTile[DIM_X][DIM_Y];

    // take fastest dimension and partition it into lengths that will go into each tile
    const size_t len0 = lengths[0];
    // size of a complete tile for this problem - ignore the first
    // element and middle element (if there is one).  those are
    // treated specially
    const size_t tile_size = (len0 - 1) / 2 < DIM_X ? (len0 - 1) / 2 : DIM_X;

    // first column to read into the left tile, offset by one because
    // first element is already handled
    const size_t left_col_start = blockIdx.x * tile_size + 1;
    const size_t middle         = (len0 + 1) / 2;

    // number of columns to actually read into the tile (can be less
    // than tile size if we're out of data)
    size_t cols_to_read = tile_size;
    if(left_col_start + tile_size >= middle)
        cols_to_read = middle - left_col_start;

    // maximum number of rows in the problem
    const size_t row_limit = dim == 2 ? lengths[1] : lengths[1] * lengths[2];

    // start+end of range this thread will work on
    const size_t row_start = blockIdx.y * DIM_Y;
    size_t       row_end   = DIM_Y + row_start;
    if(row_end > row_limit)
        row_end = row_limit;

    const size_t lds_row = threadIdx.y;
    const size_t lds_col = threadIdx.x;
    // TODO: currently assumes idist2D has no extra padding
    const size_t input_row_base = (row_start + lds_row) * idist1D;

    if(row_start + lds_row < row_end && lds_col < cols_to_read)
    {
        auto v                     = Handler<T_I>::read(input0,
                                    input_batch_start + input_row_base + left_col_start + lds_col);
        leftTile[lds_col][lds_row] = v;

        auto v2                     = Handler<T_I>::read(input0,
                                     input_batch_start + input_row_base
                                         + (len0 - (left_col_start + cols_to_read - 1)) + lds_col);
        rightTile[lds_col][lds_row] = v2;
    }

    // handle first + middle element (if there is a middle)
    T first_elem;
    T middle_elem;
    if(blockIdx.x == 0 && threadIdx.x == 0 && row_start + lds_row < row_end)
    {
        first_elem = Handler<T_I>::read(input0, input_batch_start + input_row_base);

        if(len0 % 2 == 0)
        {
            middle_elem = Handler<T_I>::read(input0, input_batch_start + input_row_base + len0 / 2);
        }
    }

    __syncthreads();

    // write first + middle
    if(blockIdx.x == 0 && threadIdx.x == 0 && row_start + lds_row < row_end)
    {
        T tmp;
        tmp.x = first_elem.x - first_elem.y;
        tmp.y = 0.0;
        Handler<T_O>::write(output0,
                            output_row_base(dim, output_batch_start, outStride, len0) + row_start
                                + lds_row,
                            tmp);
        T tmp2;
        tmp2.x = first_elem.x + first_elem.y;
        tmp2.y = 0.0;
        Handler<T_O>::write(output0,
                            output_row_base(dim, output_batch_start, outStride, 0) + row_start
                                + lds_row,
                            tmp2);

        if(len0 % 2 == 0)
        {

            tmp.x = middle_elem.x;
            tmp.y = -middle_elem.y;

            Handler<T_O>::write(output0,
                                output_row_base(dim, output_batch_start, outStride, middle)
                                    + row_start + lds_row,
                                tmp);
        }
    }

    // butterfly the two tiles we've collected (offset col by one
    // because first element is special)
    if(row_start + lds_row < row_end && lds_col < cols_to_read)
    {
        size_t col = blockIdx.x * tile_size + 1 + threadIdx.x;

        const T p = leftTile[lds_col][lds_row];
        const T q = rightTile[cols_to_read - lds_col - 1][lds_row];
        const T u = 0.5 * (p + q);
        const T v = 0.5 * (p - q);

        auto twd_p = twiddles[col];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        // write left side
        T tmp;
        tmp.x                 = u.x + v.x * twd_p.y + u.y * twd_p.x;
        tmp.y                 = v.y + u.y * twd_p.y - v.x * twd_p.x;
        auto output_left_base = output_row_base(dim, output_batch_start, outStride, col);
        Handler<T_O>::write(output0, output_left_base + row_start + lds_row, tmp);

        // write right side
        T tmp2;
        tmp2.x                 = u.x - v.x * twd_p.y - u.y * twd_p.x;
        tmp2.y                 = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        auto output_right_base = output_row_base(dim, output_batch_start, outStride, len0 - col);
        Handler<T_O>::write(output0, output_right_base + row_start + lds_row, tmp2);
    }
}

// Entrance function for r2c post-processing kernel, fused with transpose
void r2c_1d_post_transpose(const void* data_p, void*)
{
    auto data = reinterpret_cast<const DeviceCallIn*>(data_p);

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    const void* bufIn0  = data->bufIn[0];
    void*       bufOut0 = data->bufOut[0];
    void*       bufOut1 = data->bufOut[1];

    const size_t batch = data->node->batch;

    size_t count = data->node->batch;
    size_t m     = data->node->length[1];
    size_t n     = data->node->length[0];
    size_t dim   = data->node->length.size();

    // we're allocating one thread per tile element.  16x16 seems to
    // hit a sweet spot for performance, where it's enough threads to
    // be useful, but not too many.
    //
    // NOTE: template params to real_post_process_kernel_transpose
    // need to agree with these numbers
    static const size_t DIM_X = 16;
    static const size_t DIM_Y = 16;

    // grid X dimension handles 2 tiles at a time, so allocate enough
    // blocks to go halfway across 'n'
    //
    // grid Y dimension needs enough blocks to handle the second
    // dimension - multiply by the third dimension to get enough
    // blocks, if we're doing 3D
    //
    // grid Z counts number of batches
    dim3 grid((n - 1) / DIM_X / 2 + 1,
              ((m - 1) / DIM_Y + 1) * (dim > 2 ? data->node->length[2] : 1),
              count);
    grid.y = grid.y / BLOCK_ITER_SIZE;
    
    // one thread per element in a tile
    dim3 threads(DIM_X, DIM_Y, 1);
    //printf("[FF]: r2c_1d_post_transpose.\n");
    //printf("[FF]: n = %zu, m = %zu, dim = %zu.\n", n, m, dim);
    //printf("[FF]: idist = %zu, odist = %zu.\n", idist, odist);

    // rc input should always be interleaved by this point - we
    // should have done a transform just before this operation which
    // outputs interleaved
    assert(is_complex_interleaved(data->node->inArrayType));
    if(data->node->precision == rocfft_precision_single)
    {
        if(is_complex_planar(data->node->outArrayType))
        {
            cmplx_planar_device_buffer<float2> out_planar(bufOut0, bufOut1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_float,
                                                                   cmplx_float,
                                                                   cmplx_float_planar,
                                                                   16,
                                                                   16>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_float*>(bufIn0),
                idist,
                out_planar.devicePtr(),
                odist,
                data->node->twiddles.data(),
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH);
        }
        else
        {
            //printf("[FF]: r2c_1d_post_transpose 2.\n");
            //printf("[FF]: grid size = [%d, %d, %d].\n",grid.x, grid.y, grid.z);
            //printf("[FF]: group size = [%d, %d, %d].\n",threads.x, threads.y, threads.z);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_float,
                                                                                  cmplx_float,
                                                                                  cmplx_float,
                                                                                  16,
                                                                                  16>),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               dim,
                               static_cast<const cmplx_float*>(bufIn0),
                               idist,
                               static_cast<cmplx_float*>(bufOut0),
                               odist,
                               data->node->twiddles.data(),
                               data->node->devKernArg.data(),
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH);
        }
    }
    else
    {
        if(is_complex_planar(data->node->outArrayType))
        {
            cmplx_planar_device_buffer<double2> out_planar(bufOut0, bufOut1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_double,
                                                                   cmplx_double,
                                                                   cmplx_double_planar,
                                                                   16,
                                                                   16>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_double*>(bufIn0),
                idist,
                out_planar.devicePtr(),
                odist,
                data->node->twiddles.data(),
                data->node->devKernArg.data(),
                data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_double,
                                                                                  cmplx_double,
                                                                                  cmplx_double,
                                                                                  16,
                                                                                  16>),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               dim,
                               static_cast<const cmplx_double*>(bufIn0),
                               idist,
                               static_cast<cmplx_double*>(bufOut0),
                               odist,
                               data->node->twiddles.data(),
                               data->node->devKernArg.data(),
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH,
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH);
        }
    }
}
