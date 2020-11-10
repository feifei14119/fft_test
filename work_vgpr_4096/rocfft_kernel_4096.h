#pragma once
#include "rocfft_butterfly_template.h"


////////////////////////////////////////Passes kernels
template <typename T >
__device__ inline void
FwdPass0_len4096(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*stride_in];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 256 )*stride_in];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 512 )*stride_in];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 768 )*stride_in];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 1024 )*stride_in];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 1280 )*stride_in];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 1536 )*stride_in];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 1792 )*stride_in];
	(*R8) = bufIn[inOffset + ( 0 + me*1 + 0 + 2048 )*stride_in];
	(*R9) = bufIn[inOffset + ( 0 + me*1 + 0 + 2304 )*stride_in];
	(*R10) = bufIn[inOffset + ( 0 + me*1 + 0 + 2560 )*stride_in];
	(*R11) = bufIn[inOffset + ( 0 + me*1 + 0 + 2816 )*stride_in];
	(*R12) = bufIn[inOffset + ( 0 + me*1 + 0 + 3072 )*stride_in];
	(*R13) = bufIn[inOffset + ( 0 + me*1 + 0 + 3328 )*stride_in];
	(*R14) = bufIn[inOffset + ( 0 + me*1 + 0 + 3584 )*stride_in];
	(*R15) = bufIn[inOffset + ( 0 + me*1 + 0 + 3840 )*stride_in];
	}



	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 1 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 2 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 3 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 4 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 5 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 6 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 7 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 8 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 9 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 10 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 11 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 12 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 13 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 14 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 15 ) ] = (*R15).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1024 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1280 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1536 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1792 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2048 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2304 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2560 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2816 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3072 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3328 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3584 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3840 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 1 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 2 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 3 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 4 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 5 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 6 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 7 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 8 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 9 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 10 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 11 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 12 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 13 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 14 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*16 + (1*me + 0)%1 + 15 ) ] = (*R15).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1024 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1280 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1536 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1792 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2048 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2304 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2560 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2816 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3072 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3328 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3584 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3840 ) ];
	}


	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass1_len4096(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15)
{




	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 4];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 5];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 6];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 7];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 8];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 9];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 10];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 11];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 12];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 13];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		T W = twiddles[15 + 15*((1*me + 0)%16) + 14];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 16 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 32 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 48 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 64 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 80 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 96 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 112 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 128 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 144 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 160 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 176 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 192 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 208 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 224 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 240 ) ] = (*R15).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1024 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1280 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1536 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 1792 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2048 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2304 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2560 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 2816 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3072 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3328 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3584 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 3840 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 16 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 32 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 48 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 64 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 80 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 96 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 112 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 128 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 144 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 160 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 176 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 192 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 208 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 224 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((1*me + 0)/16)*256 + (1*me + 0)%16 + 240 ) ] = (*R15).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1024 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1280 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1536 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 1792 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2048 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2304 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2560 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 2816 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3072 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3328 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3584 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 3840 ) ];
	}


	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass2_len4096(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15)
{




	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 4];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 5];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 6];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 7];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 8];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 9];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 10];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 11];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 12];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 13];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		T W = twiddles[255 + 15*((1*me + 0)%256) + 14];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);


	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 )*stride_out] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 256 )*stride_out] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 512 )*stride_out] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 768 )*stride_out] = (*R3);
	bufOut[outOffset + ( 1*me + 0 + 1024 )*stride_out] = (*R4);
	bufOut[outOffset + ( 1*me + 0 + 1280 )*stride_out] = (*R5);
	bufOut[outOffset + ( 1*me + 0 + 1536 )*stride_out] = (*R6);
	bufOut[outOffset + ( 1*me + 0 + 1792 )*stride_out] = (*R7);
	bufOut[outOffset + ( 1*me + 0 + 2048 )*stride_out] = (*R8);
	bufOut[outOffset + ( 1*me + 0 + 2304 )*stride_out] = (*R9);
	bufOut[outOffset + ( 1*me + 0 + 2560 )*stride_out] = (*R10);
	bufOut[outOffset + ( 1*me + 0 + 2816 )*stride_out] = (*R11);
	bufOut[outOffset + ( 1*me + 0 + 3072 )*stride_out] = (*R12);
	bufOut[outOffset + ( 1*me + 0 + 3328 )*stride_out] = (*R13);
	bufOut[outOffset + ( 1*me + 0 + 3584 )*stride_out] = (*R14);
	bufOut[outOffset + ( 1*me + 0 + 3840 )*stride_out] = (*R15);
	}

}

////////////////////////////////////////Encapsulated passes kernels
template <typename T >
__device__ inline void 
fwd_len4096_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15;
	FwdPass0_len4096<T >(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15);
	FwdPass1_len4096<T >(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15);
	FwdPass2_len4096<T >(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15);
}

////////////////////////////////////////Global kernels

//Kernel configuration: number of threads per thread block: 256, maximum transforms: 1, Passes: 3
__global__ void 
my_fft_fwd_op_len4096( const float2 * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, float2 * __restrict__ gbIn, float2 * __restrict__ gbOut)
{
	__shared__ real_type_t<float2> lds[4096];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	if(dim == 1){
		iOffset += counter_mod*stride_in[1];
		oOffset += counter_mod*stride_out[1];
	}
	else if(dim == 2){
		int counter_1 = counter_mod / lengths[1];
		int counter_mod_1 = counter_mod % lengths[1];
		iOffset += counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	else if(dim == 3){
		int counter_2 = counter_mod / (lengths[1] * lengths[2]);
		int counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);
		int counter_1 = counter_mod_2 / lengths[1];
		int counter_mod_1 = counter_mod_2 % lengths[1];
		iOffset += counter_2*stride_in[3] + counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_2*stride_out[3] + counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	else{
		for(int i = dim; i>1; i--){
			int currentLength = 1;
			for(int j=1; j<i; j++){
				currentLength *= lengths[j];
			}

			iOffset += (counter_mod / currentLength)*stride_in[i];
			oOffset += (counter_mod / currentLength)*stride_out[i];
			counter_mod = counter_mod % currentLength;
		}
		iOffset+= counter_mod * stride_in[1];
		oOffset+= counter_mod * stride_out[1];
	}
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len4096_device<float2 >(twiddles, stride_in[0], stride_out[0],  1, b, me, 0, lwbIn, lwbOut, lds);
}
