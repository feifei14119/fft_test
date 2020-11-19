#pragma once
#include "rocfft_butterfly_template.h"


////////////////////////////////////////Passes kernels
template <typename T >
__device__ inline void
FwdPass0_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	//  if(rw)
	{
		(*R0) = bufIn[inOffset + ( 0 + me*3 + 0 + 0 )*stride_in];
		(*R3) = bufIn[inOffset + ( 0 + me*3 + 1 + 0 )*stride_in];
		(*R6) = bufIn[inOffset + ( 0 + me*3 + 2 + 0 )*stride_in];
		(*R1) = bufIn[inOffset + ( 0 + me*3 + 0 + 729 )*stride_in];
		(*R4) = bufIn[inOffset + ( 0 + me*3 + 1 + 729 )*stride_in];
		(*R7) = bufIn[inOffset + ( 0 + me*3 + 2 + 729 )*stride_in];
		(*R2) = bufIn[inOffset + ( 0 + me*3 + 0 + 1458 )*stride_in];
		(*R5) = bufIn[inOffset + ( 0 + me*3 + 1 + 1458 )*stride_in];
		(*R8) = bufIn[inOffset + ( 0 + me*3 + 2 + 1458 )*stride_in];
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 1 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 2 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 1 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 2 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 1 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 2 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 1 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/1)*3 + (3*me + 0)%1 + 2 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 1 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/1)*3 + (3*me + 1)%1 + 2 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 1 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/1)*3 + (3*me + 2)%1 + 2 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass1_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[2 + 2*((3*me + 0)%3) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[2 + 2*((3*me + 0)%3) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[2 + 2*((3*me + 1)%3) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[2 + 2*((3*me + 1)%3) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[2 + 2*((3*me + 2)%3) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[2 + 2*((3*me + 2)%3) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 3 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 6 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 3 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 6 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 3 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 6 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 3 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/3)*9 + (3*me + 0)%3 + 6 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 3 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/3)*9 + (3*me + 1)%3 + 6 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 3 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/3)*9 + (3*me + 2)%3 + 6 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}
	
	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass2_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[8 + 2*((3*me + 0)%9) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[8 + 2*((3*me + 0)%9) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[8 + 2*((3*me + 1)%9) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[8 + 2*((3*me + 1)%9) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[8 + 2*((3*me + 2)%9) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[8 + 2*((3*me + 2)%9) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 9 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 18 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 9 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 18 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 9 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 18 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 9 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/9)*27 + (3*me + 0)%9 + 18 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 9 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/9)*27 + (3*me + 1)%9 + 18 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 9 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/9)*27 + (3*me + 2)%9 + 18 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass3_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[26 + 2*((3*me + 0)%27) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[26 + 2*((3*me + 0)%27) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[26 + 2*((3*me + 1)%27) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[26 + 2*((3*me + 1)%27) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[26 + 2*((3*me + 2)%27) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[26 + 2*((3*me + 2)%27) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 27 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 54 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 27 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 54 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 27 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 54 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 27 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/27)*81 + (3*me + 0)%27 + 54 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 27 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/27)*81 + (3*me + 1)%27 + 54 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 27 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/27)*81 + (3*me + 2)%27 + 54 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass4_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[80 + 2*((3*me + 0)%81) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[80 + 2*((3*me + 0)%81) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[80 + 2*((3*me + 1)%81) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[80 + 2*((3*me + 1)%81) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[80 + 2*((3*me + 2)%81) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[80 + 2*((3*me + 2)%81) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 81 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 162 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 81 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 162 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 81 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 162 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 81 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/81)*243 + (3*me + 0)%81 + 162 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 81 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/81)*243 + (3*me + 1)%81 + 162 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 81 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/81)*243 + (3*me + 2)%81 + 162 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass5_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[242 + 2*((3*me + 0)%243) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[242 + 2*((3*me + 0)%243) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[242 + 2*((3*me + 1)%243) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[242 + 2*((3*me + 1)%243) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[242 + 2*((3*me + 2)%243) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[242 + 2*((3*me + 2)%243) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 243 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 486 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 0 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 243 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 486 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 243 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 486 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 243 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/243)*729 + (3*me + 0)%243 + 486 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 0 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 243 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/243)*729 + (3*me + 1)%243 + 486 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 243 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/243)*729 + (3*me + 2)%243 + 486 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass12345_len2187(unsigned int bias,
					const T *twiddles, 
					const size_t stride_in, const size_t stride_out, 
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe,  real_type_t<T> *bufInIm, 
					real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
					T *R0, T *R1, T *R2, 
					T *R3, T *R4, T *R5, 
					T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[bias-1 + 2*((3*me + 0)%bias) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[bias-1 + 2*((3*me + 0)%bias) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[bias-1 + 2*((3*me + 1)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[bias-1 + 2*((3*me + 1)%bias) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[bias-1 + 2*((3*me + 2)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[bias-1 + 2*((3*me + 2)%bias) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOutRe[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + 0      ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + bias   ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + bias*2 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + 0      ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + bias   ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + bias*2 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + 0      ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + bias   ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + bias*2 ) ] = (*R8).x;
		
		__syncthreads();
		
		(*R0).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 0    ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 0    ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 0    ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 729  ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 729  ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 729  ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
		
		__syncthreads();
		
		bufOutIm[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + 0      ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + bias   ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((3*me + 0)/bias)*bias*3 + (3*me + 0)%bias + bias*2 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + 0      ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + bias   ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((3*me + 1)/bias)*bias*3 + (3*me + 1)%bias + bias*2 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + 0      ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + bias   ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((3*me + 2)/bias)*bias*3 + (3*me + 2)%bias + bias*2 ) ] = (*R8).y;
		
		__syncthreads();
		
		(*R0).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 0    ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 0    ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 0    ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 729  ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 729  ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 729  ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*3 + 0 + 1458 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*3 + 1 + 1458 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*3 + 2 + 1458 ) ];
	}

	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass6_len2187(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8)
{
	{
		T W = twiddles[728 + 2*((3*me + 0)%729) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[728 + 2*((3*me + 0)%729) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[728 + 2*((3*me + 1)%729) + 0];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[728 + 2*((3*me + 1)%729) + 1];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[728 + 2*((3*me + 2)%729) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[728 + 2*((3*me + 2)%729) + 1];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	FwdRad3B1(R0, R1, R2);
	FwdRad3B1(R3, R4, R5);
	FwdRad3B1(R6, R7, R8);

	//  if(rw)
	{
		bufOut[outOffset + ( 3*me + 0 + 0 )*stride_out] = (*R0);
		bufOut[outOffset + ( 3*me + 1 + 0 )*stride_out] = (*R3);
		bufOut[outOffset + ( 3*me + 2 + 0 )*stride_out] = (*R6);
		bufOut[outOffset + ( 3*me + 0 + 729 )*stride_out] = (*R1);
		bufOut[outOffset + ( 3*me + 1 + 729 )*stride_out] = (*R4);
		bufOut[outOffset + ( 3*me + 2 + 729 )*stride_out] = (*R7);
		bufOut[outOffset + ( 3*me + 0 + 1458 )*stride_out] = (*R2);
		bufOut[outOffset + ( 3*me + 1 + 1458 )*stride_out] = (*R5);
		bufOut[outOffset + ( 3*me + 2 + 1458 )*stride_out] = (*R8);
	}
}

////////////////////////////////////////Encapsulated passes kernels
template <typename T >
__device__ inline void 
fwd_len2187_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7, R8;
	FwdPass0_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	
	unsigned int bias = 1;// 0=3; 1=9; 2=27; 3=81; 4=243
	for(uint32_t i = 0; i < 5; i++)
	{
		__syncthreads();
		__syncthreads();
		__syncthreads();
		bias = bias * 3;
		FwdPass12345_len2187<T>(bias, twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	}

	//__syncthreads();
	//__syncthreads();
	//__syncthreads();
	//FwdPass1_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	//
	//__syncthreads();
	//__syncthreads();
	//__syncthreads();
	//FwdPass2_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	//
	//
	//__syncthreads();
	//__syncthreads();
	//__syncthreads();
	//FwdPass3_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	//
	//__syncthreads();
	//__syncthreads();
	//__syncthreads();
	//FwdPass4_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	//
	//__syncthreads();
	//__syncthreads();
	//__syncthreads();
	//FwdPass5_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
	
	__syncthreads();
	__syncthreads();
	__syncthreads();
	FwdPass6_len2187<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8);
}

////////////////////////////////////////Global kernels
//Kernel configuration: number of threads per thread block: 243, maximum transforms: 1, Passes: 7
__global__ void 
my_fft_fwd_op_len2187( const float2 * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, float2 * __restrict__ gbIn, float2 * __restrict__ gbOut)
{
	__shared__ float lds[2187];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int rw = 1;

	unsigned int b = 0;

	size_t counter_mod = batch;
	//if(dim == 1)
	{
		iOffset += counter_mod*stride_in[1];
		oOffset += counter_mod*stride_out[1];
	}
	/*else if(dim == 2){
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
	}*/
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len2187_device<float2>(twiddles, stride_in[0], stride_out[0],  1, b, me, 0, lwbIn, lwbOut, lds);
}
