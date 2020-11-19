#pragma once
#include "rocfft_butterfly_template.h"


////////////////////////////////////////Passes kernels
template <typename T >
__device__ inline void
FwdPass0_len112(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	// if(rw)
	{
		(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 )*stride_in];
		(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 )*stride_in];
		(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 16 )*stride_in];
		(*R8) = bufIn[inOffset + ( 0 + me*2 + 1 + 16 )*stride_in];
		(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 )*stride_in];
		(*R9) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 )*stride_in];
		(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 48 )*stride_in];
		(*R10) = bufIn[inOffset + ( 0 + me*2 + 1 + 48 )*stride_in];
		(*R4) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 )*stride_in];
		(*R11) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 )*stride_in];
		(*R5) = bufIn[inOffset + ( 0 + me*2 + 0 + 80 )*stride_in];
		(*R12) = bufIn[inOffset + ( 0 + me*2 + 1 + 80 )*stride_in];
		(*R6) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 )*stride_in];
		(*R13) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 )*stride_in];
	}

	FwdRad7B1(R0, R1, R2, R3, R4, R5, R6);
	FwdRad7B1(R7, R8, R9, R10, R11, R12, R13);

	// if(rw)
	{
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 1 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 2 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 3 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 4 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 5 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 6 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 0 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 1 ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 2 ) ] = (*R9).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 3 ) ] = (*R10).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 4 ) ] = (*R11).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 5 ) ] = (*R12).x;
		bufOutRe[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 6 ) ] = (*R13).x;
	
	__syncthreads();
	
		(*R0).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	
	__syncthreads();
	
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 1 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 2 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 3 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 4 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 5 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((2*me + 0)/1)*7 + (2*me + 0)%1 + 6 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 0 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 1 ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 2 ) ] = (*R9).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 3 ) ] = (*R10).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 4 ) ] = (*R11).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 5 ) ] = (*R12).y;
		bufOutIm[outOffset + ( ((2*me + 1)/1)*7 + (2*me + 1)%1 + 6 ) ] = (*R13).y;
	
	__syncthreads();
	
		(*R0).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	}
	
	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass1_len112(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	{
		T W = twiddles[6 + 1*((7*me + 0)%7) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;

		W = twiddles[6 + 1*((7*me + 1)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[6 + 1*((7*me + 2)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[6 + 1*((7*me + 3)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[6 + 1*((7*me + 4)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[6 + 1*((7*me + 5)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[6 + 1*((7*me + 6)%7) + 0];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	FwdRad2B1(R0, R1);
	FwdRad2B1(R2, R3);
	FwdRad2B1(R4, R5);
	FwdRad2B1(R6, R7);
	FwdRad2B1(R8, R9);
	FwdRad2B1(R10, R11);
	FwdRad2B1(R12, R13);

	// if(rw)
	{
		bufOutRe[outOffset + ( ((7*me + 0)/7)*14 + (7*me + 0)%7 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((7*me + 0)/7)*14 + (7*me + 0)%7 + 7 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((7*me + 1)/7)*14 + (7*me + 1)%7 + 0 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((7*me + 1)/7)*14 + (7*me + 1)%7 + 7 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((7*me + 2)/7)*14 + (7*me + 2)%7 + 0 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((7*me + 2)/7)*14 + (7*me + 2)%7 + 7 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((7*me + 3)/7)*14 + (7*me + 3)%7 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((7*me + 3)/7)*14 + (7*me + 3)%7 + 7 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((7*me + 4)/7)*14 + (7*me + 4)%7 + 0 ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((7*me + 4)/7)*14 + (7*me + 4)%7 + 7 ) ] = (*R9).x;
		bufOutRe[outOffset + ( ((7*me + 5)/7)*14 + (7*me + 5)%7 + 0 ) ] = (*R10).x;
		bufOutRe[outOffset + ( ((7*me + 5)/7)*14 + (7*me + 5)%7 + 7 ) ] = (*R11).x;
		bufOutRe[outOffset + ( ((7*me + 6)/7)*14 + (7*me + 6)%7 + 0 ) ] = (*R12).x;
		bufOutRe[outOffset + ( ((7*me + 6)/7)*14 + (7*me + 6)%7 + 7 ) ] = (*R13).x;

	__syncthreads();
	
		(*R0).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	
	__syncthreads();
	
		bufOutIm[outOffset + ( ((7*me + 0)/7)*14 + (7*me + 0)%7 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((7*me + 0)/7)*14 + (7*me + 0)%7 + 7 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((7*me + 1)/7)*14 + (7*me + 1)%7 + 0 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((7*me + 1)/7)*14 + (7*me + 1)%7 + 7 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((7*me + 2)/7)*14 + (7*me + 2)%7 + 0 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((7*me + 2)/7)*14 + (7*me + 2)%7 + 7 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((7*me + 3)/7)*14 + (7*me + 3)%7 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((7*me + 3)/7)*14 + (7*me + 3)%7 + 7 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((7*me + 4)/7)*14 + (7*me + 4)%7 + 0 ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((7*me + 4)/7)*14 + (7*me + 4)%7 + 7 ) ] = (*R9).y;
		bufOutIm[outOffset + ( ((7*me + 5)/7)*14 + (7*me + 5)%7 + 0 ) ] = (*R10).y;
		bufOutIm[outOffset + ( ((7*me + 5)/7)*14 + (7*me + 5)%7 + 7 ) ] = (*R11).y;
		bufOutIm[outOffset + ( ((7*me + 6)/7)*14 + (7*me + 6)%7 + 0 ) ] = (*R12).y;
		bufOutIm[outOffset + ( ((7*me + 6)/7)*14 + (7*me + 6)%7 + 7 ) ] = (*R13).y;
	
	__syncthreads();
	
		(*R0).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	}

	__syncthreads();
}

template <typename T >
__device__ inline void
FwdPass2_len112(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	{
		T W = twiddles[13 + 1*((7*me + 0)%14) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[13 + 1*((7*me + 1)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[13 + 1*((7*me + 2)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[13 + 1*((7*me + 3)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[13 + 1*((7*me + 4)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[13 + 1*((7*me + 5)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[13 + 1*((7*me + 6)%14) + 0];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	FwdRad2B1(R0, R1);
	FwdRad2B1(R2, R3);
	FwdRad2B1(R4, R5);
	FwdRad2B1(R6, R7);
	FwdRad2B1(R8, R9);
	FwdRad2B1(R10, R11);
	FwdRad2B1(R12, R13);

	// if(rw)
	{
		bufOutRe[outOffset + ( ((7*me + 0)/14)*28 + (7*me + 0)%14 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((7*me + 0)/14)*28 + (7*me + 0)%14 + 14 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((7*me + 1)/14)*28 + (7*me + 1)%14 + 0 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((7*me + 1)/14)*28 + (7*me + 1)%14 + 14 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((7*me + 2)/14)*28 + (7*me + 2)%14 + 0 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((7*me + 2)/14)*28 + (7*me + 2)%14 + 14 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((7*me + 3)/14)*28 + (7*me + 3)%14 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((7*me + 3)/14)*28 + (7*me + 3)%14 + 14 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((7*me + 4)/14)*28 + (7*me + 4)%14 + 0 ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((7*me + 4)/14)*28 + (7*me + 4)%14 + 14 ) ] = (*R9).x;
		bufOutRe[outOffset + ( ((7*me + 5)/14)*28 + (7*me + 5)%14 + 0 ) ] = (*R10).x;
		bufOutRe[outOffset + ( ((7*me + 5)/14)*28 + (7*me + 5)%14 + 14 ) ] = (*R11).x;
		bufOutRe[outOffset + ( ((7*me + 6)/14)*28 + (7*me + 6)%14 + 0 ) ] = (*R12).x;
		bufOutRe[outOffset + ( ((7*me + 6)/14)*28 + (7*me + 6)%14 + 14 ) ] = (*R13).x;
	
	__syncthreads();
	
		(*R0).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	
	__syncthreads();
	
		bufOutIm[outOffset + ( ((7*me + 0)/14)*28 + (7*me + 0)%14 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((7*me + 0)/14)*28 + (7*me + 0)%14 + 14 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((7*me + 1)/14)*28 + (7*me + 1)%14 + 0 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((7*me + 1)/14)*28 + (7*me + 1)%14 + 14 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((7*me + 2)/14)*28 + (7*me + 2)%14 + 0 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((7*me + 2)/14)*28 + (7*me + 2)%14 + 14 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((7*me + 3)/14)*28 + (7*me + 3)%14 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((7*me + 3)/14)*28 + (7*me + 3)%14 + 14 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((7*me + 4)/14)*28 + (7*me + 4)%14 + 0 ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((7*me + 4)/14)*28 + (7*me + 4)%14 + 14 ) ] = (*R9).y;
		bufOutIm[outOffset + ( ((7*me + 5)/14)*28 + (7*me + 5)%14 + 0 ) ] = (*R10).y;
		bufOutIm[outOffset + ( ((7*me + 5)/14)*28 + (7*me + 5)%14 + 14 ) ] = (*R11).y;
		bufOutIm[outOffset + ( ((7*me + 6)/14)*28 + (7*me + 6)%14 + 0 ) ] = (*R12).y;
		bufOutIm[outOffset + ( ((7*me + 6)/14)*28 + (7*me + 6)%14 + 14 ) ] = (*R13).y;
	
	__syncthreads();
	
		(*R0).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	}

	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass3_len112(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	{
		T W = twiddles[27 + 1*((7*me + 0)%28) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;

		W = twiddles[27 + 1*((7*me + 1)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[27 + 1*((7*me + 2)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[27 + 1*((7*me + 3)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[27 + 1*((7*me + 4)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[27 + 1*((7*me + 5)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[27 + 1*((7*me + 6)%28) + 0];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	FwdRad2B1(R0, R1);
	FwdRad2B1(R2, R3);
	FwdRad2B1(R4, R5);
	FwdRad2B1(R6, R7);
	FwdRad2B1(R8, R9);
	FwdRad2B1(R10, R11);
	FwdRad2B1(R12, R13);

	// if(rw)
	{
		bufOutRe[outOffset + ( ((7*me + 0)/28)*56 + (7*me + 0)%28 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((7*me + 0)/28)*56 + (7*me + 0)%28 + 28 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((7*me + 1)/28)*56 + (7*me + 1)%28 + 0 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((7*me + 1)/28)*56 + (7*me + 1)%28 + 28 ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((7*me + 2)/28)*56 + (7*me + 2)%28 + 0 ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((7*me + 2)/28)*56 + (7*me + 2)%28 + 28 ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((7*me + 3)/28)*56 + (7*me + 3)%28 + 0 ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((7*me + 3)/28)*56 + (7*me + 3)%28 + 28 ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((7*me + 4)/28)*56 + (7*me + 4)%28 + 0 ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((7*me + 4)/28)*56 + (7*me + 4)%28 + 28 ) ] = (*R9).x;
		bufOutRe[outOffset + ( ((7*me + 5)/28)*56 + (7*me + 5)%28 + 0 ) ] = (*R10).x;
		bufOutRe[outOffset + ( ((7*me + 5)/28)*56 + (7*me + 5)%28 + 28 ) ] = (*R11).x;
		bufOutRe[outOffset + ( ((7*me + 6)/28)*56 + (7*me + 6)%28 + 0 ) ] = (*R12).x;
		bufOutRe[outOffset + ( ((7*me + 6)/28)*56 + (7*me + 6)%28 + 28 ) ] = (*R13).x;
	
	__syncthreads();
	
		(*R0).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).x = bufOutRe[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).x = bufOutRe[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).x = bufOutRe[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	
	__syncthreads();
	
		bufOutIm[outOffset + ( ((7*me + 0)/28)*56 + (7*me + 0)%28 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((7*me + 0)/28)*56 + (7*me + 0)%28 + 28 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((7*me + 1)/28)*56 + (7*me + 1)%28 + 0 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((7*me + 1)/28)*56 + (7*me + 1)%28 + 28 ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((7*me + 2)/28)*56 + (7*me + 2)%28 + 0 ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((7*me + 2)/28)*56 + (7*me + 2)%28 + 28 ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((7*me + 3)/28)*56 + (7*me + 3)%28 + 0 ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((7*me + 3)/28)*56 + (7*me + 3)%28 + 28 ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((7*me + 4)/28)*56 + (7*me + 4)%28 + 0 ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((7*me + 4)/28)*56 + (7*me + 4)%28 + 28 ) ] = (*R9).y;
		bufOutIm[outOffset + ( ((7*me + 5)/28)*56 + (7*me + 5)%28 + 0 ) ] = (*R10).y;
		bufOutIm[outOffset + ( ((7*me + 5)/28)*56 + (7*me + 5)%28 + 28 ) ] = (*R11).y;
		bufOutIm[outOffset + ( ((7*me + 6)/28)*56 + (7*me + 6)%28 + 0 ) ] = (*R12).y;
		bufOutIm[outOffset + ( ((7*me + 6)/28)*56 + (7*me + 6)%28 + 28 ) ] = (*R13).y;
	
	__syncthreads();
	
		(*R0).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 0 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 0 ) ];
		(*R4).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 0 ) ];
		(*R6).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 0 ) ];
		(*R8).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 0 ) ];
		(*R10).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 0 ) ];
		(*R12).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).y = bufOutIm[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).y = bufOutIm[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).y = bufOutIm[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	}

	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass123_len112(unsigned int loopcnt,
				const T *twiddles, 
				const size_t stride_in, const size_t stride_out,
				unsigned int rw, unsigned int b, unsigned int me, 
				unsigned int inOffset, unsigned int outOffset, 
				real_type_t<T> *bufInRe,  real_type_t<T> *bufInIm,
				real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
				T *R0, T *R1, T *R2, T *R3,  T *R4,  T *R5,  T *R6, 
				T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	{
		unsigned int bias = (1 << loopcnt)*7; // 0=6; 1=13; 2=27

		T W = twiddles[bias-1 + 1*((7*me + 0)%bias) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;

		W = twiddles[bias-1 + 1*((7*me + 1)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[bias-1 + 1*((7*me + 2)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[bias-1 + 1*((7*me + 3)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[bias-1 + 1*((7*me + 4)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[bias-1 + 1*((7*me + 5)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[bias-1 + 1*((7*me + 6)%bias) + 0];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	FwdRad2B1(R0, R1);
	FwdRad2B1(R2, R3);
	FwdRad2B1(R4, R5);
	FwdRad2B1(R6, R7);
	FwdRad2B1(R8, R9);
	FwdRad2B1(R10, R11);
	FwdRad2B1(R12, R13);

	// if(rw)
	{
		unsigned int bias = (1 << loopcnt)*7; // 0=7; 1=14; 2=28

		bufOutRe[outOffset + ( ((7*me + 0)/bias)*bias*2 + (7*me + 0)%bias + 0    ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((7*me + 0)/bias)*bias*2 + (7*me + 0)%bias + bias ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((7*me + 1)/bias)*bias*2 + (7*me + 1)%bias + 0    ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((7*me + 1)/bias)*bias*2 + (7*me + 1)%bias + bias ) ] = (*R3).x;
		bufOutRe[outOffset + ( ((7*me + 2)/bias)*bias*2 + (7*me + 2)%bias + 0    ) ] = (*R4).x;
		bufOutRe[outOffset + ( ((7*me + 2)/bias)*bias*2 + (7*me + 2)%bias + bias ) ] = (*R5).x;
		bufOutRe[outOffset + ( ((7*me + 3)/bias)*bias*2 + (7*me + 3)%bias + 0    ) ] = (*R6).x;
		bufOutRe[outOffset + ( ((7*me + 3)/bias)*bias*2 + (7*me + 3)%bias + bias ) ] = (*R7).x;
		bufOutRe[outOffset + ( ((7*me + 4)/bias)*bias*2 + (7*me + 4)%bias + 0    ) ] = (*R8).x;
		bufOutRe[outOffset + ( ((7*me + 4)/bias)*bias*2 + (7*me + 4)%bias + bias ) ] = (*R9).x;
		bufOutRe[outOffset + ( ((7*me + 5)/bias)*bias*2 + (7*me + 5)%bias + 0    ) ] = (*R10).x;
		bufOutRe[outOffset + ( ((7*me + 5)/bias)*bias*2 + (7*me + 5)%bias + bias ) ] = (*R11).x;
		bufOutRe[outOffset + ( ((7*me + 6)/bias)*bias*2 + (7*me + 6)%bias + 0    ) ] = (*R12).x;
		bufOutRe[outOffset + ( ((7*me + 6)/bias)*bias*2 + (7*me + 6)%bias + bias ) ] = (*R13).x;
	
	__syncthreads();
	
		(*R0).x  = bufOutRe[outOffset + ( 0 + me*7 + 0 + 0  ) ];
		(*R2).x  = bufOutRe[outOffset + ( 0 + me*7 + 1 + 0  ) ];
		(*R4).x  = bufOutRe[outOffset + ( 0 + me*7 + 2 + 0  ) ];
		(*R6).x  = bufOutRe[outOffset + ( 0 + me*7 + 3 + 0  ) ];
		(*R8).x  = bufOutRe[outOffset + ( 0 + me*7 + 4 + 0  ) ];
		(*R10).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 0  ) ];
		(*R12).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 0  ) ];
		(*R1).x  = bufOutRe[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).x  = bufOutRe[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).x  = bufOutRe[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).x  = bufOutRe[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).x  = bufOutRe[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).x = bufOutRe[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).x = bufOutRe[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	
	__syncthreads();
	
		bufOutIm[outOffset + ( ((7*me + 0)/bias)*bias*2 + (7*me + 0)%bias + 0    ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((7*me + 0)/bias)*bias*2 + (7*me + 0)%bias + bias ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((7*me + 1)/bias)*bias*2 + (7*me + 1)%bias + 0    ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((7*me + 1)/bias)*bias*2 + (7*me + 1)%bias + bias ) ] = (*R3).y;
		bufOutIm[outOffset + ( ((7*me + 2)/bias)*bias*2 + (7*me + 2)%bias + 0    ) ] = (*R4).y;
		bufOutIm[outOffset + ( ((7*me + 2)/bias)*bias*2 + (7*me + 2)%bias + bias ) ] = (*R5).y;
		bufOutIm[outOffset + ( ((7*me + 3)/bias)*bias*2 + (7*me + 3)%bias + 0    ) ] = (*R6).y;
		bufOutIm[outOffset + ( ((7*me + 3)/bias)*bias*2 + (7*me + 3)%bias + bias ) ] = (*R7).y;
		bufOutIm[outOffset + ( ((7*me + 4)/bias)*bias*2 + (7*me + 4)%bias + 0    ) ] = (*R8).y;
		bufOutIm[outOffset + ( ((7*me + 4)/bias)*bias*2 + (7*me + 4)%bias + bias ) ] = (*R9).y;
		bufOutIm[outOffset + ( ((7*me + 5)/bias)*bias*2 + (7*me + 5)%bias + 0    ) ] = (*R10).y;
		bufOutIm[outOffset + ( ((7*me + 5)/bias)*bias*2 + (7*me + 5)%bias + bias ) ] = (*R11).y;
		bufOutIm[outOffset + ( ((7*me + 6)/bias)*bias*2 + (7*me + 6)%bias + 0    ) ] = (*R12).y;
		bufOutIm[outOffset + ( ((7*me + 6)/bias)*bias*2 + (7*me + 6)%bias + bias ) ] = (*R13).y;
	
	__syncthreads();
	
		(*R0).y  = bufOutIm[outOffset + ( 0 + me*7 + 0 + 0  ) ];
		(*R2).y  = bufOutIm[outOffset + ( 0 + me*7 + 1 + 0  ) ];
		(*R4).y  = bufOutIm[outOffset + ( 0 + me*7 + 2 + 0  ) ];
		(*R6).y  = bufOutIm[outOffset + ( 0 + me*7 + 3 + 0  ) ];
		(*R8).y  = bufOutIm[outOffset + ( 0 + me*7 + 4 + 0  ) ];
		(*R10).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 0  ) ];
		(*R12).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 0  ) ];
		(*R1).y  = bufOutIm[outOffset + ( 0 + me*7 + 0 + 56 ) ];
		(*R3).y  = bufOutIm[outOffset + ( 0 + me*7 + 1 + 56 ) ];
		(*R5).y  = bufOutIm[outOffset + ( 0 + me*7 + 2 + 56 ) ];
		(*R7).y  = bufOutIm[outOffset + ( 0 + me*7 + 3 + 56 ) ];
		(*R9).y  = bufOutIm[outOffset + ( 0 + me*7 + 4 + 56 ) ];
		(*R11).y = bufOutIm[outOffset + ( 0 + me*7 + 5 + 56 ) ];
		(*R13).y = bufOutIm[outOffset + ( 0 + me*7 + 6 + 56 ) ];
	}

	__syncthreads();

}

template <typename T >
__device__ inline void
FwdPass4_len112(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13)
{
	{
		T W = twiddles[55 + 1*((7*me + 0)%56) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[55 + 1*((7*me + 1)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[55 + 1*((7*me + 2)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
		
		W = twiddles[55 + 1*((7*me + 3)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[55 + 1*((7*me + 4)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[55 + 1*((7*me + 5)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[55 + 1*((7*me + 6)%56) + 0];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	FwdRad2B1(R0, R1);
	FwdRad2B1(R2, R3);
	FwdRad2B1(R4, R5);
	FwdRad2B1(R6, R7);
	FwdRad2B1(R8, R9);
	FwdRad2B1(R10, R11);
	FwdRad2B1(R12, R13);

	// if(rw)
	{
		bufOut[outOffset + ( 7*me + 0 + 0 )*stride_out] = (*R0);
		bufOut[outOffset + ( 7*me + 1 + 0 )*stride_out] = (*R2);
		bufOut[outOffset + ( 7*me + 2 + 0 )*stride_out] = (*R4);
		bufOut[outOffset + ( 7*me + 3 + 0 )*stride_out] = (*R6);
		bufOut[outOffset + ( 7*me + 4 + 0 )*stride_out] = (*R8);
		bufOut[outOffset + ( 7*me + 5 + 0 )*stride_out] = (*R10);
		bufOut[outOffset + ( 7*me + 6 + 0 )*stride_out] = (*R12);
		bufOut[outOffset + ( 7*me + 0 + 56 )*stride_out] = (*R1);
		bufOut[outOffset + ( 7*me + 1 + 56 )*stride_out] = (*R3);
		bufOut[outOffset + ( 7*me + 2 + 56 )*stride_out] = (*R5);
		bufOut[outOffset + ( 7*me + 3 + 56 )*stride_out] = (*R7);
		bufOut[outOffset + ( 7*me + 4 + 56 )*stride_out] = (*R9);
		bufOut[outOffset + ( 7*me + 5 + 56 )*stride_out] = (*R11);
		bufOut[outOffset + ( 7*me + 6 + 56 )*stride_out] = (*R13);
	}

}

////////////////////////////////////////Encapsulated passes kernels
template <typename T >
__device__ inline void 
fwd_len112_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13;
	FwdPass0_len112<T>(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, 				&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
	
	for(uint32_t i = 0; i < 3; i++)
		FwdPass123_len112<T>(i, twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
	
	//FwdPass1_len112<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
	//FwdPass2_len112<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
	//FwdPass3_len112<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
	
	FwdPass4_len112<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, 			&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);
}

////////////////////////////////////////Global kernels
//Kernel configuration: number of threads per thread block: 64, maximum transforms: 8, Passes: 5
__global__ void 
my_fft_fwd_op_len112( const float2 * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, float2 * __restrict__ gbIn, float2 * __restrict__ gbOut)
{

	__shared__ float lds[896];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int upper_count = batch_count;
	/*for(int i=1; i<dim; i++)
	{
		upper_count *= lengths[i];
	}*/
	// do signed math to guard against underflow
	//unsigned int rw = (static_cast<int>(me) < (static_cast<int>(upper_count)  - static_cast<int>(batch)*8)*8) ? 1 : 0;
	unsigned int rw = 1; // only for fft batch = times of 8

	unsigned int b = 0;

	size_t counter_mod = (batch*8 + (me/8));
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
	fwd_len112_device<float2>(twiddles, stride_in[0], stride_out[0],  rw, b, me%8, (me/8)*112, lwbIn, lwbOut, lds);
}
