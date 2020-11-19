#pragma once
#include "rocfft_butterfly_template.h"


////////////////////////////////////////Passes kernels
template <typename T>
__device__ inline void
FwdPass0_len3125(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15, T *R16, T *R17, T *R18, T *R19, T *R20, T *R21, T *R22, T *R23, T *R24)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*5 + 0 + 0 )*stride_in];
	(*R5) = bufIn[inOffset + ( 0 + me*5 + 1 + 0 )*stride_in];
	(*R10) = bufIn[inOffset + ( 0 + me*5 + 2 + 0 )*stride_in];
	(*R15) = bufIn[inOffset + ( 0 + me*5 + 3 + 0 )*stride_in];
	(*R20) = bufIn[inOffset + ( 0 + me*5 + 4 + 0 )*stride_in];
	(*R1) = bufIn[inOffset + ( 0 + me*5 + 0 + 625 )*stride_in];
	(*R6) = bufIn[inOffset + ( 0 + me*5 + 1 + 625 )*stride_in];
	(*R11) = bufIn[inOffset + ( 0 + me*5 + 2 + 625 )*stride_in];
	(*R16) = bufIn[inOffset + ( 0 + me*5 + 3 + 625 )*stride_in];
	(*R21) = bufIn[inOffset + ( 0 + me*5 + 4 + 625 )*stride_in];
	(*R2) = bufIn[inOffset + ( 0 + me*5 + 0 + 1250 )*stride_in];
	(*R7) = bufIn[inOffset + ( 0 + me*5 + 1 + 1250 )*stride_in];
	(*R12) = bufIn[inOffset + ( 0 + me*5 + 2 + 1250 )*stride_in];
	(*R17) = bufIn[inOffset + ( 0 + me*5 + 3 + 1250 )*stride_in];
	(*R22) = bufIn[inOffset + ( 0 + me*5 + 4 + 1250 )*stride_in];
	(*R3) = bufIn[inOffset + ( 0 + me*5 + 0 + 1875 )*stride_in];
	(*R8) = bufIn[inOffset + ( 0 + me*5 + 1 + 1875 )*stride_in];
	(*R13) = bufIn[inOffset + ( 0 + me*5 + 2 + 1875 )*stride_in];
	(*R18) = bufIn[inOffset + ( 0 + me*5 + 3 + 1875 )*stride_in];
	(*R23) = bufIn[inOffset + ( 0 + me*5 + 4 + 1875 )*stride_in];
	(*R4) = bufIn[inOffset + ( 0 + me*5 + 0 + 2500 )*stride_in];
	(*R9) = bufIn[inOffset + ( 0 + me*5 + 1 + 2500 )*stride_in];
	(*R14) = bufIn[inOffset + ( 0 + me*5 + 2 + 2500 )*stride_in];
	(*R19) = bufIn[inOffset + ( 0 + me*5 + 3 + 2500 )*stride_in];
	(*R24) = bufIn[inOffset + ( 0 + me*5 + 4 + 2500 )*stride_in];
	}



	FwdRad5B1(R0, R1, R2, R3, R4);
	FwdRad5B1(R5, R6, R7, R8, R9);
	FwdRad5B1(R10, R11, R12, R13, R14);
	FwdRad5B1(R15, R16, R17, R18, R19);
	FwdRad5B1(R20, R21, R22, R23, R24);


	if(rw)
	{
	bufOutRe[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 1 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 2 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 3 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 4 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 0 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 1 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 2 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 3 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 4 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 0 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 1 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 2 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 3 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 4 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 0 ) ] = (*R15).x;
	bufOutRe[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 1 ) ] = (*R16).x;
	bufOutRe[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 2 ) ] = (*R17).x;
	bufOutRe[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 3 ) ] = (*R18).x;
	bufOutRe[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 4 ) ] = (*R19).x;
	bufOutRe[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 0 ) ] = (*R20).x;
	bufOutRe[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 1 ) ] = (*R21).x;
	bufOutRe[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 2 ) ] = (*R22).x;
	bufOutRe[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 3 ) ] = (*R23).x;
	bufOutRe[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 4 ) ] = (*R24).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 1 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 2 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 3 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((5*me + 0)/1)*5 + (5*me + 0)%1 + 4 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 0 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 1 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 2 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 3 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((5*me + 1)/1)*5 + (5*me + 1)%1 + 4 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 0 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 1 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 2 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 3 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((5*me + 2)/1)*5 + (5*me + 2)%1 + 4 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 0 ) ] = (*R15).y;
	bufOutIm[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 1 ) ] = (*R16).y;
	bufOutIm[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 2 ) ] = (*R17).y;
	bufOutIm[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 3 ) ] = (*R18).y;
	bufOutIm[outOffset + ( ((5*me + 3)/1)*5 + (5*me + 3)%1 + 4 ) ] = (*R19).y;
	bufOutIm[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 0 ) ] = (*R20).y;
	bufOutIm[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 1 ) ] = (*R21).y;
	bufOutIm[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 2 ) ] = (*R22).y;
	bufOutIm[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 3 ) ] = (*R23).y;
	bufOutIm[outOffset + ( ((5*me + 4)/1)*5 + (5*me + 4)%1 + 4 ) ] = (*R24).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

}

template <typename T>
__device__ inline void
FwdPass1_len3125(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15, T *R16, T *R17, T *R18, T *R19, T *R20, T *R21, T *R22, T *R23, T *R24)
{




	{
		T W = twiddles[4 + 4*((5*me + 0)%5) + 0];
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
		T W = twiddles[4 + 4*((5*me + 0)%5) + 1];
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
		T W = twiddles[4 + 4*((5*me + 0)%5) + 2];
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
		T W = twiddles[4 + 4*((5*me + 0)%5) + 3];
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
		T W = twiddles[4 + 4*((5*me + 1)%5) + 0];
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
		T W = twiddles[4 + 4*((5*me + 1)%5) + 1];
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
		T W = twiddles[4 + 4*((5*me + 1)%5) + 2];
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
		T W = twiddles[4 + 4*((5*me + 1)%5) + 3];
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
		T W = twiddles[4 + 4*((5*me + 2)%5) + 0];
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
		T W = twiddles[4 + 4*((5*me + 2)%5) + 1];
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
		T W = twiddles[4 + 4*((5*me + 2)%5) + 2];
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
		T W = twiddles[4 + 4*((5*me + 2)%5) + 3];
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
		T W = twiddles[4 + 4*((5*me + 3)%5) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R16).x; ry = (*R16).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R16).x = TR;
		(*R16).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 3)%5) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R17).x; ry = (*R17).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R17).x = TR;
		(*R17).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 3)%5) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R18).x; ry = (*R18).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R18).x = TR;
		(*R18).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 3)%5) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R19).x; ry = (*R19).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R19).x = TR;
		(*R19).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 4)%5) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R21).x; ry = (*R21).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R21).x = TR;
		(*R21).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 4)%5) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R22).x; ry = (*R22).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R22).x = TR;
		(*R22).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 4)%5) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R23).x; ry = (*R23).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R23).x = TR;
		(*R23).y = TI;
	}

	{
		T W = twiddles[4 + 4*((5*me + 4)%5) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R24).x; ry = (*R24).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R24).x = TR;
		(*R24).y = TI;
	}

	FwdRad5B1(R0, R1, R2, R3, R4);
	FwdRad5B1(R5, R6, R7, R8, R9);
	FwdRad5B1(R10, R11, R12, R13, R14);
	FwdRad5B1(R15, R16, R17, R18, R19);
	FwdRad5B1(R20, R21, R22, R23, R24);


	if(rw)
	{
	bufOutRe[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 5 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 10 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 15 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 20 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 0 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 5 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 10 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 15 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 20 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 0 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 5 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 10 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 15 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 20 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 0 ) ] = (*R15).x;
	bufOutRe[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 5 ) ] = (*R16).x;
	bufOutRe[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 10 ) ] = (*R17).x;
	bufOutRe[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 15 ) ] = (*R18).x;
	bufOutRe[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 20 ) ] = (*R19).x;
	bufOutRe[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 0 ) ] = (*R20).x;
	bufOutRe[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 5 ) ] = (*R21).x;
	bufOutRe[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 10 ) ] = (*R22).x;
	bufOutRe[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 15 ) ] = (*R23).x;
	bufOutRe[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 20 ) ] = (*R24).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 5 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 10 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 15 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((5*me + 0)/5)*25 + (5*me + 0)%5 + 20 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 0 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 5 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 10 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 15 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((5*me + 1)/5)*25 + (5*me + 1)%5 + 20 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 0 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 5 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 10 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 15 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((5*me + 2)/5)*25 + (5*me + 2)%5 + 20 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 0 ) ] = (*R15).y;
	bufOutIm[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 5 ) ] = (*R16).y;
	bufOutIm[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 10 ) ] = (*R17).y;
	bufOutIm[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 15 ) ] = (*R18).y;
	bufOutIm[outOffset + ( ((5*me + 3)/5)*25 + (5*me + 3)%5 + 20 ) ] = (*R19).y;
	bufOutIm[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 0 ) ] = (*R20).y;
	bufOutIm[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 5 ) ] = (*R21).y;
	bufOutIm[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 10 ) ] = (*R22).y;
	bufOutIm[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 15 ) ] = (*R23).y;
	bufOutIm[outOffset + ( ((5*me + 4)/5)*25 + (5*me + 4)%5 + 20 ) ] = (*R24).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

}

template <typename T>
__device__ inline void
FwdPass2_len3125(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15, T *R16, T *R17, T *R18, T *R19, T *R20, T *R21, T *R22, T *R23, T *R24)
{




	{
		T W = twiddles[24 + 4*((5*me + 0)%25) + 0];
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
		T W = twiddles[24 + 4*((5*me + 0)%25) + 1];
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
		T W = twiddles[24 + 4*((5*me + 0)%25) + 2];
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
		T W = twiddles[24 + 4*((5*me + 0)%25) + 3];
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
		T W = twiddles[24 + 4*((5*me + 1)%25) + 0];
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
		T W = twiddles[24 + 4*((5*me + 1)%25) + 1];
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
		T W = twiddles[24 + 4*((5*me + 1)%25) + 2];
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
		T W = twiddles[24 + 4*((5*me + 1)%25) + 3];
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
		T W = twiddles[24 + 4*((5*me + 2)%25) + 0];
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
		T W = twiddles[24 + 4*((5*me + 2)%25) + 1];
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
		T W = twiddles[24 + 4*((5*me + 2)%25) + 2];
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
		T W = twiddles[24 + 4*((5*me + 2)%25) + 3];
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
		T W = twiddles[24 + 4*((5*me + 3)%25) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R16).x; ry = (*R16).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R16).x = TR;
		(*R16).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 3)%25) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R17).x; ry = (*R17).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R17).x = TR;
		(*R17).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 3)%25) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R18).x; ry = (*R18).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R18).x = TR;
		(*R18).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 3)%25) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R19).x; ry = (*R19).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R19).x = TR;
		(*R19).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 4)%25) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R21).x; ry = (*R21).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R21).x = TR;
		(*R21).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 4)%25) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R22).x; ry = (*R22).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R22).x = TR;
		(*R22).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 4)%25) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R23).x; ry = (*R23).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R23).x = TR;
		(*R23).y = TI;
	}

	{
		T W = twiddles[24 + 4*((5*me + 4)%25) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R24).x; ry = (*R24).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R24).x = TR;
		(*R24).y = TI;
	}

	FwdRad5B1(R0, R1, R2, R3, R4);
	FwdRad5B1(R5, R6, R7, R8, R9);
	FwdRad5B1(R10, R11, R12, R13, R14);
	FwdRad5B1(R15, R16, R17, R18, R19);
	FwdRad5B1(R20, R21, R22, R23, R24);


	if(rw)
	{
	bufOutRe[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 25 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 50 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 75 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 100 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 0 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 25 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 50 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 75 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 100 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 0 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 25 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 50 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 75 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 100 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 0 ) ] = (*R15).x;
	bufOutRe[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 25 ) ] = (*R16).x;
	bufOutRe[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 50 ) ] = (*R17).x;
	bufOutRe[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 75 ) ] = (*R18).x;
	bufOutRe[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 100 ) ] = (*R19).x;
	bufOutRe[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 0 ) ] = (*R20).x;
	bufOutRe[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 25 ) ] = (*R21).x;
	bufOutRe[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 50 ) ] = (*R22).x;
	bufOutRe[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 75 ) ] = (*R23).x;
	bufOutRe[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 100 ) ] = (*R24).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 25 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 50 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 75 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((5*me + 0)/25)*125 + (5*me + 0)%25 + 100 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 0 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 25 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 50 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 75 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((5*me + 1)/25)*125 + (5*me + 1)%25 + 100 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 0 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 25 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 50 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 75 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((5*me + 2)/25)*125 + (5*me + 2)%25 + 100 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 0 ) ] = (*R15).y;
	bufOutIm[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 25 ) ] = (*R16).y;
	bufOutIm[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 50 ) ] = (*R17).y;
	bufOutIm[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 75 ) ] = (*R18).y;
	bufOutIm[outOffset + ( ((5*me + 3)/25)*125 + (5*me + 3)%25 + 100 ) ] = (*R19).y;
	bufOutIm[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 0 ) ] = (*R20).y;
	bufOutIm[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 25 ) ] = (*R21).y;
	bufOutIm[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 50 ) ] = (*R22).y;
	bufOutIm[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 75 ) ] = (*R23).y;
	bufOutIm[outOffset + ( ((5*me + 4)/25)*125 + (5*me + 4)%25 + 100 ) ] = (*R24).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

}

template <typename T>
__device__ inline void
FwdPass3_len3125(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15, T *R16, T *R17, T *R18, T *R19, T *R20, T *R21, T *R22, T *R23, T *R24)
{




	{
		T W = twiddles[124 + 4*((5*me + 0)%125) + 0];
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
		T W = twiddles[124 + 4*((5*me + 0)%125) + 1];
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
		T W = twiddles[124 + 4*((5*me + 0)%125) + 2];
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
		T W = twiddles[124 + 4*((5*me + 0)%125) + 3];
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
		T W = twiddles[124 + 4*((5*me + 1)%125) + 0];
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
		T W = twiddles[124 + 4*((5*me + 1)%125) + 1];
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
		T W = twiddles[124 + 4*((5*me + 1)%125) + 2];
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
		T W = twiddles[124 + 4*((5*me + 1)%125) + 3];
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
		T W = twiddles[124 + 4*((5*me + 2)%125) + 0];
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
		T W = twiddles[124 + 4*((5*me + 2)%125) + 1];
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
		T W = twiddles[124 + 4*((5*me + 2)%125) + 2];
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
		T W = twiddles[124 + 4*((5*me + 2)%125) + 3];
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
		T W = twiddles[124 + 4*((5*me + 3)%125) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R16).x; ry = (*R16).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R16).x = TR;
		(*R16).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 3)%125) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R17).x; ry = (*R17).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R17).x = TR;
		(*R17).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 3)%125) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R18).x; ry = (*R18).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R18).x = TR;
		(*R18).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 3)%125) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R19).x; ry = (*R19).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R19).x = TR;
		(*R19).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 4)%125) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R21).x; ry = (*R21).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R21).x = TR;
		(*R21).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 4)%125) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R22).x; ry = (*R22).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R22).x = TR;
		(*R22).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 4)%125) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R23).x; ry = (*R23).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R23).x = TR;
		(*R23).y = TI;
	}

	{
		T W = twiddles[124 + 4*((5*me + 4)%125) + 3];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R24).x; ry = (*R24).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R24).x = TR;
		(*R24).y = TI;
	}

	FwdRad5B1(R0, R1, R2, R3, R4);
	FwdRad5B1(R5, R6, R7, R8, R9);
	FwdRad5B1(R10, R11, R12, R13, R14);
	FwdRad5B1(R15, R16, R17, R18, R19);
	FwdRad5B1(R20, R21, R22, R23, R24);


	if(rw)
	{
	bufOutRe[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 125 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 250 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 375 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 500 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 0 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 125 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 250 ) ] = (*R7).x;
	bufOutRe[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 375 ) ] = (*R8).x;
	bufOutRe[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 500 ) ] = (*R9).x;
	bufOutRe[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 0 ) ] = (*R10).x;
	bufOutRe[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 125 ) ] = (*R11).x;
	bufOutRe[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 250 ) ] = (*R12).x;
	bufOutRe[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 375 ) ] = (*R13).x;
	bufOutRe[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 500 ) ] = (*R14).x;
	bufOutRe[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 0 ) ] = (*R15).x;
	bufOutRe[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 125 ) ] = (*R16).x;
	bufOutRe[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 250 ) ] = (*R17).x;
	bufOutRe[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 375 ) ] = (*R18).x;
	bufOutRe[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 500 ) ] = (*R19).x;
	bufOutRe[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 0 ) ] = (*R20).x;
	bufOutRe[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 125 ) ] = (*R21).x;
	bufOutRe[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 250 ) ] = (*R22).x;
	bufOutRe[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 375 ) ] = (*R23).x;
	bufOutRe[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 500 ) ] = (*R24).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).x = bufOutRe[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).x = bufOutRe[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).x = bufOutRe[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).x = bufOutRe[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 125 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 250 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 375 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((5*me + 0)/125)*625 + (5*me + 0)%125 + 500 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 0 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 125 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 250 ) ] = (*R7).y;
	bufOutIm[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 375 ) ] = (*R8).y;
	bufOutIm[outOffset + ( ((5*me + 1)/125)*625 + (5*me + 1)%125 + 500 ) ] = (*R9).y;
	bufOutIm[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 0 ) ] = (*R10).y;
	bufOutIm[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 125 ) ] = (*R11).y;
	bufOutIm[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 250 ) ] = (*R12).y;
	bufOutIm[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 375 ) ] = (*R13).y;
	bufOutIm[outOffset + ( ((5*me + 2)/125)*625 + (5*me + 2)%125 + 500 ) ] = (*R14).y;
	bufOutIm[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 0 ) ] = (*R15).y;
	bufOutIm[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 125 ) ] = (*R16).y;
	bufOutIm[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 250 ) ] = (*R17).y;
	bufOutIm[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 375 ) ] = (*R18).y;
	bufOutIm[outOffset + ( ((5*me + 3)/125)*625 + (5*me + 3)%125 + 500 ) ] = (*R19).y;
	bufOutIm[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 0 ) ] = (*R20).y;
	bufOutIm[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 125 ) ] = (*R21).y;
	bufOutIm[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 250 ) ] = (*R22).y;
	bufOutIm[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 375 ) ] = (*R23).y;
	bufOutIm[outOffset + ( ((5*me + 4)/125)*625 + (5*me + 4)%125 + 500 ) ] = (*R24).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 0 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 0 ) ];
	(*R10).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 0 ) ];
	(*R15).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 0 ) ];
	(*R20).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 625 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 625 ) ];
	(*R11).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 625 ) ];
	(*R16).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 625 ) ];
	(*R21).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 625 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1250 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1250 ) ];
	(*R12).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1250 ) ];
	(*R17).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1250 ) ];
	(*R22).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1250 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 1875 ) ];
	(*R8).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 1875 ) ];
	(*R13).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 1875 ) ];
	(*R18).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 1875 ) ];
	(*R23).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 1875 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*5 + 0 + 2500 ) ];
	(*R9).y = bufOutIm[outOffset + ( 0 + me*5 + 1 + 2500 ) ];
	(*R14).y = bufOutIm[outOffset + ( 0 + me*5 + 2 + 2500 ) ];
	(*R19).y = bufOutIm[outOffset + ( 0 + me*5 + 3 + 2500 ) ];
	(*R24).y = bufOutIm[outOffset + ( 0 + me*5 + 4 + 2500 ) ];
	}


	__syncthreads();

}

template <typename T>
__device__ inline void
FwdPass4_len3125(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15, T *R16, T *R17, T *R18, T *R19, T *R20, T *R21, T *R22, T *R23, T *R24)
{
	{
		T W = twiddles[624 + 4*((5*me + 0)%625) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
		
		W = twiddles[624 + 4*((5*me + 0)%625) + 1];
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
		
		W = twiddles[624 + 4*((5*me + 0)%625) + 2];
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
		
		W = twiddles[624 + 4*((5*me + 0)%625) + 3];
		wx = W.x; wy = W.y;
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
		
		W = twiddles[624 + 4*((5*me + 1)%625) + 0];
		wx = W.x; wy = W.y;
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
		
		W = twiddles[624 + 4*((5*me + 1)%625) + 1];
		wx = W.x; wy = W.y;
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
		
		W = twiddles[624 + 4*((5*me + 1)%625) + 2];
		wx = W.x; wy = W.y;
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
		
		W = twiddles[624 + 4*((5*me + 1)%625) + 3];
		wx = W.x; wy = W.y;
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
		
		W = twiddles[624 + 4*((5*me + 2)%625) + 0];
		wx = W.x; wy = W.y;
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
		
		W = twiddles[624 + 4*((5*me + 2)%625) + 1];
		wx = W.x; wy = W.y;
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
		
		W = twiddles[624 + 4*((5*me + 2)%625) + 2];
		wx = W.x; wy = W.y;
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
		
		W = twiddles[624 + 4*((5*me + 2)%625) + 3];
		wx = W.x; wy = W.y;
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
		
		W = twiddles[624 + 4*((5*me + 3)%625) + 0];
		wx = W.x; wy = W.y;
		rx = (*R16).x; ry = (*R16).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R16).x = TR;
		(*R16).y = TI;
		
		W = twiddles[624 + 4*((5*me + 3)%625) + 1];
		wx = W.x; wy = W.y;
		rx = (*R17).x; ry = (*R17).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R17).x = TR;
		(*R17).y = TI;
		
		W = twiddles[624 + 4*((5*me + 3)%625) + 2];
		wx = W.x; wy = W.y;
		rx = (*R18).x; ry = (*R18).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R18).x = TR;
		(*R18).y = TI;
		
		W = twiddles[624 + 4*((5*me + 3)%625) + 3];
		wx = W.x; wy = W.y;
		rx = (*R19).x; ry = (*R19).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R19).x = TR;
		(*R19).y = TI;
		
		W = twiddles[624 + 4*((5*me + 4)%625) + 0];
		wx = W.x; wy = W.y;
		rx = (*R21).x; ry = (*R21).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R21).x = TR;
		(*R21).y = TI;
		
		W = twiddles[624 + 4*((5*me + 4)%625) + 1];
		wx = W.x; wy = W.y;
		rx = (*R22).x; ry = (*R22).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R22).x = TR;
		(*R22).y = TI;
		
		W = twiddles[624 + 4*((5*me + 4)%625) + 2];
		wx = W.x; wy = W.y;
		rx = (*R23).x; ry = (*R23).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R23).x = TR;
		(*R23).y = TI;
		
		W = twiddles[624 + 4*((5*me + 4)%625) + 3];
		wx = W.x; wy = W.y;
		rx = (*R24).x; ry = (*R24).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R24).x = TR;
		(*R24).y = TI;
	}

	FwdRad5B1(R0, R1, R2, R3, R4);
	FwdRad5B1(R5, R6, R7, R8, R9);
	FwdRad5B1(R10, R11, R12, R13, R14);
	FwdRad5B1(R15, R16, R17, R18, R19);
	FwdRad5B1(R20, R21, R22, R23, R24);


#if 1 // use 120 vgprs
	size_t addr;
	{
		bufOut[outOffset + ( 5*me + 0 + 0 )*stride_out] = (*R0);
		bufOut[outOffset + ( 5*me + 1 + 0 )*stride_out] = (*R5);
		bufOut[outOffset + ( 5*me + 2 + 0 )*stride_out] = (*R10);
		bufOut[outOffset + ( 5*me + 3 + 0 )*stride_out] = (*R15);
		bufOut[outOffset + ( 5*me + 4 + 0 )*stride_out] = (*R20);
		bufOut[outOffset + ( 5*me + 0 + 625 )*stride_out] = (*R1);
		bufOut[outOffset + ( 5*me + 1 + 625 )*stride_out] = (*R6);
		bufOut[outOffset + ( 5*me + 2 + 625 )*stride_out] = (*R11);
		bufOut[outOffset + ( 5*me + 3 + 625 )*stride_out] = (*R16);
		bufOut[outOffset + ( 5*me + 4 + 625 )*stride_out] = (*R21);
		bufOut[outOffset + ( 5*me + 0 + 1250 )*stride_out] = (*R2);
		bufOut[outOffset + ( 5*me + 1 + 1250 )*stride_out] = (*R7);
		bufOut[outOffset + ( 5*me + 2 + 1250 )*stride_out] = (*R12);
		bufOut[outOffset + ( 5*me + 3 + 1250 )*stride_out] = (*R17);
		bufOut[outOffset + ( 5*me + 4 + 1250 )*stride_out] = (*R22);
		bufOut[outOffset + ( 5*me + 0 + 1875 )*stride_out] = (*R3);
		bufOut[outOffset + ( 5*me + 1 + 1875 )*stride_out] = (*R8);
		bufOut[outOffset + ( 5*me + 2 + 1875 )*stride_out] = (*R13);
		bufOut[outOffset + ( 5*me + 3 + 1875 )*stride_out] = (*R18);
		bufOut[outOffset + ( 5*me + 4 + 1875 )*stride_out] = (*R23);
		bufOut[outOffset + ( 5*me + 0 + 2500 )*stride_out] = (*R4);
		bufOut[outOffset + ( 5*me + 1 + 2500 )*stride_out] = (*R9);
		bufOut[outOffset + ( 5*me + 2 + 2500 )*stride_out] = (*R14);
		bufOut[outOffset + ( 5*me + 3 + 2500 )*stride_out] = (*R19);
		bufOut[outOffset + ( 5*me + 4 + 2500 )*stride_out] = (*R24);
	}
#else // not less enough
	size_t addr;
	{
		addr = outOffset + ( 5*me + 0 + 1 )*stride_out   ; bufOut[addr - stride_out] = (*R0);
		addr = outOffset + ( 5*me + 1 + 1 )*stride_out   ; bufOut[addr - stride_out] = (*R5);
		addr = outOffset + ( 5*me + 2 + 1 )*stride_out   ; bufOut[addr - stride_out] = (*R10);
		addr = outOffset + ( 5*me + 3 + 1 )*stride_out   ; bufOut[addr - stride_out] = (*R15);
		addr = outOffset + ( 5*me + 4 + 1 )*stride_out   ; bufOut[addr - stride_out] = (*R20);
		addr = outOffset + ( 5*me + 0 + 626 )*stride_out ; bufOut[addr - stride_out] = (*R1);
		addr = outOffset + ( 5*me + 1 + 626 )*stride_out ; bufOut[addr - stride_out] = (*R6);
		addr = outOffset + ( 5*me + 2 + 626 )*stride_out ; bufOut[addr - stride_out] = (*R11);
		addr = outOffset + ( 5*me + 3 + 626 )*stride_out ; bufOut[addr - stride_out] = (*R16);
		addr = outOffset + ( 5*me + 4 + 626 )*stride_out ; bufOut[addr - stride_out] = (*R21);
		addr = outOffset + ( 5*me + 0 + 1251 )*stride_out; bufOut[addr - stride_out] = (*R2);
		addr = outOffset + ( 5*me + 1 + 1251 )*stride_out; bufOut[addr - stride_out] = (*R7);
		addr = outOffset + ( 5*me + 2 + 1251 )*stride_out; bufOut[addr - stride_out] = (*R12);
		addr = outOffset + ( 5*me + 3 + 1251 )*stride_out; bufOut[addr - stride_out] = (*R17);
		addr = outOffset + ( 5*me + 4 + 1251 )*stride_out; bufOut[addr - stride_out] = (*R22);
		addr = outOffset + ( 5*me + 0 + 1876 )*stride_out; bufOut[addr - stride_out] = (*R3);
		addr = outOffset + ( 5*me + 1 + 1876 )*stride_out; bufOut[addr - stride_out] = (*R8);
		addr = outOffset + ( 5*me + 2 + 1876 )*stride_out; bufOut[addr - stride_out] = (*R13);
		addr = outOffset + ( 5*me + 3 + 1876 )*stride_out; bufOut[addr - stride_out] = (*R18);
		addr = outOffset + ( 5*me + 4 + 1876 )*stride_out; bufOut[addr - stride_out] = (*R23);
		addr = outOffset + ( 5*me + 0 + 2501 )*stride_out; bufOut[addr - stride_out] = (*R4);
		addr = outOffset + ( 5*me + 1 + 2501 )*stride_out; bufOut[addr - stride_out] = (*R9);
		addr = outOffset + ( 5*me + 2 + 2501 )*stride_out; bufOut[addr - stride_out] = (*R14);
		addr = outOffset + ( 5*me + 3 + 2501 )*stride_out; bufOut[addr - stride_out] = (*R19);
		addr = outOffset + ( 5*me + 4 + 2501 )*stride_out; bufOut[addr - stride_out] = (*R24);
	}
#endif

}

////////////////////////////////////////Encapsulated passes kernels
template <typename T>
__device__ inline void 
fwd_len3125_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24;
	FwdPass0_len3125<T>(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, 			&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, &R20, &R21, &R22, &R23, &R24);
	FwdPass1_len3125<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, &R20, &R21, &R22, &R23, &R24);
	FwdPass2_len3125<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, &R20, &R21, &R22, &R23, &R24);
	FwdPass3_len3125<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, 	&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, &R20, &R21, &R22, &R23, &R24);
	FwdPass4_len3125<T>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, 			&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, &R20, &R21, &R22, &R23, &R24);
}

////////////////////////////////////////Global kernels
//Kernel configuration: number of threads per thread block: 125, maximum transforms: 1, Passes: 5
__attribute__((amdgpu_num_vgpr(84))) // could limit to 84, but increase scratch memory
__global__ void 
my_fft_fwd_op_len3125( const float2 * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, float2 * __restrict__ gbIn, float2 * __restrict__ gbOut)
{

	__shared__ float lds[3125];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int rw = 1;
	unsigned int b = 0;

	size_t counter_mod = batch;
	/*if(dim == 1){
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
	lwbOut = gbOut + oOffset;*/
	lwbIn = gbIn;
	lwbOut = gbOut;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len3125_device<float2>(twiddles, stride_in[0], stride_out[0],  1, b, me, 0, lwbIn, lwbOut, lds);
}

