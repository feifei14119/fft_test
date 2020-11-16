#pragma once
#include "rocfft_butterfly_template.h"


////////////////////////////////////////Passes kernels
template <typename T >
__device__ inline void
FwdPass0_len4096(const T *twiddles, 
					const size_t stride_in, const size_t stride_out, 
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, 
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	//if(rw)
	{
		*addr = inOffset + ( 0 + me*1 + 0 + 0 )*stride_in;      (*R0) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 256 )*stride_in;	(*R1) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 512 )*stride_in;	(*R2) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 768 )*stride_in;	(*R3) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 1024 )*stride_in;	(*R4) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 1280 )*stride_in;	(*R5) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 1536 )*stride_in;	(*R6) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 1792 )*stride_in;	(*R7) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 2048 )*stride_in;	(*R8) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 2304 )*stride_in;	(*R9) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 2560 )*stride_in;	(*R10) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 2816 )*stride_in;	(*R11) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 3072 )*stride_in;	(*R12) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 3328 )*stride_in;	(*R13) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 3584 )*stride_in;	(*R14) = bufIn[*addr];
		*addr = inOffset + ( 0 + me*1 + 0 + 3840 )*stride_in;	(*R15) = bufIn[*addr];
	}
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	//if(rw)
	{
		uint32_t a, b, c, t;
		t = me * 1; t = t + 0; t = t / 1;
		a = t * 16;
		b = t % 1;
		c = a + b;
		c = c + outOffset;
		*addr = c + 0;  bufOutRe[*addr] = (*R0).x;
		*addr = c + 1;  bufOutRe[*addr] = (*R1).x;
		*addr = c + 2;  bufOutRe[*addr] = (*R2).x;
		*addr = c + 3;  bufOutRe[*addr] = (*R3).x;
		*addr = c + 4;  bufOutRe[*addr] = (*R4).x;
		*addr = c + 5;  bufOutRe[*addr] = (*R5).x;
		*addr = c + 6;  bufOutRe[*addr] = (*R6).x;
		*addr = c + 7;  bufOutRe[*addr] = (*R7).x;
		*addr = c + 8;  bufOutRe[*addr] = (*R8).x;
		*addr = c + 9;  bufOutRe[*addr] = (*R9).x;
		*addr = c + 10; bufOutRe[*addr] = (*R10).x;
		*addr = c + 11; bufOutRe[*addr] = (*R11).x;
		*addr = c + 12; bufOutRe[*addr] = (*R12).x;
		*addr = c + 13; bufOutRe[*addr] = (*R13).x;
		*addr = c + 14; bufOutRe[*addr] = (*R14).x;
		*addr = c + 15; bufOutRe[*addr] = (*R15).x;
	}

	__syncthreads();

	//if(rw)
	{
		*addr = outOffset + ( 0 + me*1 + 0 + 0 )    ;(*R0).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 256 )  ;(*R1).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 512 )  ;(*R2).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 768 )  ;(*R3).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1024 ) ;(*R4).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1280 ) ;(*R5).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1536 ) ;(*R6).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1792 ) ;(*R7).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2048 ) ;(*R8).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2304 ) ;(*R9).x =  bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2560 ) ;(*R10).x = bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2816 ) ;(*R11).x = bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3072 ) ;(*R12).x = bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3328 ) ;(*R13).x = bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3584 ) ;(*R14).x = bufOutRe[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3840 ) ;(*R15).x = bufOutRe[*addr];
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 1; a = a *16;
		b = t % 1;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;  bufOutRe[*addr ] = (*R0).y;
		*addr = c + 1;  bufOutRe[*addr ] = (*R1).y;
		*addr = c + 2;  bufOutRe[*addr ] = (*R2).y;
		*addr = c + 3;  bufOutRe[*addr ] = (*R3).y;
		*addr = c + 4;  bufOutRe[*addr ] = (*R4).y;
		*addr = c + 5;  bufOutRe[*addr ] = (*R5).y;
		*addr = c + 6;  bufOutRe[*addr ] = (*R6).y;
		*addr = c + 7;  bufOutRe[*addr ] = (*R7).y;
		*addr = c + 8;  bufOutRe[*addr ] = (*R8).y;
		*addr = c + 9;  bufOutRe[*addr ] = (*R9).y;
		*addr = c + 10; bufOutRe[*addr ] = (*R10).y;
		*addr = c + 11; bufOutRe[*addr ] = (*R11).y;
		*addr = c + 12; bufOutRe[*addr ] = (*R12).y;
		*addr = c + 13; bufOutRe[*addr ] = (*R13).y;
		*addr = c + 14; bufOutRe[*addr ] = (*R14).y;
		*addr = c + 15; bufOutRe[*addr ] = (*R15).y;
	}

	__syncthreads();

	//if(rw)
	{
		*addr = outOffset + ( 0 + me*1 + 0 + 0 )    ;(*R0).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 256 )  ;(*R1).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 512 )  ;(*R2).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 768 )  ;(*R3).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1024 ) ;(*R4).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1280 ) ;(*R5).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1536 ) ;(*R6).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 1792 ) ;(*R7).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2048 ) ;(*R8).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2304 ) ;(*R9).y = bufOutIm [*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2560 ) ;(*R10).y = bufOutIm[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 2816 ) ;(*R11).y = bufOutIm[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3072 ) ;(*R12).y = bufOutIm[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3328 ) ;(*R13).y = bufOutIm[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3584 ) ;(*R14).y = bufOutIm[*addr];
		*addr = outOffset + ( 0 + me*1 + 0 + 3840 ) ;(*R15).y = bufOutIm[*addr];
	}

	__syncthreads();

}

// ----------------------------------------------------------------------------
template <typename T >
__device__ inline void
FwdPass1_len4096_0(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out, 
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, 
					real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7,
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	__syncthreads();
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
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).x;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).x;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).x;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).x;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).x;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).x;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).x;
		*addr = c + 112; bufOutRe[*addr] = (*R7).x;
		*addr = c + 128; bufOutRe[*addr] = (*R8).x;
		*addr = c + 144; bufOutRe[*addr] = (*R9).x;
		*addr = c + 160; bufOutRe[*addr] = (*R10).x;
		*addr = c + 176; bufOutRe[*addr] = (*R11).x;
		*addr = c + 192; bufOutRe[*addr] = (*R12).x;
		*addr = c + 208; bufOutRe[*addr] = (*R13).x;
		*addr = c + 224; bufOutRe[*addr] = (*R14).x;
		*addr = c + 240; bufOutRe[*addr] = (*R15).x;
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = b + 0;   (*R0).x  = bufOutRe[*addr];
		*addr = b + 256; (*R1).x  = bufOutRe[*addr];
		*addr = b + 512; (*R2).x  = bufOutRe[*addr];
		*addr = b + 768; (*R3).x  = bufOutRe[*addr];
		*addr = b + 1024;(*R4).x  = bufOutRe[*addr];
		*addr = b + 1280;(*R5).x  = bufOutRe[*addr];
		*addr = b + 1536;(*R6).x  = bufOutRe[*addr];
		*addr = b + 1792;(*R7).x  = bufOutRe[*addr];
		*addr = b + 2048;(*R8).x  = bufOutRe[*addr];
		*addr = b + 2304;(*R9).x  = bufOutRe[*addr];
		*addr = b + 2560;(*R10).x = bufOutRe[*addr];
		*addr = b + 2816;(*R11).x = bufOutRe[*addr];
		*addr = b + 3072;(*R12).x = bufOutRe[*addr];
		*addr = b + 3328;(*R13).x = bufOutRe[*addr];
		*addr = b + 3584;(*R14).x = bufOutRe[*addr];
		*addr = b + 3840;(*R15).x = bufOutRe[*addr];
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).y;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).y;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).y;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).y;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).y;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).y;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).y;
		*addr = c + 112; bufOutRe[*addr] = (*R7).y;
		*addr = c + 128; bufOutRe[*addr] = (*R8).y;
		*addr = c + 144; bufOutRe[*addr] = (*R9).y;
		*addr = c + 160; bufOutRe[*addr] = (*R10).y;
		*addr = c + 176; bufOutRe[*addr] = (*R11).y;
		*addr = c + 192; bufOutRe[*addr] = (*R12).y;
		*addr = c + 208; bufOutRe[*addr] = (*R13).y;
		*addr = c + 224; bufOutRe[*addr] = (*R14).y;
		*addr = c + 240; bufOutRe[*addr] = (*R15).y;
	}
	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = a + 0;   (*R0).y  = bufOutIm[*addr];
		*addr = a + 256; (*R1).y  = bufOutIm[*addr];
		*addr = a + 512; (*R2).y  = bufOutIm[*addr];
		*addr = a + 768; (*R3).y  = bufOutIm[*addr];
		*addr = a + 1024;(*R4).y  = bufOutIm[*addr];
		*addr = a + 1280;(*R5).y  = bufOutIm[*addr];
		*addr = a + 1536;(*R6).y  = bufOutIm[*addr];
		*addr = a + 1792;(*R7).y  = bufOutIm[*addr];
		*addr = a + 2048;(*R8).y  = bufOutIm[*addr];
		*addr = a + 2304;(*R9).y  = bufOutIm[*addr];
		*addr = a + 2560;(*R10).y = bufOutIm[*addr];
		*addr = a + 2816;(*R11).y = bufOutIm[*addr];
		*addr = a + 3072;(*R12).y = bufOutIm[*addr];
		*addr = a + 3328;(*R13).y = bufOutIm[*addr];
		*addr = a + 3584;(*R14).y = bufOutIm[*addr];
		*addr = a + 3840;(*R15).y = bufOutIm[*addr];
	}
	__syncthreads();
}
template <typename T >
__device__ inline void
FwdPass1_len4096_1(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out, 
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, 
					real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7,
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	double phase;
	float phase_fp;
	float wx, wy, rx, ry;
	float TR, TI;
	__syncthreads();
	{
		phase = -1.0 * (me%16) * 0.00390625; // 0.00390625 = 1 / 256
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		phase = -2.0*0.00390625 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		phase = -3.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		phase = -4.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		phase = -5.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		phase = -6.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		phase = -7.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		phase = -8.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		phase = -9.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		phase = -10.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		phase = -11.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		phase = -12.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		phase = -13.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		phase = -14.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		phase = -15.0/256 * (me%16);
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).x;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).x;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).x;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).x;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).x;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).x;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).x;
		*addr = c + 112; bufOutRe[*addr] = (*R7).x;
		*addr = c + 128; bufOutRe[*addr] = (*R8).x;
		*addr = c + 144; bufOutRe[*addr] = (*R9).x;
		*addr = c + 160; bufOutRe[*addr] = (*R10).x;
		*addr = c + 176; bufOutRe[*addr] = (*R11).x;
		*addr = c + 192; bufOutRe[*addr] = (*R12).x;
		*addr = c + 208; bufOutRe[*addr] = (*R13).x;
		*addr = c + 224; bufOutRe[*addr] = (*R14).x;
		*addr = c + 240; bufOutRe[*addr] = (*R15).x;
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = b + 0;   (*R0).x  = bufOutRe[*addr];
		*addr = b + 256; (*R1).x  = bufOutRe[*addr];
		*addr = b + 512; (*R2).x  = bufOutRe[*addr];
		*addr = b + 768; (*R3).x  = bufOutRe[*addr];
		*addr = b + 1024;(*R4).x  = bufOutRe[*addr];
		*addr = b + 1280;(*R5).x  = bufOutRe[*addr];
		*addr = b + 1536;(*R6).x  = bufOutRe[*addr];
		*addr = b + 1792;(*R7).x  = bufOutRe[*addr];
		*addr = b + 2048;(*R8).x  = bufOutRe[*addr];
		*addr = b + 2304;(*R9).x  = bufOutRe[*addr];
		*addr = b + 2560;(*R10).x = bufOutRe[*addr];
		*addr = b + 2816;(*R11).x = bufOutRe[*addr];
		*addr = b + 3072;(*R12).x = bufOutRe[*addr];
		*addr = b + 3328;(*R13).x = bufOutRe[*addr];
		*addr = b + 3584;(*R14).x = bufOutRe[*addr];
		*addr = b + 3840;(*R15).x = bufOutRe[*addr];
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).y;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).y;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).y;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).y;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).y;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).y;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).y;
		*addr = c + 112; bufOutRe[*addr] = (*R7).y;
		*addr = c + 128; bufOutRe[*addr] = (*R8).y;
		*addr = c + 144; bufOutRe[*addr] = (*R9).y;
		*addr = c + 160; bufOutRe[*addr] = (*R10).y;
		*addr = c + 176; bufOutRe[*addr] = (*R11).y;
		*addr = c + 192; bufOutRe[*addr] = (*R12).y;
		*addr = c + 208; bufOutRe[*addr] = (*R13).y;
		*addr = c + 224; bufOutRe[*addr] = (*R14).y;
		*addr = c + 240; bufOutRe[*addr] = (*R15).y;
	}
	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = a + 0;   (*R0).y  = bufOutIm[*addr];
		*addr = a + 256; (*R1).y  = bufOutIm[*addr];
		*addr = a + 512; (*R2).y  = bufOutIm[*addr];
		*addr = a + 768; (*R3).y  = bufOutIm[*addr];
		*addr = a + 1024;(*R4).y  = bufOutIm[*addr];
		*addr = a + 1280;(*R5).y  = bufOutIm[*addr];
		*addr = a + 1536;(*R6).y  = bufOutIm[*addr];
		*addr = a + 1792;(*R7).y  = bufOutIm[*addr];
		*addr = a + 2048;(*R8).y  = bufOutIm[*addr];
		*addr = a + 2304;(*R9).y  = bufOutIm[*addr];
		*addr = a + 2560;(*R10).y = bufOutIm[*addr];
		*addr = a + 2816;(*R11).y = bufOutIm[*addr];
		*addr = a + 3072;(*R12).y = bufOutIm[*addr];
		*addr = a + 3328;(*R13).y = bufOutIm[*addr];
		*addr = a + 3584;(*R14).y = bufOutIm[*addr];
		*addr = a + 3840;(*R15).y = bufOutIm[*addr];
	}
	__syncthreads();
}
template <typename T >
__device__ inline void
FwdPass1_len4096_2(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out, 
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, 
					real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7,
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	double2 W0 = dtwiddles[15 + 15*((1*me + 0)%16) + 0];
	double wx0 = W0.x;
	double wy0 = W0.y;
	double cwx = 1.0f;
	double cwy = 0;
	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;
		
		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}
	__syncthreads();
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	__syncthreads();
	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).x;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).x;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).x;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).x;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).x;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).x;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).x;
		*addr = c + 112; bufOutRe[*addr] = (*R7).x;
		*addr = c + 128; bufOutRe[*addr] = (*R8).x;
		*addr = c + 144; bufOutRe[*addr] = (*R9).x;
		*addr = c + 160; bufOutRe[*addr] = (*R10).x;
		*addr = c + 176; bufOutRe[*addr] = (*R11).x;
		*addr = c + 192; bufOutRe[*addr] = (*R12).x;
		*addr = c + 208; bufOutRe[*addr] = (*R13).x;
		*addr = c + 224; bufOutRe[*addr] = (*R14).x;
		*addr = c + 240; bufOutRe[*addr] = (*R15).x;
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = b + 0;   (*R0).x  = bufOutRe[*addr];
		*addr = b + 256; (*R1).x  = bufOutRe[*addr];
		*addr = b + 512; (*R2).x  = bufOutRe[*addr];
		*addr = b + 768; (*R3).x  = bufOutRe[*addr];
		*addr = b + 1024;(*R4).x  = bufOutRe[*addr];
		*addr = b + 1280;(*R5).x  = bufOutRe[*addr];
		*addr = b + 1536;(*R6).x  = bufOutRe[*addr];
		*addr = b + 1792;(*R7).x  = bufOutRe[*addr];
		*addr = b + 2048;(*R8).x  = bufOutRe[*addr];
		*addr = b + 2304;(*R9).x  = bufOutRe[*addr];
		*addr = b + 2560;(*R10).x = bufOutRe[*addr];
		*addr = b + 2816;(*R11).x = bufOutRe[*addr];
		*addr = b + 3072;(*R12).x = bufOutRe[*addr];
		*addr = b + 3328;(*R13).x = bufOutRe[*addr];
		*addr = b + 3584;(*R14).x = bufOutRe[*addr];
		*addr = b + 3840;(*R15).x = bufOutRe[*addr];
	}

	__syncthreads();

	//if(rw)
	{
		uint32_t a,b,c, t;
		a = me*1;t = me*1+0;
		a = t / 16; a = a *256;
		b = t % 16;
		c = outOffset + a;
		c = c + b;
		*addr = c + 0;   bufOutRe[*addr] = (*R0).y;
		*addr = c + 16;  bufOutRe[*addr] = (*R1).y;
		*addr = c + 32;  bufOutRe[*addr] = (*R2).y;
		*addr = c + 48;  bufOutRe[*addr] = (*R3).y;
		*addr = c + 64;  bufOutRe[*addr] = (*R4).y;
		*addr = c + 80;  bufOutRe[*addr] = (*R5).y;
		*addr = c + 96;  bufOutRe[*addr] = (*R6).y;
		*addr = c + 112; bufOutRe[*addr] = (*R7).y;
		*addr = c + 128; bufOutRe[*addr] = (*R8).y;
		*addr = c + 144; bufOutRe[*addr] = (*R9).y;
		*addr = c + 160; bufOutRe[*addr] = (*R10).y;
		*addr = c + 176; bufOutRe[*addr] = (*R11).y;
		*addr = c + 192; bufOutRe[*addr] = (*R12).y;
		*addr = c + 208; bufOutRe[*addr] = (*R13).y;
		*addr = c + 224; bufOutRe[*addr] = (*R14).y;
		*addr = c + 240; bufOutRe[*addr] = (*R15).y;
	}
	__syncthreads();

	//if(rw)
	{
		uint32_t a,b;
		a = me*1;
		a = 0 + a;
		a = a + 0;
		b = a + outOffset;
		*addr = a + 0;   (*R0).y  = bufOutIm[*addr];
		*addr = a + 256; (*R1).y  = bufOutIm[*addr];
		*addr = a + 512; (*R2).y  = bufOutIm[*addr];
		*addr = a + 768; (*R3).y  = bufOutIm[*addr];
		*addr = a + 1024;(*R4).y  = bufOutIm[*addr];
		*addr = a + 1280;(*R5).y  = bufOutIm[*addr];
		*addr = a + 1536;(*R6).y  = bufOutIm[*addr];
		*addr = a + 1792;(*R7).y  = bufOutIm[*addr];
		*addr = a + 2048;(*R8).y  = bufOutIm[*addr];
		*addr = a + 2304;(*R9).y  = bufOutIm[*addr];
		*addr = a + 2560;(*R10).y = bufOutIm[*addr];
		*addr = a + 2816;(*R11).y = bufOutIm[*addr];
		*addr = a + 3072;(*R12).y = bufOutIm[*addr];
		*addr = a + 3328;(*R13).y = bufOutIm[*addr];
		*addr = a + 3584;(*R14).y = bufOutIm[*addr];
		*addr = a + 3840;(*R15).y = bufOutIm[*addr];
	}
	__syncthreads();
}

// ----------------------------------------------------------------------------
template <typename T >
__device__ inline void
FwdPass2_len4096_0(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out,
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, 
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
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
	__syncthreads();
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	__syncthreads();
	//if(rw)
	{
		*addr = outOffset + ( 1*me + 0 + 0    )*stride_out;bufOut[*addr] = (*R0);
		*addr = outOffset + ( 1*me + 0 + 256  )*stride_out;bufOut[*addr] = (*R1);
		*addr = outOffset + ( 1*me + 0 + 512  )*stride_out;bufOut[*addr] = (*R2);
		*addr = outOffset + ( 1*me + 0 + 768  )*stride_out;bufOut[*addr] = (*R3);
		*addr = outOffset + ( 1*me + 0 + 1024 )*stride_out;bufOut[*addr] = (*R4);
		*addr = outOffset + ( 1*me + 0 + 1280 )*stride_out;bufOut[*addr] = (*R5);
		*addr = outOffset + ( 1*me + 0 + 1536 )*stride_out;bufOut[*addr] = (*R6);
		*addr = outOffset + ( 1*me + 0 + 1792 )*stride_out;bufOut[*addr] = (*R7);
		*addr = outOffset + ( 1*me + 0 + 2048 )*stride_out;bufOut[*addr] = (*R8);
		*addr = outOffset + ( 1*me + 0 + 2304 )*stride_out;bufOut[*addr] = (*R9);
		*addr = outOffset + ( 1*me + 0 + 2560 )*stride_out;bufOut[*addr] = (*R10);
		*addr = outOffset + ( 1*me + 0 + 2816 )*stride_out;bufOut[*addr] = (*R11);
		*addr = outOffset + ( 1*me + 0 + 3072 )*stride_out;bufOut[*addr] = (*R12);
		*addr = outOffset + ( 1*me + 0 + 3328 )*stride_out;bufOut[*addr] = (*R13);
		*addr = outOffset + ( 1*me + 0 + 3584 )*stride_out;bufOut[*addr] = (*R14);
		*addr = outOffset + ( 1*me + 0 + 3840 )*stride_out;bufOut[*addr] = (*R15);	
	}
}
template <typename T >
__device__ inline void
FwdPass2_len4096_1(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out,
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, 
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	double phase;
	float phase_fp;
	float wx, wy, rx, ry;
	float TR, TI;
	{
		phase = -1.0 * 0.000244140625 * me; // 0.000244140625 = 1/4096
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		phase = -2.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		phase = -3.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		phase = -4.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		phase = -5.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		phase = -6.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		phase = -7.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		phase = -8.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		phase = -9.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		phase = -10.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		phase = -11.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		phase = -12.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		phase = -13.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		phase = -14.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		phase = -15.0 * 0.000244140625 * me;
		phase_fp = (float)(phase);
		asm volatile("v_cos_f32 %0, %1\n":"=v"(wx):"v"(phase_fp));
		asm volatile("v_sin_f32 %0, %1\n":"=v"(wy):"v"(phase_fp));
		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	//if(rw)
	{
		*addr = outOffset + ( 1*me + 0 + 0    )*stride_out;bufOut[*addr] = (*R0);
		*addr = outOffset + ( 1*me + 0 + 256  )*stride_out;bufOut[*addr] = (*R1);
		*addr = outOffset + ( 1*me + 0 + 512  )*stride_out;bufOut[*addr] = (*R2);
		*addr = outOffset + ( 1*me + 0 + 768  )*stride_out;bufOut[*addr] = (*R3);
		*addr = outOffset + ( 1*me + 0 + 1024 )*stride_out;bufOut[*addr] = (*R4);
		*addr = outOffset + ( 1*me + 0 + 1280 )*stride_out;bufOut[*addr] = (*R5);
		*addr = outOffset + ( 1*me + 0 + 1536 )*stride_out;bufOut[*addr] = (*R6);
		*addr = outOffset + ( 1*me + 0 + 1792 )*stride_out;bufOut[*addr] = (*R7);
		*addr = outOffset + ( 1*me + 0 + 2048 )*stride_out;bufOut[*addr] = (*R8);
		*addr = outOffset + ( 1*me + 0 + 2304 )*stride_out;bufOut[*addr] = (*R9);
		*addr = outOffset + ( 1*me + 0 + 2560 )*stride_out;bufOut[*addr] = (*R10);
		*addr = outOffset + ( 1*me + 0 + 2816 )*stride_out;bufOut[*addr] = (*R11);
		*addr = outOffset + ( 1*me + 0 + 3072 )*stride_out;bufOut[*addr] = (*R12);
		*addr = outOffset + ( 1*me + 0 + 3328 )*stride_out;bufOut[*addr] = (*R13);
		*addr = outOffset + ( 1*me + 0 + 3584 )*stride_out;bufOut[*addr] = (*R14);
		*addr = outOffset + ( 1*me + 0 + 3840 )*stride_out;bufOut[*addr] = (*R15);	
	}
}
template <typename T >
__device__ inline void
FwdPass2_len4096_2(const T *twiddles, const double2 *dtwiddles,
					const size_t stride_in, const size_t stride_out,
					unsigned int rw, unsigned int b, unsigned int me, 
					unsigned int inOffset, unsigned int outOffset, 
					real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, 
					T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7, 
					T *R8, T *R9, T *R10, T *R11, T *R12, T *R13, T *R14, T *R15,
					uint32_t *addr,
					double2 * ddebug)
{
	double2 W0 = dtwiddles[255 + 15*((1*me + 0)%256) + 0];
	double wx0 = W0.x;
	double wy0 = W0.y;
	double cwx = 1.0f;
	double cwy = 0;
	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R4).x; ry = (*R4).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R5).x; ry = (*R5).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R6).x; ry = (*R6).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R7).x; ry = (*R7).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R7).x = TR;
		(*R7).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R8).x; ry = (*R8).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R8).x = TR;
		(*R8).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R9).x; ry = (*R9).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R9).x = TR;
		(*R9).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R10).x; ry = (*R10).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R10).x = TR;
		(*R10).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R11).x; ry = (*R11).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R11).x = TR;
		(*R11).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R12).x; ry = (*R12).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R12).x = TR;
		(*R12).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R13).x; ry = (*R13).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R13).x = TR;
		(*R13).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R14).x; ry = (*R14).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R14).x = TR;
		(*R14).y = TI;
	}

	{
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;

		double twx = wx0;
		double twy = wy0;
		twx = cwx*wx0 - cwy*wy0;	
		twy = cwy*wx0 + cwx*wy0;
		cwx = twx;
		cwy = twy;
		wx = (float)cwx; 
		wy = (float)cwy;

		rx = (*R15).x; ry = (*R15).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R15).x = TR;
		(*R15).y = TI;
	}
	__syncthreads();
	__syncthreads();

	FwdRad16B1(R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

	__syncthreads();
	__syncthreads();
	//if(rw)
	{
		*addr = outOffset + ( 1*me + 0 + 0    )*stride_out;bufOut[*addr] = (*R0);
		*addr = outOffset + ( 1*me + 0 + 256  )*stride_out;bufOut[*addr] = (*R1);
		*addr = outOffset + ( 1*me + 0 + 512  )*stride_out;bufOut[*addr] = (*R2);
		*addr = outOffset + ( 1*me + 0 + 768  )*stride_out;bufOut[*addr] = (*R3);
		*addr = outOffset + ( 1*me + 0 + 1024 )*stride_out;bufOut[*addr] = (*R4);
		*addr = outOffset + ( 1*me + 0 + 1280 )*stride_out;bufOut[*addr] = (*R5);
		*addr = outOffset + ( 1*me + 0 + 1536 )*stride_out;bufOut[*addr] = (*R6);
		*addr = outOffset + ( 1*me + 0 + 1792 )*stride_out;bufOut[*addr] = (*R7);
		*addr = outOffset + ( 1*me + 0 + 2048 )*stride_out;bufOut[*addr] = (*R8);
		*addr = outOffset + ( 1*me + 0 + 2304 )*stride_out;bufOut[*addr] = (*R9);
		*addr = outOffset + ( 1*me + 0 + 2560 )*stride_out;bufOut[*addr] = (*R10);
		*addr = outOffset + ( 1*me + 0 + 2816 )*stride_out;bufOut[*addr] = (*R11);
		*addr = outOffset + ( 1*me + 0 + 3072 )*stride_out;bufOut[*addr] = (*R12);
		*addr = outOffset + ( 1*me + 0 + 3328 )*stride_out;bufOut[*addr] = (*R13);
		*addr = outOffset + ( 1*me + 0 + 3584 )*stride_out;bufOut[*addr] = (*R14);
		*addr = outOffset + ( 1*me + 0 + 3840 )*stride_out;bufOut[*addr] = (*R15);	
	}
}

////////////////////////////////////////Encapsulated passes kernels
template <typename T >
__device__ inline void 
fwd_len4096_device(const T *twiddles, const double2 *dtwiddles, 
	const size_t stride_in, const size_t stride_out, 
	unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, 
	T *lwbIn, T *lwbOut, real_type_t<T> *lds,
	double2 * ddebug)
{
	T R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15;
	uint32_t addr;
	FwdPass0_len4096<T >(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &addr, ddebug);

	__syncthreads();
	__syncthreads();
	__syncthreads();
	FwdPass1_len4096_0<T>(twiddles, dtwiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &addr, ddebug);
	
	__syncthreads();
	__syncthreads();
	__syncthreads();
	FwdPass2_len4096_0<T >(twiddles, dtwiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13, &R14, &R15, &addr, ddebug);
}

////////////////////////////////////////Global kernels

//Kernel configuration: number of threads per thread block: 256, maximum transforms: 1, Passes: 3
__global__ void 
my_fft_fwd_op_len4096( 
	const float2 * __restrict__ twiddles, const double2 * __restrict__ dtwiddles, 
	const size_t dim, const size_t *lengths, 
	const size_t *stride_in, const size_t *stride_out, 
	const size_t batch_count, 
	float2 * __restrict__ gbIn, float2 * __restrict__ gbOut,
	double2 * __restrict__ ddebug)
{
	__shared__ real_type_t<float2> lds[4096];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int rw = 1;
	unsigned int b = 0;

	size_t counter_mod = batch;
	if(dim == 1)
	{
		iOffset += counter_mod*stride_in[1];
		oOffset += counter_mod*stride_out[1];
	}
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len4096_device<float2 >(twiddles, dtwiddles, stride_in[0], stride_out[0],  1, b, me, 0, lwbIn, lwbOut, lds, ddebug);
}
