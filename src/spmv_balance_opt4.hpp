#ifndef __INC_H__
#define __INC_H__
#include "ap_fixed.h"
#include <stdint.h>
#include "hls_math.h"
#include "hls_stream.h"
#include <ap_int.h>

#define DATA_TYPE float
#define II 4
#define P 2

const static int MAX_N = 512;
const static int MAX_M = 512;
const static int MAX_SZ = 20000;
#endif

#ifndef __SPMV_OPT4_H__
#define __SPMV_OPT4_H__


void spmv_opt4(
		int *row_ptr,
        int        *col_idx,
        DATA_TYPE  *val,
        DATA_TYPE  *y,
        int               M,
        int               N,
        int               nnz,
		int *row_ptr1,
		int        *col_idx1,
		DATA_TYPE  *val1,
		DATA_TYPE  *y1,
		int               M1,
		int               N1,
		int               nnz1,
        DATA_TYPE  *x);

void spmv_stream(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_idx_strm,
    hls::stream<DATA_TYPE> &val_strm,
    hls::stream<DATA_TYPE> &x_strm,
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
	int N,
    int                      NNZ);

void spmv_stream_lb(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_idx_strm,
    hls::stream<DATA_TYPE> &val_strm,
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
	int N,
    int                      NNZ,
	hls::stream<int>       &row_ptr_strm1,
	hls::stream<int>       &col_idx_strm1,
	hls::stream<DATA_TYPE> &val_strm1,
	hls::stream<DATA_TYPE> &y_strm1,
	int                      M1,
	int N1,
	int                      NNZ1,
	DATA_TYPE x[MAX_N]);
#endif
