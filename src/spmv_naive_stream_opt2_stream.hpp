#ifndef __INC_H__
#define __INC_H__
#include "ap_fixed.h"
#include <stdint.h>
#include "hls_math.h"
#include "hls_stream.h"
#include <ap_int.h>

#define DATA_TYPE float

const static int MAX_N = 256;
const static int MAX_M = 256;
const static int MAX_SZ = 20000;
#define _STREAM_
#endif

#ifndef __SPMV_OPT2S_H__
#define __SPMV_OPT2S_H__



void spmv_naive_stream_opt2_stream(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_idx_strm,
    hls::stream<DATA_TYPE> &val_strm,
    DATA_TYPE	x[MAX_N],
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
	int N,
    int                      NNZ);

#endif
