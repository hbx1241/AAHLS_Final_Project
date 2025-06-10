#ifndef __INC_H__
#define __INC_H__
#include "ap_fixed.h"
#include <stdint.h>
#include "hls_math.h"
#include "hls_stream.h"
#include <ap_int.h>

#define DATA_TYPE float
#define II 4

const static int MAX_N = 256;
const static int MAX_M = 256;
const static int MAX_SZ = 20000;

#endif


#ifndef __SPMV_OPT3_H__
#define __SPMV_OPT3_H__


void spmv_opt3(const int  *row_ptr,
        const int  *col_idx,
        const DATA_TYPE *val,
        const DATA_TYPE *x,
        DATA_TYPE *y,
        int M, int N, int nnz);

#endif
