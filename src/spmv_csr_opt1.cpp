#include "spmv_csr_opt1.hpp"



void spmv_opt1(const int  row_ptr[MAX_M],
          const int  col_idx[MAX_SZ],
          const DATA_TYPE val[MAX_SZ],
          const DATA_TYPE x[MAX_N],
          DATA_TYPE y[MAX_M],
          int M, int N, int nnz)
{
/*
#pragma HLS INTERFACE m_axi port=row_ptr  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=col_idx offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=val     offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y        offset=slave bundle=gmem0*/
#pragma HLS INTERFACE s_axilite port=row_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx bundle=control
#pragma HLS INTERFACE s_axilite port=val     bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
#pragma HLS DATAFLOW
	DATA_TYPE x_local[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1

	for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE
		x_local[i] = x[i];
	}
RowLoop:
    for (int r = 0, start = 0; r < M; ++r) {
        DATA_TYPE sum = 0;
        int end   = row_ptr[r+1];
        //printf("end = %d\n", row_ptr[r + 1]);
    ColLoop:
        for (int k = start; k < end; ++k) {
#pragma HLS PIPELINE
            sum += val[k] * x_local[col_idx[k]];
            //printf("%d %f %f\n", val[k], x_local[col_idx[k]], col_idx[k]);
        }
        start = end;
        y[r] = sum;
        //printf("r = %d %f\n",r, y[r]);
    }

}




