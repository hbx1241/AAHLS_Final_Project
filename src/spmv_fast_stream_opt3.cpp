#include "spmv_fast_stream_opt3.hpp"



void spmv_kernel_opt3(
		const int  row_ptr[MAX_M],
        const int  col_idx[MAX_SZ],
        const DATA_TYPE val[MAX_SZ],
        const DATA_TYPE x[MAX_N],
        DATA_TYPE y[MAX_M],
        int M, int N, int nnz) {
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control
#pragma HLS DATAFLOW
//
	int rz, rz_pad, cnt;
	DATA_TYPE sum;
	DATA_TYPE temp[II];


	int row_size[MAX_M];
	int row_pad_size[MAX_M];
	int nnz_new;
	hls::stream<int> row_fifo, row_pad_fifo, col_fifo;
	hls::stream<DATA_TYPE> val_fifo, y_fifo;

	nnz_new = 0;
	sum = 0;
	rz_pad = 0;
	rz= 0;
	//

		LOOP_ROW_SIZE: for (int i = 0, prv = 0; i < M; i++) {
	#pragma HLS PIPELINE
			int r, cur;

			cur = row_ptr[i + 1];
			r = cur - prv;
			prv = cur;
			int left = r % II;
			int r_new, rr = r + II - left;
			if (r == 0) {
				r_new = II;
			} else if (left){
				r_new = rr;
			} else r_new = r;

			nnz_new += r_new;
			row_fifo.write(r);
			row_pad_fifo.write(r_new);

		}
	//#pragma HLS DATAFLOW
		LOOP_NNZ_FIFO: for (int i = 0; i < nnz; i++) {
			#pragma HLS PIPELINE
			col_fifo.write(col_idx[i]);
			val_fifo.write(val[i]);
		}


	LOOP_FAST_STREAM: for (int i = 0; i < nnz_new; i+=II) {
		#pragma  HLS PIPELINE
		if (rz_pad == 0) {
			rz = row_fifo.read();
			rz_pad = row_pad_fifo.read();
			sum = 0;
			cnt = 0;
		}

		LOOP_II: for (int j = 0; j < II; j++) {
#pragma HLS UNROLL
			cnt++;
			if (cnt > rz) {
				temp[j] = 0;
			} else {
				int col = col_fifo.read();
				DATA_TYPE val = val_fifo.read();

				temp[j] = val * x[col];
			}
		}
		DATA_TYPE sum_tmp = 0;
		LOOP_II2: for (int  j= 0; j < II; j++) {
#pragma HLS UNROLL
			sum_tmp += temp[j];
		}
		sum += sum_tmp;
		rz_pad -= II;
		if (rz_pad == 0) {
			y_fifo.write(sum);
		}
	}
	LOOP_RESULT: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
		y[i] = y_fifo.read();
	}

}

void spmv_opt3(const int  row_ptr[MAX_M],
          const int  col_idx[MAX_SZ],
          const DATA_TYPE val[MAX_SZ],
          const DATA_TYPE x[MAX_N],
          DATA_TYPE y[MAX_M],
          int M, int N, int nnz)
{
	/*
#pragma HLS INTERFACE m_axi port=row_ptr  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=col_idx offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=val     offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y        offset=slave bundle=gmem4*/
#pragma HLS INTERFACE s_axilite port=row_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx bundle=control
#pragma HLS INTERFACE s_axilite port=val     bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control



	DATA_TYPE x_local[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=1
//
	for (int i = 0; i < N; i++) {
		x_local[i] = x[i];
	}
	spmv_kernel_opt3(row_ptr, col_idx, val, x_local, y, M, N, nnz);
}



