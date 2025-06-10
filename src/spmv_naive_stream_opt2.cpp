#include "spmv_naive_stream_opt2.hpp"



void spmv_kernel_opt2(const int  row_ptr[MAX_M],
        const int  col_idx[MAX_SZ],
        const DATA_TYPE val[MAX_SZ],
        const DATA_TYPE x[MAX_N],
        DATA_TYPE y[MAX_M],
        int M, int N, int nnz) {
#pragma HLS INTERFACE m_axi port=row_ptr  depth=256 offset=slave
#pragma HLS INTERFACE m_axi port=col_idx depth=256 offset=slave
#pragma HLS INTERFACE m_axi port=val      depth=256 offset=slave
#pragma HLS INTERFACE m_axi port=x        depth=256  offset=slave
#pragma HLS INTERFACE m_axi port=y         depth=256 offset=slave
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control
#pragma HLS DATAFLOW
	int col_left;
	DATA_TYPE sum;

	col_left = 0;
	sum = 0;
	hls::stream<int> row_fifo("row stream"), col_fifo;
	hls::stream<DATA_TYPE> val_fifo, y_fifo;

	LOOP_ROW_SIZE: for (int i = 0, prv = 0; i < M; i++) {
	#pragma HLS PIPELINE
			int cur = row_ptr[i + 1];
			int r = cur - prv;
			prv = cur;
			row_fifo.write(r);
		}

		LOOP_NNZ_FIFO: for (int i = 0; i < nnz; i++) {
	#pragma HLS PIPELINE
			col_fifo.write(col_idx[i]);
			val_fifo.write(val[i]);
		}

#pragma HLS DATAFLOW
	LOOP_NAIVE_STREAM: for (int i = 0; i < nnz; i++) {
#pragma  HLS PIPELINE
			if (col_left == 0) {
				col_left = row_fifo.read();
				sum = 0;
			}

			//if (col_left) {
				int col = col_fifo.read();
				DATA_TYPE val = val_fifo.read();

				sum += val * x[col];
				col_left--;
			//}
			if (col_left == 0) {
				y_fifo.write(sum);
			}
	}
	for (int i = 0; i < M; i++) {
		y[i] = y_fifo.read();
	}

}


void spmv_opt2(const int  row_ptr[MAX_M],
          const int  col_idx[MAX_SZ],
          const DATA_TYPE val[MAX_SZ],
          const DATA_TYPE x[MAX_N],
          DATA_TYPE y[MAX_M],
          int M, int N, int nnz)
{


#pragma HLS INTERFACE s_axilite port=row_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx bundle=control
#pragma HLS INTERFACE s_axilite port=val     bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=N       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
//#pragma HLS INTERFACE s_axilite port=return  bundle=control


	//static int row_size[MAX_M];

	DATA_TYPE x_local[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1

	for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE
		x_local[i] = x[i];
	}


	spmv_kernel_opt2(row_ptr, col_idx, val, x_local, y, M, N, nnz);

}


