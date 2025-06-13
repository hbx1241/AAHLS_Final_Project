#include "spmv_balance_opt4.hpp"


void spmv_kernel_opt4(
		const int  row_ptr[MAX_M],
        const int  col_idx[MAX_SZ],
        const DATA_TYPE val[MAX_SZ],
        const DATA_TYPE x[MAX_N],
        DATA_TYPE y[MAX_M],
        int M, int N, int nnz) {
#pragma HLS INTERFACE m_axi port=row_ptr  offset=slave depth=MAX_M+1
#pragma HLS INTERFACE m_axi port=col_idx  offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=val      offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=y        offset=slave depth=MAX_M
#pragma HLS INTERFACE m_axi port=x        offset=slave depth=MAX_N
#pragma HLS INTERFACE s_axilite port=row_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx bundle=control
#pragma HLS INTERFACE s_axilite port=val     bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=N       bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control
#pragma HLS DATAFLOW
#pragma HLS INLINE OFF
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
void spmv_opt4(
		int *row_ptr,
		int *col_idx,
        DATA_TYPE  *val,
        DATA_TYPE  *y,
        int  M,
        int  N,
        int  nnz,
		int *row_ptr1,
		int        *col_idx1,
		DATA_TYPE  *val1,
		DATA_TYPE  *y1,
		int               M1,
		int               N1,
		int               nnz1,
        DATA_TYPE  *x)
{
#pragma HLS INTERFACE m_axi port=row_ptr  offset=slave depth=MAX_M+1
#pragma HLS INTERFACE m_axi port=col_idx  offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=val      offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=y        offset=slave depth=MAX_M
#pragma HLS INTERFACE m_axi port=row_ptr1  offset=slave depth=MAX_M+1
#pragma HLS INTERFACE m_axi port=col_idx1  offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=val1      offset=slave depth=MAX_SZ
#pragma HLS INTERFACE m_axi port=y1        offset=slave depth=MAX_M
#pragma HLS INTERFACE m_axi port=x        offset=slave depth=MAX_N
#pragma HLS INTERFACE s_axilite port=row_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx bundle=control
#pragma HLS INTERFACE s_axilite port=val     bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz       bundle=control
#pragma HLS INTERFACE s_axilite port=row_ptr1  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx1 bundle=control
#pragma HLS INTERFACE s_axilite port=val1     bundle=control
#pragma HLS INTERFACE s_axilite port=y1       bundle=control
#pragma HLS INTERFACE s_axilite port=M1       bundle=control
#pragma HLS INTERFACE s_axilite port=nnz1       bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control
#pragma HLS DATAFLOW


    DATA_TYPE x_local[MAX_N], x_local1[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=x_local1 cyclic factor=8 dim=1


    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE
    	x_local[i] = x[i];
    	x_local1[i] = x[i];
    }
    spmv_kernel_opt4(row_ptr, col_idx, val, x_local, y, M, N, nnz);
    spmv_kernel_opt4(row_ptr1, col_idx1, val1, x_local1, y1, M1, N1, nnz1);

}



// Compute kernel: purely streaming, no interface pragmas here
void spmv_compute(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_fifo,
    hls::stream<DATA_TYPE> &val_fifo,
    const DATA_TYPE         x[MAX_N],
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
    int                      NNZ)
{
#pragma HLS DATAFLOW
	// Local FIFOs
    hls::stream<int>       row_fifo("spmv_row");
    hls::stream<int>       row_pad_fifo("spmv_pad");
#pragma HLS STREAM variable=row_fifo depth=(MAX_M + 10)
#pragma HLS STREAM variable=row_pad_fifo depth=(MAX_M + 10)

    // Stage 1: compute padded row lengths
    int nnz_padded = 0;

        for (int r = 0, prev = 0; r < M; ++r) {
        #pragma HLS PIPELINE II=1
            int cur = row_ptr_strm.read();
            int len = cur - prev;
            prev = cur;
            int rem = len % II;
            int pad = (len == 0) ? II : (rem ? len + (II - rem) : len);
            //printf("len %d pad%d\n", len, pad);
            nnz_padded += pad;
            row_fifo.write(len);
            row_pad_fifo.write(pad);
        }


    DATA_TYPE buf[II];
    int rowLen, padLen;
    //printf("col %d val %d %d\n", col_fifo.size(), val_fifo.size(), nnz_padded);
    rowLen = 0;
    padLen = 0;
    DATA_TYPE acc;
    int cnt;

    cnt = 0;
    // Stage 3: block-wise multiply-accumulate
    for (int blk = 0; blk < nnz_padded; blk += II) {
    #pragma HLS PIPELINE
    	if (padLen == 0) {
    		rowLen = row_fifo.read();
    		padLen = row_pad_fifo.read();
    		acc = 0;
    		cnt = 0;
    	}

    	DATA_TYPE acc_tmp = 0;
        for (int j = 0; j < II; ++j) {
        #pragma HLS UNROLL
            if (cnt++ < rowLen) {
                int c   = col_fifo.read();
                auto v  = val_fifo.read();
                buf[j]  = v * x[c];
            } else {
                buf[j] = 0;
            }
        }
        // reduction
        for (int j = 0; j < II; ++j) {
        #pragma HLS UNROLL
            acc_tmp += buf[j];
        }
        acc += acc_tmp;
        padLen -= II;
        if (padLen == 0) {
        	//printf("%f\n", acc);
        	y_strm.write(acc);
        	acc = 0;
        }

    }

}


// Top-level: AXI-Stream interface; CONTROL via AXI-Lite
// M (#rows) and NNZ (#non-zeros) are passed at runtime
void spmv_stream(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_idx_strm,
    hls::stream<DATA_TYPE> &val_strm,
	DATA_TYPE 	x[MAX_N],
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
	int N,
    int                      NNZ)
{
    // AXI4-Stream ports
#pragma HLS INTERFACE axis port=row_ptr_strm
#pragma HLS INTERFACE axis port=col_idx_strm
#pragma HLS INTERFACE axis port=val_strm
#pragma HLS INTERFACE axis port=y_strm
    // Control ports
#pragma HLS INTERFACE s_axilite port=x    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=M    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=N    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=NNZ  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Overlap load, compute, store
#pragma HLS DATAFLOW

    // 1) Read input vector x into BRAM (size assumed MAX_N)
    static DATA_TYPE x_local[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
    for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
        x_local[i] = x[i];
    }

    // 2) Compute SPMS via streaming kernel, passing M and NNZ
    spmv_compute(
        row_ptr_strm,
        col_idx_strm,
        val_strm,
        x_local,
        y_strm,
        M,
        NNZ);
}

void spmv_stream_lb(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<int>       &col_idx_strm,
    hls::stream<DATA_TYPE> &val_strm,
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
    int                      NNZ,
	hls::stream<int>       &row_ptr_strm1,
	hls::stream<int>       &col_idx_strm1,
	hls::stream<DATA_TYPE> &val_strm1,
	hls::stream<DATA_TYPE> &y_strm1,
	int                      M1,
	int                      NNZ1,
	int N,
	DATA_TYPE x[MAX_N])
{
    // AXI4-Stream ports
#pragma HLS INTERFACE axis port=row_ptr_strm
#pragma HLS INTERFACE axis port=col_idx_strm
#pragma HLS INTERFACE axis port=val_strm
#pragma HLS INTERFACE axis port=y_strm
#pragma HLS INTERFACE axis port=row_ptr_strm1
#pragma HLS INTERFACE axis port=col_idx_strm1
#pragma HLS INTERFACE axis port=val_strm1
#pragma HLS INTERFACE axis port=y_strm1
    // Control ports
#pragma HLS INTERFACE s_axilite port=x    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=N 	  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=M    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=M1    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=NNZ  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=NNZ1   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Overlap load, compute, store
#pragma HLS DATAFLOW

    // 1) Read input vector x into BRAM (size assumed MAX_N)
    static DATA_TYPE x_local[MAX_N], x_local1[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=x_local1 cyclic factor=8 dim=1

    for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
        x_local[i] = x[i];
        x_local1[i] = x[i];
    }

    // 2) Compute SPMS via streaming kernel, passing M and NNZ
    spmv_compute(
        row_ptr_strm,
        col_idx_strm,
        val_strm,
        x_local,
        y_strm,
        M,
        NNZ);
    spmv_compute(
           row_ptr_strm1,
           col_idx_strm1,
           val_strm1,
           x_local1,
           y_strm1,
           M1,
           NNZ1);
}

void spmv_compute_vec(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<ivec_t>       &col_fifo,
    hls::stream<vvec_t> &val_fifo,
    const DATA_TYPE         x[MAX_N],
    hls::stream<DATA_TYPE> &y_strm,
    int                      M,
    int                      NNZ)
{
#pragma HLS DATAFLOW



    DATA_TYPE buf[II], buf1[II], acc;
    int padLen;

    padLen  = 0;
    acc = 0;
    // Stage 3: block-wise multiply-accumulate
    for (int blk = 0; blk < NNZ; blk += II) {
    #pragma HLS PIPELINE
    	if (padLen == 0) {
    		padLen = row_ptr_strm.read();
    		acc = 0;
    	}

    	DATA_TYPE acc_tmp = 0, acc_tmp1 = 0;
        for (int j = 0; j < II; j+=2) {
        #pragma HLS UNROLL
             ivec_t c   = col_fifo.read();
             vvec_t v  = val_fifo.read();
			buf[j]  = v[0] * x[c[0]];
			buf[j+1] = v[1] * x[c[1]];
        }
        // reduction
        for (int j = 0; j < II; j++) {
        #pragma HLS UNROLL
            acc_tmp += buf[j];
        }
        acc += acc_tmp;
        padLen -= II;
        if (padLen == 0) {
        	//printf("%f\n", acc);
        	y_strm.write(acc);
        	acc = 0;
        }

    }

}

void spmv_stream_vec(
    hls::stream<int>       &row_ptr_strm,
    hls::stream<ivec_t>       &col_idx_strm,
    hls::stream<vvec_t> &val_strm,
	DATA_TYPE 	x[MAX_N],
	hls::stream<DATA_TYPE> &y_strm,
    int                      M,
	int N,
    int                      NNZ)
{
    // AXI4-Stream ports
#pragma HLS INTERFACE axis port=row_ptr_strm
#pragma HLS INTERFACE axis port=col_idx_strm
#pragma HLS INTERFACE axis port=val_strm
#pragma HLS INTERFACE axis port=y_strm
    // Control ports
#pragma HLS INTERFACE s_axilite port=x    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=M    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=N    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=NNZ  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Overlap load, compute, store
#pragma HLS DATAFLOW

    // 1) Read input vector x into BRAM (size assumed MAX_N)
    static DATA_TYPE x_local[MAX_N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
    for (int i = 0; i < N; ++i) {
    #pragma HLS PIPELINE II=1
        x_local[i] = x[i];
    }

    // 2) Compute SPMS via streaming kernel, passing M and NNZ
    spmv_compute_vec(
        row_ptr_strm,
        col_idx_strm,
        val_strm,
        x_local,
        y_strm,
        M,
        NNZ);
}
