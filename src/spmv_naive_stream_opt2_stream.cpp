#include "spmv_naive_stream_opt2_stream.hpp"

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

    int rowLen;
    rowLen = 0;

    DATA_TYPE acc;
    int cnt;

    // Stage 3: block-wise multiply-accumulate
    for (int i = 0, prv = 0; i < NNZ; i += 1) {
    #pragma HLS PIPELINE
    	if (rowLen == 0) {
    		int cur = row_ptr_strm.read();

    		rowLen = cur - prv;
    		prv = cur;
    		acc = 0;
    	}
    	//if (rowLen != 0) {
			int c   = rowLen == 0 ? 0 : col_fifo.read();
			DATA_TYPE v  = rowLen == 0 ? 0 : val_fifo.read();

			acc += v * x[c];
			rowLen--;
    	//}
        if (rowLen == 0) {
        	y_strm.write(acc);
        }

    }

}


// Top-level: AXI-Stream interface; CONTROL via AXI-Lite
// M (#rows) and NNZ (#non-zeros) are passed at runtime
void spmv_naive_stream_opt2_stream(
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


