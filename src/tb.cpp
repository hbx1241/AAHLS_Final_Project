// tb_spmv.cpp --------------------------------------------------------------
//#include "spmv_csr_opt1.hpp"
//#include "spmv_naive_stream_opt2_stream.hpp"
//#include "spmv_naive_stream_opt2.hpp"
#include "spmv_balance_opt4.hpp"
//#include "spmv_fast_stream_opt3.hpp"

#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <map>

// ─────────────────── design-time static limits (must match spmv.hpp) ──────
//constexpr int MAX_M  =  256;      // max rows
//constexpr int MAX_N  =  256;      // max columns

// HLS-friendly static buffers
static int          row_ptr_arr [MAX_M + 1];
static int          col_idx_arr [MAX_SZ];
static DATA_TYPE    val_arr     [MAX_SZ];
static int          row_pad_arr [MAX_M + 1];
static int          col_idx_pad_arr [MAX_SZ];
static DATA_TYPE    val_pad_arr     [MAX_SZ];
static DATA_TYPE    x_arr       [MAX_N];
static DATA_TYPE    y_hw        [MAX_M];
static DATA_TYPE    y_gold      [MAX_M];

static int 			row_ptr_arr0 [MAX_M + 1];
static int 			row_ptr_arr1 [MAX_M + 1];
static int          col_idx_arr0 [MAX_SZ];
static int          col_idx_arr1 [MAX_SZ];
static DATA_TYPE    val_arr0     [MAX_SZ];
static DATA_TYPE    val_arr1     [MAX_SZ];
static DATA_TYPE    y_hw0        [MAX_M];
static DATA_TYPE    y_hw1        [MAX_M];
int nnz0, nnz1;
int part0_row, part1_row;

#ifdef _STREAM_VEC_
    hls::stream<int> row_pad_vec_strm;
    hls::stream<ivec_t> col_idx_vec_strm;
    hls::stream<vvec_t> val_vec_strm;
    hls::stream<DATA_TYPE> y_strm;
#endif

template<typename T>
struct Csr {
    int m, n, nnz;
    std::vector<int>    row_ptr;
    std::vector<int>    col_idx;
    std::vector<T>      val;
    // For partitions
    int row_start = 0;
    int row_end   = 0;
};

// skip first line & optional comment, then slurp numbers of type T
template<typename T>
static std::vector<T> read_vector_body(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string dummy;
    std::getline(f, dummy);           // header
    if (dummy.find('#') == std::string::npos)
        std::getline(f, dummy);       // second line could be a comment
    std::vector<T> v;  T tmp;
    while (f >> tmp) v.push_back(tmp);
    return v;
}

static Csr<float> read_csr(const std::string& dir = "./")
{
    Csr<float> A;
    // row_ptr.txt – holds shape & nnz on first header line
    std::ifstream fr(dir + "row_ptr.txt");
    if (!fr) throw std::runtime_error("cannot open row_ptr.dat");

    fr >> A.m >> A.n >> A.nnz;
    //std::cout << A.m << " " << A.n << " " <<  A.nnz << std::endl;
    fr.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // rest
    fr.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // comment
    A.row_ptr.resize(A.m + 1);
    for (int i = 0; i < A.m + 1; ++i) fr >> A.row_ptr[i];

    // col_idx.txt & val.txt
    A.col_idx = read_vector_body<int  >(dir + "col_idx.txt");
    A.val     = read_vector_body<float>(dir + "val.txt");

    if ((int)A.col_idx.size() != A.nnz || (int)A.val.size() != A.nnz) {
    	std::cout << (int) A.col_idx.size() << " " << (int) A.val.size() << std::endl;
        throw std::runtime_error("nnz mismatch between headers & bodies");
    }
    return A;
}

static std::vector<float> read_dense_x(const std::string& dir = "./")
{
    std::ifstream fx(dir + "x.txt");
    if (!fx) throw std::runtime_error("cannot open x.txt");

    int n;  fx >> n;
    fx.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    fx.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // comment

    std::vector<float> x;  x.reserve(n);
    float v;  while (fx >> v) x.push_back(v);
    if ((int)x.size() != n) throw std::runtime_error("x length mismatch");
    return x;
}


// Read partitioned CSR P<id> from directory
template<typename T>
Csr<T> read_csr_p(int pid, const std::string &dir) {
    Csr<T> p;
    // Region info
    std::string fn_reg = dir + "/region_" + std::to_string(pid) + ".txt";
    std::ifstream in_reg(fn_reg);
    if (!in_reg) throw std::runtime_error("Cannot open " + fn_reg);
    std::string line;
    std::getline(in_reg, line); // skip comment
    in_reg >> p.row_start >> p.row_end;

    // Filenames for partition
    std::string fn_rptr = dir + "/row_ptr_" + std::to_string(pid) + ".txt";
    std::string fn_col  = dir + "/col_idx_" + std::to_string(pid) + ".txt";
    std::string fn_val  = dir + "/val_"     + std::to_string(pid) + ".txt";

    // Read row_ptr
    std::ifstream in_r(fn_rptr);
    if (!in_r) throw std::runtime_error("Cannot open " + fn_rptr);
    //std::getline(in_r, line);

    in_r >> p.m >> p.n >> p.nnz;
    //parse_header(line, p.m, p.n, p.nnz);

    std::getline(in_r, line); // skip comment
    std::getline(in_r, line); // skip comment
    //p.row_ptr.resize(p.m + 1);
    //for (int i = 0; i <= p.m; ++i) {in_r >> p.row_ptr[i];}
    int rp;
    while(in_r >> rp) p.row_ptr.push_back(rp);

    // Read col_idx
    std::ifstream in_c(fn_col);
    if (!in_c) throw std::runtime_error("Cannot open " + fn_col);
    std::getline(in_c, line);
    int idx;
    while (in_c >> idx) {
    	p.col_idx.push_back(idx);
    	//printf("col %d\n", idx);
    }

    // Read vals
    std::ifstream in_v(fn_val);
    if (!in_v) throw std::runtime_error("Cannot open " + fn_val);
    std::getline(in_v, line);
    T val;
    while (in_v >> val) {
    	p.val.push_back(val);
    	//printf("val %f\n", val);
    }

    return p;
}

#ifdef _STREAM_VEC_
void spmv_pad_vec(Csr<float> &A, hls::stream<int> &row_pad_fifo, hls::stream<ivec_t> &col_vec_fifo, hls::stream<vvec_t> &val_vec_fifo, int &NNZ) {
	 int ptr = 0;
	 int prv = 0;
	 int nnz = 0;
	for (int i = 0; i < A.m; i++) {
		int cur = row_ptr_arr[i + 1];
		int row_length = cur - prv;

		int row_pad = (row_length + II - 1) / II * II;
		nnz += row_pad;
		row_pad_fifo.write(row_pad);
		ivec_t iv;
		vvec_t vv;

		for (int j = 0, k = 0, t = prv; j < row_pad; j++, t++) {
			DATA_TYPE v;
			int col;

			if (t < cur) {
				v = val_arr[t];
				col = col_idx_arr[t];
			} else {
				v =  0;
				col = 0;
			}
			iv[k] = col;
			vv[k] = v;
			if (k == 1) {
				col_vec_fifo.write(iv);
				val_vec_fifo.write(vv);
				k = 0;
			} else k++;
		}
		prv = cur;
	}
	NNZ = nnz;
}
#endif

int main()
{
    std::string dir = "/home/ubuntu/HLS/SpMv/dat/";

    // ───────── load data from txt files ─────────
    Csr<float> A = read_csr(dir);
    auto       x = read_dense_x(dir);



    assert(A.m <= MAX_M && A.n <= MAX_N && A.nnz <= MAX_SZ);
    assert((int)x.size() == A.n);

    // copy into static arrays
    std::copy(A.row_ptr.begin(), A.row_ptr.end(), row_ptr_arr);
    std::copy(A.col_idx.begin(), A.col_idx.end(), col_idx_arr);
    std::copy(A.val    .begin(), A.val    .end(), val_arr    );
    std::copy(x.begin(),           x.end(),           x_arr   );
#ifdef _STREAM_
#ifdef _LB_
    Csr<float> P0 = read_csr_p<float>(0, dir);
    Csr<float> P1 = read_csr_p<float>(1, dir);

    // copy into partitions
    std::copy(P0.row_ptr.begin(), P0.row_ptr.end(), row_ptr_arr0);
	std::copy(P0.col_idx.begin(), P0.col_idx.end(), col_idx_arr0);
	std::copy(P0.val    .begin(), P0.val    .end(), val_arr0    );

    std::copy(P1.row_ptr.begin(), P1.row_ptr.end(), row_ptr_arr1);
	std::copy(P1.col_idx.begin(), P1.col_idx.end(), col_idx_arr1);
	std::copy(P1.val    .begin(), P1.val    .end(), val_arr1    );

    std::cout << P0.m << " " << P0.n << " " << P0.nnz << std::endl;
    std::cout << P1.m << " " << P1.n << " " << P1.nnz << std::endl;
    hls::stream<int>       row_ptr_strm("row_ptr");
	hls::stream<int>       col_idx_strm("col_idx");
	hls::stream<DATA_TYPE> val_strm("val");
	hls::stream<DATA_TYPE> y_strm("y");
    hls::stream<int>       row_ptr_strm1("row_ptr1");
	hls::stream<int>       col_idx_strm1("col_idx1");
	hls::stream<DATA_TYPE> val_strm1("val1");
	hls::stream<DATA_TYPE> y_strm1("y1");

	// 2) Stream in row_ptr (length M+1)
	for (int i = 1; i <= P0.m; i++) {
		row_ptr_strm.write(row_ptr_arr0[i]);
	}

	// 3) Stream in the NNZ entries
	for (int e = 0; e < P0.nnz; e++) {
		col_idx_strm.write(col_idx_arr0[e]);
		val_strm.write(val_arr0[e]);
	}
	for (int i = 1; i <= P1.m; i++) {
		row_ptr_strm1.write(row_ptr_arr1[i]);
	}

	// 3) Stream in the NNZ entries
	for (int e = 0; e < P1.nnz; e++) {
		col_idx_strm1.write(col_idx_arr1[e]);
		val_strm1.write(val_arr1[e]);
	}
#else
	hls::stream<int>       row_ptr_strm("row_ptr");
	hls::stream<int>       col_idx_strm("col_idx");
	hls::stream<DATA_TYPE> val_strm("val");
	hls::stream<DATA_TYPE> y_strm("y");
	for (int i = 1; i <= A.m; i++) {
		row_ptr_strm.write(row_ptr_arr[i]);
	}

	// 3) Stream in the NNZ entries
	for (int e = 0; e < A.nnz; e++) {
		col_idx_strm.write(col_idx_arr[e]);
		val_strm.write(val_arr[e]);
	}
#endif
#endif



    //int nnz_upd = pad_spmv(A.m, A.n, A.nnz);
    // ───────── invoke DUT (Device Under Test) ─────────
#ifdef __SPMV_OPT1_H__
    spmv_opt1(row_ptr_arr, col_idx_arr, val_arr, x_arr, y_hw, A.m, A.n, A.nnz);
#endif
	//spmv_opt4(row_ptr_arr0, col_idx_arr0, val_arr0, y_hw0, P0.m, P0.n, P0.nnz
	//	,row_ptr_arr1, col_idx_arr1, val_arr1, y_hw1, P1.m, P1.n, P1.nnz, x_arr);
#ifdef __SPMV_OPT3_H__
    spmv_opt3(row_ptr_arr, col_idx_arr, val_arr, x_arr, y_hw, A.m, A.n, A.nnz);
#endif
#ifdef _STREAM_VEC_
    int nnz;
    spmv_pad_vec(A, row_pad_vec_strm, col_idx_vec_strm, val_vec_strm, nnz);
    spmv_stream_vec(row_pad_vec_strm, col_idx_vec_strm, val_vec_strm, x_arr, y_strm, A.m, A.n, nnz);
    for (int i = 0; i < A.m; i++) {
    		y_hw[i] = y_strm.read();
	}
#endif
#ifdef _STREAM_

#ifdef _LB_
	spmv_stream_lb(row_ptr_strm, col_idx_strm, val_strm, y_strm, P0.m, P0.nnz, row_ptr_strm1, col_idx_strm1, val_strm1, y_strm1, P1.m,  P1.nnz, A.n, x_arr);
	for (int i = P0.row_start; i <= P0.row_end; i++) {
	        y_hw[i] = y_strm.read();
	}
	for (int i = P1.row_start; i <= P1.row_end; i++) {
	        y_hw[i] = y_strm1.read();
	}

#else

#ifdef __SPMV_OPT2S_H__
	//
	spmv_naive_stream_opt2_stream(row_ptr_strm, col_idx_strm, val_strm, x_arr, y_strm, A.m, A.n, A.nnz);
#else
	spmv_stream(row_ptr_strm, col_idx_strm, val_strm, x_arr, y_strm, A.m, A.n, A.nnz);
#endif
		printf("End function\n");
	for (int i = 0; i < A.m; i++) {
		y_hw[i] = y_strm.read();
	}
#endif
#endif
    //spmv_opt2(row_ptr_arr, col_idx_arr, val_arr, x_arr, y_hw, A.m, A.n, A.nnz);

    //spmv_opt4(row_ptr_arr, col_idx_arr, val_arr, x_arr, y_hw, A.m, A.n, A.nnz);

	/*
    for (int i = P0.row_start, j = 0; i <= P0.row_end; i++) {
    	y_hw[i] = y_hw0[j++];
    }
    for (int i = P1.row_start, j = 0; i <= P1.row_end; i++) {
    	y_hw[i] = y_hw1[j++];
    }*/
    // ───────── software reference ─────────
    std::fill_n(y_gold, A.m, 0.0f);
    for (int r = 0; r < A.m; ++r)
        for (int k = row_ptr_arr[r]; k < row_ptr_arr[r+1]; ++k)
            y_gold[r] += val_arr[k] * x_arr[col_idx_arr[k]];

    // ───────── compare ─────────

    const float eps = 1e-3f;
    int errors = 0;
    for (int i = 0; i < A.m; ++i)
        if (std::fabs(y_gold[i] - y_hw[i]) > eps) {
            if (++errors < 10)   // print only first few mismatches
                std::cerr << "[mismatch] row " << i
                          << "  gold=" << y_gold[i]
                          << "  hw="   << y_hw[i] << '\n';
      }

    /*
    const float eps = 1e-3f;
    int errors = 0;
    for (int i = P0.row_start, j = 0; i <= P0.row_end; ++i, ++j)
        if (std::fabs(y_gold[i] - y_hw0[j]) > eps) {
            if (++errors < 10)   // print only first few mismatches
                std::cerr << "[mismatch] row " << i
                          << "  gold=" << y_gold[i]
                          << "  hw="   << y_hw0[j] << '\n';
        }
    for (int i = P1.row_start, j = 0; i <= P1.row_end; ++i, ++j)
            if (std::fabs(y_gold[i] - y_hw1[j]) > eps) {
                if (++errors < 10)   // print only first few mismatches
                    std::cerr << "[mismatch] row " << i
                              << "  gold=" << y_gold[i]
                              << "  hw="   << y_hw1[j] << '\n';
            }
            */
    if (errors == 0)
        std::cout << "✓ PASS – HW matches SW reference.\n";
    else
        std::cout << "✗ FAIL – " << errors << " mismatches.\n";

    return errors ? 1 : 0;
}
