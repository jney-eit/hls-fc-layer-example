#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "config.hpp"
#include "fc_layer.hpp"



void fc_example_top(
        hls::stream<ap_int<SIMD * in_t::width>> &in_s,
		hls::stream<ap_int<PE * out_t::width>> &out_s,
        w_t weights[PE][SIMD][SIMD_FOLD * PE_FOLD],
        b_t bias[PE][PE_FOLD]
){

    #pragma HLS INTERFACE axis port=in_s
    #pragma HLS INTERFACE axis port=out_s
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=bias bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS ARRAY_PARTITION variable=weights dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=weights dim=2 type=complete
    #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete


    Fc<in_t, out_t, NUM_INPUTS, NUM_NEURONS, SIMD, PE>(
        in_s,
        out_s,
        weights, 
        bias
    );


}