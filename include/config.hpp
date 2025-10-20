#pragma once 

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>


// Datatypes
typedef ap_fixed<16, 8> in_t;
typedef ap_fixed<16, 8> out_t;
typedef ap_fixed<16, 8> w_t;
typedef ap_fixed<16, 8> b_t;

// Layer Dimensions
constexpr int NUM_INPUTS = 4;
constexpr int NUM_NEURONS = 2;

// NUM_INPUTS needs to be multiple of SIMD
constexpr int SIMD = 2;
// NUM_NEURONS needs to be multiple of PE
constexpr int PE = 2;


// Derived Constants - Do not mofify 
constexpr int SIMD_FOLD = NUM_INPUTS / SIMD;
constexpr int PE_FOLD = NUM_NEURONS / PE;
