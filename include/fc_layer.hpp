#pragma once 
#include <cassert>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

/**
 * Fully Connected layer with streaming interface and ajustable parallelization (Pe, Simd)
 */
template<
	typename In_t,
	typename Out_t,
	int NumInputs,
	int NumNeurons,
	int Simd,
	int Pe,
	typename Weight_t,
	typename Bias_t
>
void Fc(
		hls::stream<ap_int<Simd * In_t::width>> &input_stream,
		hls::stream<ap_int<Pe * Out_t::width>> &output_stream,
		Weight_t weights[Pe][Simd][(NumInputs/Simd)*(NumNeurons/Pe)],
		Bias_t bias[Pe][NumNeurons/Pe])
{

	assert((NumInputs % Simd == 0) && "Simd is not divider of NumInputs");
	assert((NumNeurons % Pe == 0) && "Pe is not divider of NumNeurons");

    constexpr unsigned simd_fold = NumInputs / Simd;
    constexpr unsigned pe_fold = NumNeurons / Pe;

	Out_t accumulator[Pe][pe_fold];
	#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1

	// Init accumulator with bias
	loop_bias_pe_fold:for(unsigned pe_f = 0; pe_f < pe_fold; pe_f++){
		#pragma HLS PIPELINE

		loop_bias_pe:for(unsigned pe_count = 0; pe_count < Pe; pe_count++){
			#pragma HLS UNROLL
			accumulator[pe_count][pe_f] = static_cast<Out_t>(bias[pe_count][pe_f]);
		}
	}

	// Mul weights by input and accumulate
	ap_int<Simd * In_t::width> inputs;
	loop_simd_fold:for(unsigned simd_f = 0; simd_f < simd_fold; simd_f++){
		loop_pe_fold:for(unsigned pe_f = 0; pe_f < pe_fold; pe_f++){
			#pragma HLS PIPELINE

			if(pe_f == 0){
				inputs = input_stream.read();
			}

			loop_pe:for(unsigned pe_count = 0; pe_count < Pe; pe_count++){
				#pragma HLS UNROLL

				Out_t acc_temp = accumulator[pe_count][pe_f];

				loop_simd:for(unsigned simd_count = 0; simd_count < Simd; simd_count++){
					#pragma HLS UNROLL

					ap_int<In_t::width> input_temp = inputs((simd_count + 1) * In_t::width - 1, simd_count * In_t::width);
					In_t input = *reinterpret_cast<In_t *>(&input_temp);

					Weight_t weight = weights[pe_count][simd_count][pe_f * simd_fold + simd_f];

					Out_t mul = input * weight;
					acc_temp += mul;

				}
				accumulator[pe_count][pe_f] = acc_temp;
			}
		}
	}

	// Write output
	loop_output_ne:for(unsigned pe_f = 0; pe_f < pe_fold; pe_f++){
		#pragma HLS PIPELINE
		ap_int<Pe * Out_t::width> output;
		loop_output_pe:for(unsigned pe_count = 0; pe_count < Pe; pe_count++){
			#pragma HLS UNROLL
			Out_t output_temp = accumulator[pe_count][pe_f];

			output((pe_count+1) * Out_t::width - 1, pe_count * Out_t::width) = *reinterpret_cast<ap_int<Out_t::width> *>(&output_temp);
		}
		output_stream.write(output);
	}
}
