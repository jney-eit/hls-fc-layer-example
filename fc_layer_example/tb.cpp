#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <cmath>   // For fabs()
#include "config.hpp"

// Forward declaration of the top-level function (Device Under Test)
void fc_example_top(
        hls::stream<ap_int<SIMD * in_t::width>> &in_s,
		hls::stream<ap_int<PE * out_t::width>> &out_s,
        w_t weights[PE][SIMD][SIMD_FOLD * PE_FOLD],
        b_t bias[PE][PE_FOLD]
);

// --- Golden Model for Verification ---
// A simple, standard C++ implementation of a fully connected layer.
void fc_golden(
    const std::vector<float>& inputs,
    const std::vector<std::vector<float>>& weights,
    const std::vector<float>& biases,
    std::vector<float>& outputs
) {
    for (int n = 0; n < NUM_NEURONS; ++n) {
        outputs[n] = biases[n];
        for (int i = 0; i < NUM_INPUTS; ++i) {
            outputs[n] += inputs[i] * weights[n][i];
        }
    }
}

// --- Data Reordering Functions ---
// This is CRITICAL. It converts a standard [neuron][input] weight matrix
// into the specific memory layout required by the parallel HLS kernel.
void reorder_weights(
    const std::vector<std::vector<float>>& golden_weights,
    w_t hls_weights[PE][SIMD][SIMD_FOLD * PE_FOLD]
) {
    for (int n = 0; n < NUM_NEURONS; ++n) {
        for (int i = 0; i < NUM_INPUTS; ++i) {
            // Deconstruct indices
            int pe_count = n % PE;
            int pe_fold_idx = n / PE;
            int simd_count = i % SIMD;
            int simd_fold_idx = i / SIMD;

            // Calculate the HLS memory index
            int hls_mem_idx = pe_fold_idx * SIMD_FOLD + simd_fold_idx;

            // Assign the weight, converting from float to the HLS fixed-point type
            hls_weights[pe_count][simd_count][hls_mem_idx] = (w_t)golden_weights[n][i];
        }
    }
}

// Reorders the bias vector for parallel access by PE units.
void reorder_biases(
    const std::vector<float>& golden_biases,
    b_t hls_biases[PE][PE_FOLD]
) {
    for (int n = 0; n < NUM_NEURONS; ++n) {
        int pe_count = n % PE;
        int pe_fold_idx = n / PE;
        hls_biases[pe_count][pe_fold_idx] = (b_t)golden_biases[n];
    }
}


int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "Starting Testbench for FC Layer..." << std::endl;
    std::cout << "Configuration: " << std::endl;
    std::cout << "  NUM_INPUTS  = " << NUM_INPUTS << std::endl;
    std::cout << "  NUM_NEURONS = " << NUM_NEURONS << std::endl;
    std::cout << "  SIMD        = " << SIMD << std::endl;
    std::cout << "  PE          = " << PE << std::endl;
    std::cout << "================================================================" << std::endl;

    // --- 1. Data Generation ---
    srand(123); // Use a fixed seed for reproducibility

    std::vector<float> golden_inputs(NUM_INPUTS);
    std::vector<std::vector<float>> golden_weights(NUM_NEURONS, std::vector<float>(NUM_INPUTS));
    std::vector<float> golden_biases(NUM_NEURONS);
    std::vector<float> golden_outputs(NUM_NEURONS, 0.0f);
    std::vector<float> hls_outputs(NUM_NEURONS);

    // Generate random data between -1.0 and 1.0
    for (int i = 0; i < NUM_INPUTS; ++i) {
        golden_inputs[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
    for (int n = 0; n < NUM_NEURONS; ++n) {
        golden_biases[n] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        for (int i = 0; i < NUM_INPUTS; ++i) {
            golden_weights[n][i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        }
    }

    // --- 2. Run Golden Model ---
    std::cout << "Running Golden C++ Model..." << std::endl;
    fc_golden(golden_inputs, golden_weights, golden_biases, golden_outputs);

    // --- 3. Prepare Data for HLS DUT ---
    hls::stream<ap_int<SIMD * in_t::width>> input_stream("input_stream");
    hls::stream<ap_int<PE * out_t::width>> output_stream("output_stream");

    // Reorder weights and biases into the HLS-specific layout
    static w_t hls_weights[PE][SIMD][SIMD_FOLD * PE_FOLD];
    static b_t hls_biases[PE][PE_FOLD];
    reorder_weights(golden_weights, hls_weights);
    reorder_biases(golden_biases, hls_biases);

    // Pack inputs into the stream
    for (unsigned i = 0; i < SIMD_FOLD; ++i) {
        ap_int<SIMD * in_t::width> packed_input;
        for (unsigned j = 0; j < SIMD; ++j) {
            in_t val = (in_t)golden_inputs[i * SIMD + j];
            // Place the value into the correct slice of the wide ap_int
            packed_input.range((j + 1) * in_t::width - 1, j * in_t::width) = *reinterpret_cast<ap_int<in_t::width>*>(&val);
        }
        input_stream.write(packed_input);
    }

    // --- 4. Call the HLS DUT ---
    std::cout << "Running HLS DUT (fc_top)..." << std::endl;
    fc_example_top(input_stream, output_stream, hls_weights, hls_biases);
    std::cout << "HLS DUT execution finished." << std::endl;

    // --- 5. Unpack HLS Output and Verify ---
    int errors = 0;
    float max_diff = 0.0f;
    // The tolerance depends on your fixed-point precision.
    // It's related to the smallest representable value (LSB).
    const float tolerance = 0.01;

    for (unsigned i = 0; i < PE_FOLD; ++i) {
        ap_int<PE * out_t::width> packed_output = output_stream.read();
        for (unsigned j = 0; j < PE; ++j) {
            ap_int<out_t::width> temp = packed_output.range((j + 1) * out_t::width - 1, j * out_t::width);
            out_t val = *reinterpret_cast<out_t*>(&temp);

            int output_idx = i * PE + j;
            hls_outputs[output_idx] = (float)val;

            float diff = fabs(golden_outputs[output_idx] - hls_outputs[output_idx]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tolerance) {
                errors++;
                std::cout << "ERROR: Mismatch at output[" << output_idx << "]: "
                          << "Golden = " << golden_outputs[output_idx]
                          << ", HLS = " << hls_outputs[output_idx]
                          << ", Diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "Verification Summary:" << std::endl;
    std::cout << "  Maximum difference between Golden and HLS: " << max_diff << std::endl;

    if (errors == 0) {
        std::cout << "  TEST PASSED!" << std::endl;
    } else {
        std::cout << "  TEST FAILED! Found " << errors << " mismatches." << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    return (errors == 0) ? 0 : 1;
}