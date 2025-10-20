Vitis High-Level-Synthesis implementation of a simple fully-connected layer with a variable degree of parallelism

fc_layer_example/tb.cpp is the testbench file to simulate the hardware 
fc_layer_example/top.cpp contains the top module, which is synthesised and defines top-level interfaces 
include/config.hpp allows to configure the fully-connected layer, e.g. the number of inputs, the number of neurons, the parallelism, and the datatypes 
include/fc_layer.hpp contains the HLS implementation of the fully-connected layer 
