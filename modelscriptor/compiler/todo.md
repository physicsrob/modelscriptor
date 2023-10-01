# Must do:
1. Add should be implemented by linear layers, currently only implemented in skip connections.
1. Implement compilation of concat layers
1. Implement Attention compilation
1. Implement embedding/deembedding layer components
1. Implement full transformer compilation
1. More compilation tests

# Optimizations
1. Constants can be implemented in FFNSubLayer
    1. Use unused allocation in residual stream.
    1. Subtract incoming unused allocation, add bias for constant

# Other
1. Export to ONNX
