# Optimizations
1. Constants can be implemented in FFNSubLayer
    1. Use unused allocation in residual stream.
    1. Subtract incoming unused allocation, add bias for constant
1. Add could be implemented in linear layers, currently only implemented in skip connections.

