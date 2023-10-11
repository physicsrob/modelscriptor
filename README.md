![CI](https://github.com/physicsrob/modelscriptor/actions/workflows/ci.yml/badge.svg)

# ModelScriptor

This project is a work in progress.


## TODO List
This list is a non-exhaustive list of all the things that needs to be done before this project is relatively feature complete.

- [ ] Include compilation statistics (number of parameters, efficiency, etc)
- [ ] Better pass-through support for attention layers; We shouldn't require one head per zero.
- [ ] Add support for compiling Zero as Add(Unallocated, -Unallocated)
- [ ] Test that linear can implement constant
- [ ] Implement `add` in linear, currently only implemented in skip connections.
- [ ] Implement linear compilation in attention sublayer
- [ ] Implement `add` in attention sublayer
- [ ] Implement embedding / deembedding compilation
- [ ] Export to ONNX

Recently complete:
- [x] Implement attn compilation in attention sublayer
- [x] Refactor compilation scoring
