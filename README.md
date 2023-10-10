![CI](https://github.com/physicsrob/modelscriptor/actions/workflows/ci.yml/badge.svg)

# ModelScriptor

This project is a work in progress.


## TODO List
This list is a non-exhaustive list of all the things that needs to be done before this project is relatively feature complete.

- [ ] Test that linear can implement constant
- [ ] Implement `add` in linear, currently only implemented in skip connections.
- [ ] Implement linear compilation in attention sublayer
- [ ] Implement `add` in attention sublayer
- [ ] Implement embedding / deembedding compilation
- [ ] Export to ONNX
- [ ] Include compilation statistics (number of parameters, efficiency, etc)
- [ ] Add support for compiling Zero as Add(Unallocated, -Unallocated)
- [ ] Better pass-through support for attention layers; We shouldn't require one head per zero.

Recently complete:
- [x] Implement attn compilation in attention sublayer
- [x] Refactor compilation scoring
