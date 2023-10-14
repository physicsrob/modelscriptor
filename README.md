![CI](https://github.com/physicsrob/modelscriptor/actions/workflows/ci.yml/badge.svg)

# ModelScriptor

This project is a work in progress.


## TODO List
This list is a non-exhaustive list of all the things that needs to be done before this project is relatively feature complete.

- [ ] Implement embedding / deembedding compilation
- [ ] Separate tokenizer from embedding
- [ ] Export to ONNX

Recently complete:
- [x] Implement attn compilation in attention sublayer
- [x] Refactor compilation scoring
- [x] Include compilation statistics (number of parameters, efficiency, etc)
- [x] Better pass-through support for attention layers; We shouldn't require one head per zero.
