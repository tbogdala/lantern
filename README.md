# Lantern

A utility crate providing higher level functionality built on top of [Candle](https://github.com/huggingface/candle).


## Status

Very early development and much of the public API is subject to change. Open issues for requests.


## Features

- [x] `TextGeneratorManager` to handle text generation requests with a LLM on a worker thread
- [x] More robust LLM sampling with `TextGeneratorSampler`


## Buiding From Source

Simply cloning the repository and choosing a hardware acceleration feature should be enough to build.

```bash
git clone https://github.com/tbogdala/lantern.git
cd lantern
cargo build --release --features cuda
```

If you have a Mac, build with the `metal` feature instead and use that for running tests as well. 
Those without the ability to run either `metal` or `cuda` should still be able to compile and run
the library in CPU mode.


## Running Tests

The preferred way to run the unit tests is with `--nocapture` so that the tokens are visible as they're generated.

```bash
RUST_LOG=debug cargo test --release --features cuda -- --nocapture
```

## Notes:

- Passing bad hugging face ids will result in the library timing out on the HTTP requests
  to download the required files and it will appear like the program stalled. If you
  suspect this, you can set `RUST_LOG=debug` in the shell to see extended log information.


## TODO:

### Text Generation TODO: 

- [ ] Handle context overflowing better
- [ ] Add custom stop tokens
- [ ] Keep last used model in memory
- [ ] Add grammar support to sampling a la llama.cpp 
- [ ] Attempt to add controllable number of layers to offload to GPU

(... and so much more ...)

## License

MIT to match Candle.