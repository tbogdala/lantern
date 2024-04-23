# Lantern (alpha version)

A utility crate providing higher level functionality built on top of [Candle](https://github.com/huggingface/candle).


## Status

Very early development and much of the public API is subject to change. Open issues for requests.
The dependences for Candle are snagged straight from git for now while


## Features

- [x] `TextGeneratorManager` to handle text generation requests with a LLM on a worker thread
- [x] More robust LLM sampling with `TextGeneratorSampler`
- [x] Enhanced quantized llama handling compared to candle's impelmentation: can processed prompts in 
      batches, max context length is pulled from the gguf file.


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


## Running Examples

Examples can be run simply through cargo. Both examples and tests be configured to look at environment variables
for the models to use. For example the following four environment variables can override the default settings
in examples and tests:

* `LANTERN_MODEL_ID`: the huggingface model repo id to use
* `LANTERN_MODEL_FILE`: the file name within the model repo to download
* `LANTERN_TOKENIZER_ID`: the huggingface repo id that has the `tokenizer.json` file for the model
* `LANTERN_EOS_STRING`: the EOS string to detect end of prediction; must be able to be tokenized.

The following is how to run the basic example with `cuda` accelleration; switch to `metal` on Mac.

```bash
cargo run --example basic --release --features cuda
```


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
- [ ] Keep last used model in memory
- [ ] Add grammar support to sampling a la llama.cpp 
- [ ] Attempt to add controllable number of layers to offload to GPU
- [ ] Customizable prompt builder for different format rules

(... and so much more ...)


## License

MIT to match Candle.