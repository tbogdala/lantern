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
- [x] Easier abstraction to use image generators based on stable diffusion. Currently supports
      SD v1.5 and SDXL Turbo.


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


### Basic Text Generation

The following is how to run the basic example with `cuda` accelleration; switch to `metal` on Mac.

```bash
cargo run --example basic --release --features cuda
```

The following is the block of text that was the output of running this command (block formatting added manually:

```
Default Prompt:

Explain large language models and the LLaMa 3 model architecture like I'm
five years old and can only understand simple concepts.

Generated Answer:

Oh boy, I'm excited to explain something cool to you!

So, you know how we can talk and understand what each other is saying?
Like, you can say "I like ice cream" and I can understand what you mean?

Well, computers don't understand human language like we do. They can only
understand special codes that they can read. But, some really smart people
have made special computers that can understand and talk like humans!
These computers are called "large language models".

Large language models are like super smart friends that can have conversations
with you. They can understand what you say and respond in a way that makes
sense. They can even write their own stories, poems, or even whole books!

Now, let me tell you about the LLaMa 3 model architecture. This is like the
special blueprint for making these super smart friends.

Imagine you have a big box full of blocks, and each block has a letter on it
(like "A", "B", "C", etc.). The LLaMa 3 model is like a special machine that
takes all those blocks and arranges them into words and sentences.

The machine has three special parts:

1. The first part is like a super smart librarian. It takes the blocks and
sorts them into categories, like all the "A" blocks together, all the "B"
blocks together, and so on.

2. The second part is like a super clever writer. It takes the sorted blocks
and uses them to create sentences and paragraphs.

3. The third part is like a super fast typist. It writes down all the sentences
and paragraphs it creates.

The LLaMa 3 model is special because it can do all these things really, really
well. It's like having a team of super smart friends working together to create
amazing things!

So, that's what large language models and the LLaMa 3 model architecture are like!
Pretty cool, right?<|eot_id|>
```

On a Macbook Air M3 with 24 gb of memory, this block is generated at ~7 tokens/s.

Should you wish to use a prompt in a file, and you use an OS with a real shell, you can pass the
prompt in as `--prompt "$(cat ~/test_prompt.txt)"`. If it's long, you can even batch process the prompts
with the `--batch-size` parameter. All together, it might look like this:

```bash
cargo run --example basic --release --features cuda -- --prompt "$(cat ~/long_test_prompt.txt)" --batch-size 256
```


### Basic Image Generation

The diffusion example takes a few more parameters to get right. If you specify the same `--save-as` output file, the example will save
the new generation with an iteration number on the filename. The sample defaults to using `SDXL Turbo` at it's trained size
of 512x512. `--cfg` should be left at 0.0 and it's trained to work on `--steps` values of 1, 2, 3 or 4.

```bash
cargo run --release --features cuda --example diffusion -- -p "A surreal three-quarter angle photograph of a genius husky dog, wearing glasses, in a spacesuit, taking a picture for a photo ID. Hi-tech equipment. Derpy husky drama face." --steps 3 --save-as "out-images/husky-id.png"
```

This is an image that was generated with the above command:

![sdxl turbo sample image showing a husky dog as an astronaut](https://github.com/tbogdala/lantern/blob/c2c7ad629f3765464a3ad22bdb2205bbb4f57896/assets/SDXLTurboSample512.png)

On a Macbook Air M3 with 24 gb of memory, each step takes about ~7.2s to iterate over the steps and aonther 9.1s to decode the iamge in the VAE. 

If you want to see what Stable Diffusion 1.5 looked like stock, you can send the same prompt but give more `--steps` (30-50) and change the `--cfg` to something other than zero (~7.5).

```bash
cargo run --release --features cuda --example diffusion -- -p "A surreal three-quarter angle photograph of a genius husky dog, wearing glasses, in a spacesuit, taking a picture for a photo ID. Hi-tech equipment. Derpy husky drama face." --steps 30 --cfg 5.5 --sd-ver v1-5 --save-as "out-images/husky-id.png"
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
