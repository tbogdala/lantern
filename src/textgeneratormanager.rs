use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama as model;
use crossbeam::channel::{Receiver, Sender};
use hf_hub::api::sync::Api;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::TextGeneratorSampler;

/// TextGenerationParams is a structure that contains all the options for generating text
/// from a large language model. The bulk of the parameters control sampling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextGenerationParams {
    /// The maximum number of tokens to sample. A result less than this count can still be produced
    /// if the stop token is encountered earlier than that.
    pub to_sample: usize,

    /// The temperature value, when different from 1.0f, will change the probability distribution of the
    /// output tokens. Temperature > 1.0f makes less likely tokens more likely to be chosen and temperature
    /// values < 1.0f make more common tokens more likely to be chosen.
    /// Setting to <= 0.0 disables other samplers and just does greedy sampling of most probable tokens.
    pub temperature: f32,

    /// Top-p sampling restricts the possibilities to a subset of tokens that cumulatively add up to a
    /// probability of at least the value of top_p provided.
    /// Setting to 1.0 will disable this sampler.
    pub top_p: f32,

    /// Min-p sampling restricts the possibilities to a subset of tokens that are at least as probable
    /// as the value of min_p, relative to the most likely token's probability. For example, if the
    /// most probable token has a probability of 0.9, then a min_p value of 0.05 means that only tokens
    /// with a probability of at least 0.045 are considered.
    /// Setting to 0.0 will disable this sampler.
    pub min_p: f32,

    /// Top-k sampling restricts the possibilities to the first top_k number of most likely tokens.
    /// Setting to 0 will disable this sampler.
    pub top_k: usize,

    /// Repeat penalty helps prevent the model from generating repetitive or monotonous text. Values greater
    /// than 1.0 penalize repetitions more strongly while a lower value will be more lenient.
    pub repeat_penalty: f32,

    /// Repeat_last_n controls the number of tokens in the history to consider while applying the repeat penalty.
    pub repeat_last_n: usize,

    /// The seed value used to initializes the random number generator and setting it to a constant value
    /// should yield reproducable results. No behavior is applied to the value, so the applicaiton *must*
    /// seed this appropriately if deterministic behavior is not desired.
    pub seed: u64,

    /// This is a collection of strings that will stop the generation of text. Note: they will not get removed
    /// from the generated string, because the individual tokens have already been sent out, and it will
    /// be the client code's responsibility to remove it from the whole string.
    pub stop_strings: Vec<String>,

    /// The text prompt to initialize the text generator with as its context for generating a response.
    pub user_prompt: String,
}
impl Default for TextGenerationParams {
    fn default() -> Self {
        Self {
            to_sample: 128,
            temperature: 1.1,
            top_p: 1.0,
            min_p: 0.08,
            top_k: 0,
            repeat_penalty: 1.05,
            repeat_last_n: 64,
            seed: 42,
            stop_strings: Vec::new(),
            user_prompt: "".into(),
        }
    }
}

/// Represents different types of updates coming from the worker thread during text generation.
/// Once the `Finished` value is received, no more `Token` messages should be generated until
/// the next request has been started.
#[derive(Debug, Clone, PartialEq)]
pub enum TextGenUpdate {
    /// A single token of the text generation response converted to a string
    Token(String),

    /// The whole string of the text generation response that is completed
    Finished(String),
}

/// Represents the different types of errors that can happen during text generation.
#[derive(Debug)]
pub enum TextGenerationErrors {
    /// This error value is returned when the worker thread is already busy with a request
    BusySignal,
}
impl std::fmt::Display for TextGenerationErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BusySignal => {
                write!(f, "The worker thread is already busy with another request.")
            }
        }
    }
}

/// This is the manager object that acts as the controller for the worker thread that gets
/// spawned behind-the-scenes on each `generate_text` request. The exposed functions simplify
/// the work needed to interface between the threads.
pub struct TextGeneratorManager {
    busy_signal: Arc<Mutex<bool>>,
    send: Sender<TextGenUpdate>,
    recv: Receiver<TextGenUpdate>,

    tokens_to_predict: usize,
    tokens_returned: usize,
}
impl TextGeneratorManager {
    /// Create a new instance of the manager with default values.
    pub fn new() -> Self {
        let (send, recv) = crossbeam::channel::unbounded();
        TextGeneratorManager {
            busy_signal: Arc::new(Mutex::new(false)),
            send,
            recv,
            tokens_to_predict: 0,
            tokens_returned: 0,
        }
    }

    /// Will return a TextGenUpdate if one is available from the worker thread or None if there isn't.
    pub fn maybe_get_update(&mut self) -> Option<TextGenUpdate> {
        if let Ok(update) = self.recv.try_recv() {
            self.tokens_returned += 1;
            Some(update)
        } else {
            None
        }
    }

    /// Will return a TextGenUpdate from the worker thread, blocking if necessary.
    pub fn get_update(&mut self) -> TextGenUpdate {
        let update = self.recv.recv().unwrap();
        self.tokens_returned += 1;
        update
    }

    /// Checks the internal 'busy signal' to see if there's a text generation job already running.
    /// Client code should wait until `is_busy` returns false before calling `generate_text`.
    pub fn is_busy(&self) -> bool {
        let busy = self.busy_signal.lock().unwrap();
        *busy
    }

    /// Returns a tuple of (current predicted token count, total count to be predicted).
    pub fn get_progress(&self) -> (usize, usize) {
        (self.tokens_returned, self.tokens_to_predict)
    }

    /// Starts a worker thread to handle the text generation and flips the 'busy_signal' mutex
    /// so that only one job can run at a time. This function will return a `TextGenerationErrors::BusySignal`
    /// error if the worker thread is already busy with a request.
    pub fn generate_text(
        &mut self,
        model_id: String,
        model_file: String,
        tokenizer_repo_id: String,
        eos_token_str: String,
        params: TextGenerationParams,
    ) -> Result<()> {
        {
            let mut busy = self.busy_signal.lock().unwrap();
            if *busy == true {
                warn!("Unable to process text generation while already busy.");
                return Err(anyhow!(TextGenerationErrors::BusySignal));
            }

            // set the busy signal
            *busy = true;
        }

        let sender_clone = self.send.clone();
        let busy_signal_clone = self.busy_signal.clone();

        // make sure to clear the progress tracking
        self.tokens_to_predict = params.to_sample;
        self.tokens_returned = 0;

        std::thread::spawn(move || {
            let _ = worker_generate_text(
                model_id.as_str(),
                model_file.as_str(),
                tokenizer_repo_id.as_str(),
                eos_token_str.as_str(),
                &params,
                sender_clone,
            );
            {
                // clear the busy signal
                let mut busy = busy_signal_clone.lock().unwrap();
                *busy = false;
            }
        });

        Ok(())
    }
}

/// worker function for text generation; does all the actual work.
fn worker_generate_text(
    model_id: &str,
    model_file: &str,
    tokenizer_repo_id: &str,
    eos_token_str: &str,
    params: &TextGenerationParams,
    sender: Sender<TextGenUpdate>,
) -> Result<String> {
    debug!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    // get the model filepath from the cache
    let api = Api::new().context("Attempting to create Huggingface API endpoint")?;
    let repo = api.model(model_id.into());
    let model_filepath = repo
        .get(model_file)
        .context("Attempted to get the model weights filepath")?;
    debug!("Using model weights from file: {:?} ...", model_filepath);

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("Creating GPU Cuda device for Candle")?;
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).context("Creating GPU Metal device for Candle")?;
    #[cfg(not(feature = "cuda"))]
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    debug!(
        "Device is: cpu={} | cuda={} | metal={}",
        device.is_cpu(),
        device.is_cuda(),
        device.is_metal()
    );

    // NOTE: Only working with gguf...
    let mut file = std::fs::File::open(&model_filepath)
        .context("Attempting to open the model weights file")?;
    let model =
        gguf_file::Content::read(&mut file).context("Attempting to read the model weights file")?;
    let mut model = model::ModelWeights::from_gguf(model, &mut file, &device)
        .context("Processing model weights")?;
    debug!("Model built successfully.");

    // setup tokenizer
    let tokenizer_repo = api.model(tokenizer_repo_id.into());
    let tokenizer_filepath = tokenizer_repo
        .get("tokenizer.json")
        .context("Attempting to get the tokenizer filepath")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filepath)
        .map_err(anyhow::Error::msg)
        .context("Processing model tokenizer")?;
    debug!("Tokenizer deserialized.");

    // NOTE: no safeguarding about overrunning max seq length with sample count!
    let prompt = tokenizer
        .encode(params.user_prompt.clone(), true)
        .map_err(anyhow::Error::msg)
        .context("Tokenizing prompt")?;
    let prompt_tokens = prompt.get_ids();

    // setup an EOS token
    let eos_token = *tokenizer
        .get_vocab(true)
        .get(eos_token_str)
        .context("Attempting to get the EOS token")
        .unwrap();

    // output the prompt tokens to debug log
    // let prompt_token_strings = prompt.get_tokens();
    // for i in 0..prompt_tokens.len() {
    //     debug!(
    //         "Prompt token index {i} is token {} ({})",
    //         prompt_tokens[i], prompt_token_strings[i]
    //     );
    // }
    debug!(
        "Finished tokenization process for {} tokens.",
        prompt_tokens.len()
    );

    // TODO: adjust params.to_sample to not overflow model::MAX_SEQ_LEN

    let mut all_tokens = vec![];
    let mut logits_processor = TextGeneratorSampler::new(params.seed);

    let start_prompt_processing = std::time::Instant::now();

    // process prompt in one shot
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    debug!("Finished prompt processing");

    // prime the loop with a single token generation
    let mut next_token = logits_processor.sample(params, &logits)?;
    all_tokens.push(next_token);
    let prompt_dt = start_prompt_processing.elapsed();
    debug!("Loop primed with prompt logits.");

    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 1;
    let mut prev_full_decode = String::new();
    if let Some(new_decode) = send_token_update(&tokenizer, &all_tokens, &prev_full_decode, &sender)
    {
        prev_full_decode = new_decode;
    }

    for index in 0..params.to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if params.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(params.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                params.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(params, &logits)?;
        all_tokens.push(next_token);
        sampled += 1;

        // see if we can decode the token to send it out in an update
        if let Some(new_decode) =
            send_token_update(&tokenizer, &all_tokens, &prev_full_decode, &sender)
        {
            prev_full_decode = new_decode;
        }

        if next_token == eos_token {
            break;
        };

        // are we checking for additional stop strings?
        if !params.stop_strings.is_empty(){
            // decode a partial string of the response to see if any of the stop_strings match
            const STOPWORD_CHECK_DISTANCE: usize = 32;
            let last_few_tokens = &all_tokens[all_tokens.len().saturating_sub(STOPWORD_CHECK_DISTANCE)..];
            let last_few_tokens_string = tokenizer.decode(last_few_tokens, true)
                .map_err(anyhow::Error::msg)
                .context("Decodng last few tokens to check for stop strings");
            if let Ok(last_few_decoded) = last_few_tokens_string {
                let mut halt_generation = false;
                for stopper in &params.stop_strings {
                    if last_few_decoded.ends_with(stopper) {
                        debug!("Stop string \"{}\" found. Halting text generation", stopper);
                        halt_generation = true;
                        break;
                    }
                }
                if halt_generation {
                    break;
                }
            }
        }

    }
    let dt = start_post_prompt.elapsed();
    debug!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    debug!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );

    // writing out all the generated tokens to debug log
    // for i in 0..all_tokens.len() {
    //     let t_str = tokenizer.decode(&all_tokens[i..i + 1], false).unwrap();
    //     debug!(
    //         "generated token index {i} is token {} ({})",
    //         all_tokens[i], t_str
    //     );
    // }

    let whole_string = tokenizer
        .decode(&all_tokens[00..all_tokens.len()], false)
        .map_err(anyhow::Error::msg)
        .context("Decoding the predicted text");

    let decoded_string = if let Ok(decoded) = &whole_string {
        debug!("Whole predicted text is:");
        debug!("{}", decoded);
        decoded
    } else {
        "Failed to decode generated string!"
    };

    if let Err(err) = sender.send(TextGenUpdate::Finished(decoded_string.to_string())) {
        error!(
            "Error while sending finish message to text generator channel: {}",
            err
        );
    }

    whole_string
}

/// decodes the tokens and sends out the updated part of the string as the new token
/// along the Sender channel and then messages the UI context to update. Returns the
/// new decoded string if the dedoce process was successful or None otherwise.
fn send_token_update(
    tokenizer: &Tokenizer,
    all_tokens: &Vec<u32>,
    prev_full_decode: &String,
    sender: &Sender<TextGenUpdate>,
) -> Option<String> {
    // see if we can decode the token to send it out in an update
    if let Ok(update_str) = tokenizer.decode(&all_tokens, false) {
        // Gonna do this in the most naive, braindead way possible: decode the whole incoming
        // buffer and then send new tail off. Doing the decode one at a time skips all the spacing...
        let (_, new_tail) = update_str.split_at(prev_full_decode.len());
        if let Err(err) = sender.send(TextGenUpdate::Token(new_tail.to_string())) {
            error!(
                "Error while sending text generation update message along channel: {}",
                err
            );
        }
        Some(update_str)
    } else {
        None
    }
}
