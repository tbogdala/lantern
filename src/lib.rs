//! Lantern is a higher level library built around [Candle](https://github.com/huggingface/candle).
//! Its goal is to provide more ready-made solutions to common tasks for AI generation, with initial
//! support focusing on running quantized LLMs to generate text.
//!
//! ```
//! let mut manager = lantern::TextGeneratorManager::new();
//! let params = lantern::TextGenerationParams {
//!     to_sample: 64,
//!     user_prompt: "The best thing about coding in rust is ".to_string(),
//!     ..Default::default()
//! };
//!
//! // The manager will automatically download the file from huggingface to do inference on
//! manager.generate_text(
//!     "bartowski/Meta-Llama-3-8B-Instruct-GGUF".to_string(), // model id on huggingface
//!     "Meta-Llama-3-8B-Instruct-Q8_0.gguf".to_string(), // model file in the repo
//!     "NousResearch/Meta-Llama-3-8B-Instruct".to_string(), // repo id on huggingface that has `tokenizer.json` for the model
//!     "<|eot_id|>".to_string(), // EOS token to stop at
//!     params,
//! );
//! loop {
//!     let update = manager.get_update();
//!     match update {
//!         lantern::TextGenUpdate::Finished(str) => {
//!             println!("Predicted text:\n{}", str);
//!             break;
//!         },
//!         lantern::TextGenUpdate::Token(_) => {},
//!     }
//! }
//!```

mod diffusion;
mod quantized_llama;
mod textgeneratormanager;
mod textgeneratorsampler;

pub use diffusion::StableDiffusionVersion;
pub use diffusion::DiffusionConfig;

pub use textgeneratormanager::TextGenUpdate;
pub use textgeneratormanager::TextGenerationParams;
pub use textgeneratormanager::TextGeneratorManager;
pub use textgeneratorsampler::TextGeneratorSampler;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textgeneratormanager::TextGenUpdate;
    use log::info;
    use std::{
        env,
        io::{stdout, Write},
    };
    const DEFAULT_MODEL_ID: &str = "bartowski/Meta-Llama-3-8B-Instruct-GGUF";
    const DEFAULT_MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct-Q8_0.gguf";
    const DEFAULT_TOKENIZER_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
    const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

    #[test]
    fn text_generation_tests() {
        env_logger::init();

        // pull environment variables or get default model settings to use for tests
        let model_id = env::var("LANTERN_MODEL_ID").unwrap_or(DEFAULT_MODEL_ID.to_string());
        let model_file = env::var("LANTERN_MODEL_FILE").unwrap_or(DEFAULT_MODEL_FILE.to_string());
        let tokenizer_repo_id =
            env::var("LANTERN_TOKENIZER_ID").unwrap_or(DEFAULT_TOKENIZER_ID.to_string());
        let eos_token_str = env::var("LANTERN_EOS_STRING").unwrap_or(DEFAULT_EOS_TOKEN.to_string());

        info!(
            "Text generation test starting for {}/{}.",
            model_id, model_file
        );

        let mut manager = TextGeneratorManager::new();
        assert_eq!(manager.is_busy(), false);
        assert_eq!(manager.get_progress(), (0, 0));
        assert_eq!(manager.maybe_get_update(), None);

        let inner_system = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.";
        let inner_prompt = "Explain why pigs are unable to fly to a grade school child.";
        let prompt = format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
           inner_system, inner_prompt);

        let params = TextGenerationParams {
            to_sample: 128,
            user_prompt: prompt,
            ..Default::default()
        };
        let mut number_sampled = 0;
        let number_to_predict = params.to_sample;

        info!("Text generation sending generate_text request.");
        let start_time = std::time::Instant::now();
        let result = manager.generate_text(
            model_id,
            model_file,
            tokenizer_repo_id,
            eos_token_str,
            params,
        );
        assert_eq!(result.is_ok(), true);
        assert_eq!(manager.is_busy(), true);

        info!("Text generation test request sent.");
        loop {
            let update = manager.get_update();
            match update {
                TextGenUpdate::Token(t) => {
                    number_sampled += 1;
                    print!("{}", t);
                    let _ = stdout().flush();
                    assert_eq!(manager.get_progress(), (number_sampled, number_to_predict));
                }
                TextGenUpdate::Finished(str) => {
                    info!("Predicted text:\n{}", str);
                    break;
                }
            }
        }
        let elapsed_time = start_time.elapsed();
        info!(
            "Generation of {} of {} tokens took {} seconds ({:.2} t/s)",
            number_sampled,
            number_to_predict,
            elapsed_time.as_secs(),
            number_sampled as f32 / elapsed_time.as_secs_f32()
        );
        assert_eq!(manager.maybe_get_update(), None);
    }
}
