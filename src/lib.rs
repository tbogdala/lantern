//! Lantern is a higher level library built around [Candle](https://github.com/huggingface/candle).
//! Its goal is to provide more ready-made solutions to common tasks for AI generation, with initial
//! support focusing on running quantized LLMs to generate text.
//!
//! ```
//! let mut manager = lantern::TextGeneratorManager::new();
//! let params = lantern::TextGenerationParams {
//!     to_sample: 256,
//!     user_prompt: "The best thing about coding in rust is ".to_string(),
//!     ..Default::default()
//! };
//!
//! // The manager will automatically download the file from huggingface to do inference on
//! manager.generate_text(
//!     "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF".to_string(), // model id on huggingface
//!     "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf".to_string(), // model file in the repo
//!     "mistralai/Mistral-7B-v0.1".to_string(), // repo id on huggingface that has `tokenizer.json` for the model
//!     "</s>".to_string(), // EOS token to stop at
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

mod quantized_llama;
mod textgeneratormanager;
mod textgeneratorsampler;

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
    const DEFAULT_MODEL_ID: &str = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF";
    const DEFAULT_MODEL_FILE: &str = "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf";
    const DEFAULT_TOKENIZER_ID: &str = "mistralai/Mistral-7B-v0.1";
    const DEFAULT_EOS_TOKEN: &str = "</s>";

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

        let params = TextGenerationParams {
            to_sample: 256,
            user_prompt: "<s><|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nExplain large language models and the llama 2 model architecture like I'm five years old and can only understand simple concepts.<|im_end|>\n<|im_start|>assistant\n".to_string(),
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
