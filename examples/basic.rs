use std::{env, io::{stdout, Write}, time::SystemTime};

use lantern::{TextGenUpdate, TextGenerationParams, TextGeneratorManager};
use log::info;

// setup the default model, quant size, tokenizer and EOS string
const DEFAULT_MODEL_ID: &str = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF";
const DEFAULT_MODEL_FILE: &str = "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf";
const DEFAULT_TOKENIZER_ID: &str = "mistralai/Mistral-7B-v0.1";
const DEFAULT_EOS_TOKEN: &str = "</s>";

fn main() {
    // initialize the logging functions and set the minimum filter to 'info' so we see more.
    let log_env = env_logger::Env::default().filter_or("MY_LOG_LEVEL", "debug");
    env_logger::init_from_env(log_env);

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
    

    let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as u64;
    info!("Seed: {}", seed);

    let mut manager = TextGeneratorManager::new();
    let params = TextGenerationParams {
        to_sample: 256,
        seed: seed,
        stop_strings: vec!["<|im_end|>".to_string()],
        user_prompt: "<s><|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nExplain large language models and the llama 2 model architecture like I'm five years old and can only understand simple concepts.<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ..Default::default()
    };
    let mut number_sampled = 0;
    let number_to_predict = params.to_sample;

    info!("Text generation sending generate_text request.");
    let start_time = std::time::Instant::now();
    manager.generate_text(
        model_id,
        model_file,
        tokenizer_repo_id,
        eos_token_str,
        params,
    ).unwrap();

    info!("Text generation test request sent.");
    loop {
        let update = manager.get_update();
        match update {
            TextGenUpdate::Token(t) => {
                number_sampled += 1;
                print!("{}", t);
                let _ = stdout().flush();
            }
            TextGenUpdate::Finished(_whole_str) => {
                println!("");
                let _ = stdout().flush();
                info!("Text prediction finished");
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
}