use std::{
    env,
    io::{stdout, Write},
    time::SystemTime,
};

use clap::Parser;
use lantern::{TextGenUpdate, TextGenerationParams, TextGeneratorManager};
use log::info;

// setup the default model, quant size, tokenizer and EOS string
const DEFAULT_MODEL_ID: &str = "bartowski/Meta-Llama-3-8B-Instruct-old-GGUF";
const DEFAULT_MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct-Q8_0.gguf";
const DEFAULT_TOKENIZER_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(short, long)]
    batch_size: Option<usize>,

    #[arg(short, long)]
    seed: Option<u64>,
}

fn main() {
    // It's possible to initialize the logging functions and set the minimum filter to 'info' so we see more.
    //let log_env = env_logger::Env::default().filter_or("RUST_LOG", "info");
    //env_logger::init_from_env(log_env);
    env_logger::init();

    // pull in the CLI parameters passed in.
    let args = Args::parse();

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

    // we seed the sampler based on the current time
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let seed = args.seed.unwrap_or(seed);
    info!("Seed: {}", seed);

    // setup a TextGeneratorManager object which acts as a controller for the text generation operations
    let mut manager = TextGeneratorManager::new();

    // setup the parameters for text generation; the example uses a prompt format for the default model
    let prompt = if let Some(user_prompt) = args.prompt {
        user_prompt
    } else {
        let inner_system = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.";
        let inner_prompt = "Explain large language models and the LLaMa 3 model architecture like I'm five years old and can only understand simple concepts.";
        println!(
            "\nDefault Prompt:\n\n{}\n\nGenerated Answer:\n",
            inner_prompt
        );
        format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", inner_system, inner_prompt)
    };

    // if you've used LLMs before, these parameters should all seem familiar. we're going to generate
    // 512 new tokens at most, using the seed provided or generated, based on the prompt we just assembled above.
    // we'll break the prompt up into chunks of 256 tokens if necessary and then we'll set a few
    // sampler parameters to control how the text is picked. Specifically, we'll turn the temp down to 0.7 for
    // less randomness in the answer, disable top_p by setting it to 1.0, use min_p to cull selection to tokens
    // that are within 7.5% of the probability of the most probable token, and then give a slight penalty
    // to repeating tokens.
    let params = TextGenerationParams {
        to_sample: 512,
        seed: seed,
        user_prompt: prompt,
        batch_size: args.batch_size.unwrap_or(256),
        temperature: 0.7,
        top_p: 1.0,
        min_p: 0.075,
        repeat_penalty: 1.02,
        ..Default::default()
    };

    // a little data tracking for figuring out the speed at the end
    let mut number_sampled = 0;
    let number_to_predict = params.to_sample;

    // start the text generation request. this function is non-blocking as it will execute the operation on a separate thread
    info!("Text generation sending generate_text request.");
    let start_time = std::time::Instant::now();
    manager
        .generate_text(
            model_id,
            model_file,
            tokenizer_repo_id,
            eos_token_str,
            params,
        )
        .unwrap();
    info!("Text generation test request sent.");

    // since the `generate_text()` method is non-blocking, we now loop through the tokens it generates
    // until we get the `TextGenUpdate::Finished` value, indicating the job is finished.
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

    // calculate the total time needed to generate the response; this does not take into account
    // the different speed rates between prompt processing and text generation which may affect the number.
    let elapsed_time = start_time.elapsed();
    println!(
        "\n\nGeneration of {} of {} tokens took {} seconds ({:.2} t/s)",
        number_sampled,
        number_to_predict,
        elapsed_time.as_secs(),
        number_sampled as f32 / elapsed_time.as_secs_f32()
    );
}
