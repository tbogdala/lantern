use clap::Parser;
use lantern::{TextGenUpdate, TextGenerationParams, TextGeneratorManager};
use log::{debug, error, info};
use reqwest;
use scraper::{Html, Selector};
use std::{
    env,
    fs::File,
    io::{stdout, Write},
    time::SystemTime,
};

// setup the default model, quant size, tokenizer and EOS string
const DEFAULT_MODEL_ID: &str = "bartowski/Meta-Llama-3-8B-Instruct-GGUF";
const DEFAULT_MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct-Q8_0.gguf";
const DEFAULT_TOKENIZER_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

const CACHE_FILENAME: &str = "examples/.rag_cached.html";
const SAMPLE_RAG_URL: &str = "https://explained.ai/matrix-calculus/";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    batch_size: Option<usize>,

    #[arg(short, long)]
    seed: Option<u64>,
}

fn main() {
    // initialize the logging functions and set the minimum filter to 'info' so we see more.
    let log_env = env_logger::Env::default().filter_or("RUST_LOG", "info");
    env_logger::init_from_env(log_env);

    // pull in the CLI parameters passed in.
    let args = Args::parse();

    // Check to see if we have a cached file for the web page we're using
    let raw_html: String;
    let rag_text = if std::fs::metadata(CACHE_FILENAME).is_ok() {
        info!("Using cache file for HTML: {}", CACHE_FILENAME);
        let cached_bytes =
            std::fs::read(CACHE_FILENAME).expect("Failed to read bytes from the cached html file.");
        String::from_utf8(cached_bytes)
            .expect("Failed to convert cached html bytes to a utf-8 string.")
    } else {
        // pull a webpage down to get the raw data
        let resp = reqwest::blocking::get(SAMPLE_RAG_URL);
        if let Err(err) = resp {
            error!("Failed to get the webpage \"{}\": {}", SAMPLE_RAG_URL, err);
            return;
        }
        let resp_body = resp.unwrap().text();
        if let Err(err) = resp_body {
            error!(
                "Failed to get the webpage body for \"{}\": {}",
                SAMPLE_RAG_URL, err
            );
            return;
        }
        raw_html = resp_body.unwrap();

        // write the cached html
        let mut cache_file =
            File::create(CACHE_FILENAME).expect("Failed to create cached html file.");
        cache_file
            .write_all(raw_html.as_bytes())
            .expect("Failed to write cached html file.");

        // get it into scraper to parse out the text we want to summarize
        // and yes, this is a rough hackjob.
        // FIXME: This workflow is still incomplete here and the cache file had
        // to be manually edited to be sized down to appropriate length.
        println!("Chances are real good that you'll have to edit the cached '.rag_cached.html' file and make it smaller to process for now...");
        let html_document = Html::parse_document(&raw_html);
        let detective = Selector::parse("p").unwrap();
        let victims = html_document.select(&detective);
        let mut all_text = String::new();
        for victim in victims {
            all_text.push_str(victim.text().collect::<String>().as_str());
        }
        all_text
    };

    // pull environment variables or get default model settings to use for tests
    let model_id = env::var("LANTERN_MODEL_ID").unwrap_or(DEFAULT_MODEL_ID.to_string());
    let model_file = env::var("LANTERN_MODEL_FILE").unwrap_or(DEFAULT_MODEL_FILE.to_string());
    let tokenizer_repo_id =
        env::var("LANTERN_TOKENIZER_ID").unwrap_or(DEFAULT_TOKENIZER_ID.to_string());
    let eos_token_str = env::var("LANTERN_EOS_STRING").unwrap_or(DEFAULT_EOS_TOKEN.to_string());

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
    let inner_system = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.";
    let rag_prompt = format!("Summarize the following raw web page text in a few paragraphs with easily digestible language:\n\n{}", rag_text);
    let prompt = format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", inner_system, rag_prompt);
    debug!("Sending out the following prompt:\n{}", prompt);

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
            TextGenUpdate::Finished(whole_str) => {
                println!("");
                let _ = stdout().flush();
                info!("Text prediction finished!");
                info!("Result: {}", whole_str);
                break;
            }
        }
    }

    // calculate the total time needed to generate the response; this does not take into account
    // the different speed rates between prompt processing and text generation which may affect the number.
    let elapsed_time = start_time.elapsed();
    info!(
        "Generation of {} of {} tokens took {} seconds ({:.2} t/s)",
        number_sampled,
        number_to_predict,
        elapsed_time.as_secs(),
        number_sampled as f32 / elapsed_time.as_secs_f32()
    );
}
