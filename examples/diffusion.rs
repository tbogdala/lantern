use std::{path::PathBuf, time::SystemTime};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use clap::Parser;
use lantern::DiffusionConfig;
use log::{debug, info};

// This is a copy of what's in the diffusion module, but here we add
// the clap::ValueEnum implementation which is a dependency only for
// the examples and not the main library...
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SDVersionArg {
    V1_5,
    Turbo,
}


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "turbo")]
    sd_ver: SDVersionArg,

    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    #[arg(short, long)]
    seed: Option<u32>,

    #[arg(long)]
    cfg: Option<f64>,

    #[arg(long)]
    steps: Option<usize>,

    #[arg(long, default_value_t = 512)]
    height: usize,

    #[arg(long, default_value_t = 512)]
    width: usize,

    #[arg(long, default_value = "output.png")]
    save_as: String,

    #[arg(long, value_name = "Base-Image-File")]
    img2img: Option<String>,

    #[arg(long, default_value_t = 0.7)]
    img_strength: f32,
}

fn main() {
    let log_env = env_logger::Env::default().filter_or("RUST_LOG", "info");
    env_logger::init_from_env(log_env);
    let args = Args::parse();

    // we seed the sampler based on the current time
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32;
    let seed = args.seed.unwrap_or(seed);
    info!("Seed: {}", seed);

    // get the configured device type
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();
    #[cfg(not(feature = "cuda"))]
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    debug!(
        "Device is: cpu={} | cuda={} | metal={}",
        device.is_cpu(),
        device.is_cuda(),
        device.is_metal()
    );
    device.set_seed(seed as u64).unwrap();

    // default cfg to 0.0 for Turbo, 7.5 for everything else, or use specfied parameter
    let guidance_scale = if let Some(cfg) = args.cfg {
        cfg
    } else {
        if args.sd_ver == SDVersionArg::Turbo {
            0.0
        } else {
            7.5
        }
    };

    // default steps value to 1 for Tubo and 30 for everything else, or use specified parameter
    let steps = if let Some(s) = args.steps {
        s
    } else {
        if args.sd_ver == SDVersionArg::Turbo {
            1
        } else {
            30
        }
    };

    // pull the prompts in from the command line or suprise the user with a default one.
    let prompt = args
        .prompt
        .unwrap_or("A giant red crustacean on a beach. Highly stylized.".to_owned());
    let uncond_prompt = args.uncond_prompt;
    
    // we do a little fancy work with the filename to put iteration numbers at the
    // end of the filename passed in as a parameter if that one already exists.
    let output_filename = get_available_filename(args.save_as.as_str()).unwrap();
    info!("Output filename: {:?}", output_filename);

    // clamp the img2img strength values to the valid range; 1.0 will essentially create
    // a whole new image because it'll all be noised out, while 0.0 will essentially return
    // the original image.
    let img_strength = args.img_strength.clamp(0.0, 1.0);
    let img2img = args.img2img;
    if let Some(source_file) = &img2img {
        info!(
            "Image to image: Strength: {}; Source: {}",
            img_strength, source_file
        );
    }

    let config = match args.sd_ver {
        SDVersionArg::V1_5 => DiffusionConfig::new_sd15(args.height, args.width, steps, device.clone()),
        SDVersionArg::Turbo => DiffusionConfig::new_sdxl_turbo(args.height, args.width, steps, device.clone())
    };

    let scheduler = config.build_scheduler().unwrap();

    info!("Building the clip transformers and tokenizing the prompts.");
    let prompt_embeddings = config
        .build_text_encoders(prompt.as_str(), uncond_prompt.as_str(), guidance_scale)
        .unwrap();

    info!("Building the autoencoder.");
    let vae = config.build_vae().unwrap();

    info!("Building the unet.");
    let unet = config.build_model().unwrap();

    info!("starting sampling");
    let timesteps = scheduler.timesteps();
    let t_start = if img2img.is_some() {
        steps - (steps as f32 * img_strength) as usize
    } else {
        0
    };

    let mut latents = if let Some(source_fp) = img2img {
        config
            .build_img2img_latents(&source_fp, &vae, &scheduler, timesteps[t_start])
            .unwrap()
    } else {
        config.build_init_latents(&scheduler).unwrap()
    };

    for (timestep_index, &timestep) in timesteps.iter().enumerate() {
        if timestep_index < t_start {
            continue;
        }

        let start_time = std::time::Instant::now();

        let latent_model_input = if guidance_scale > 0.0 {
            Tensor::cat(&[&latents, &latents], 0).unwrap()
        } else {
            latents.clone()
        };

        let latent_model_input = scheduler
            .scale_model_input(latent_model_input, timestep)
            .unwrap();
        
        let noise_pred = unet
            .forward(&latent_model_input, timestep as f64, &prompt_embeddings)
            .unwrap();
        let noise_pred = if guidance_scale > 0.0 {
            let noise_pred = noise_pred.chunk(2, 0).unwrap();
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond).unwrap() * guidance_scale)).unwrap()
        } else {
            noise_pred
        };

        latents = scheduler.step(&noise_pred, timestep, &latents).unwrap();
        let dt = start_time.elapsed().as_secs_f32();
        info!(
            "Iteration #{} (Timestep: {}) completed in {:.2}s...",
            timestep_index + 1,
            timestep,
            dt
        );
    }

    save_image(&vae, &latents, config.vae_scale, &output_filename).unwrap();
}

fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    final_image: &PathBuf,
) -> Result<()> {
    info!("Decoding latents with the vae...");
    let start_time = std::time::Instant::now();
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    let img = images.i(0)?;

    info!(
        "Imaged finished decoding in {:.2}s; saving to: {:?}",
        start_time.elapsed().as_secs_f32(),
        final_image
    );
    let p = std::path::Path::new(final_image);
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        return Err(anyhow!(
            "save_image expects an input of shape (3, height, width)"
        ));
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => return Err(anyhow!("error saving image {p:?}")),
        };
    image.save(p).map_err(anyhow::Error::msg)?;
    Ok(())
}

fn get_available_filename(filename: &str) -> Result<PathBuf> {
    // if it doesn't exist, we just return it
    if !std::fs::metadata(&filename).is_ok() {
        return Ok(PathBuf::from(filename));
    }

    // but since we have a collision on the filename, just keep adding
    // iteration numbers until we get one that doesn't exist and return
    // that instead.
    let mut buffer = PathBuf::from(filename);
    let base_filename_stem = buffer.file_stem().unwrap().to_str().unwrap().to_owned();
    buffer.pop();

    let mut index = 1;
    loop {
        let new_filename = format!("{}_{:04}.png", base_filename_stem, index);
        buffer.push(new_filename);

        if !std::fs::metadata(buffer.to_str().unwrap()).is_ok() {
            return Ok(buffer);
        }

        // still running into existing files so try again...
        index += 1;
        buffer.pop();
    }
}
