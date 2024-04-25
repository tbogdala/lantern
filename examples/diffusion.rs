use std::{path::PathBuf, time::SystemTime};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::Module;
use candle_transformers::models::stable_diffusion::{
    self, clip, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL, StableDiffusionConfig,
};
use clap::Parser;
use hf_hub::api::sync::Api;
use log::{debug, info};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    V1_5,
    Turbo,
}
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_enum, default_value = "Turbo")]
    sd_ver: StableDiffusionVersion,

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

/// type alias for a hugginface repository id and filename in that repository.
/// used for getting file paths via the hf_hub library.
pub type RepoIdAndFile = (String, String);

#[derive(Debug)]
pub struct DiffusionConfig {
    pub height: usize,
    pub width: usize,
    inner_config: StableDiffusionConfig,
    sd_ver: StableDiffusionVersion,
    pub steps: usize,
    pub tokenizer1: RepoIdAndFile,
    pub tokenizer2: RepoIdAndFile,
    pub clip1: RepoIdAndFile,
    pub clip2: RepoIdAndFile,
    pub vae: RepoIdAndFile,
    pub model: RepoIdAndFile,
    device: Device,
    pub dtype: DType,
    pub vae_scale: f64,
}
impl DiffusionConfig {
    /// creates a new configuration set for SDXL Turbo use. This model is tuned for 512x512.
    pub fn new_sd15(height: usize, width: usize, steps: usize, device: Device) -> Self {
        Self {
            height,
            width,
            inner_config: StableDiffusionConfig::v1_5(None, Some(height), Some(width)),
            sd_ver: StableDiffusionVersion::V1_5,
            steps,
            tokenizer1: (
                "openai/clip-vit-base-patch32".to_owned(),
                "tokenizer.json".to_owned(),
            ),
            tokenizer2: (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_owned(),
                "tokenizer.json".to_owned(),
            ),
            clip1: (
                "runwayml/stable-diffusion-v1-5".to_owned(),
                "text_encoder/model.fp16.safetensors".to_owned(),
            ),
            clip2: (
                "runwayml/stable-diffusion-v1-5".to_owned(),
                "text_encoder_2/model.fp16.safetensors".to_owned(),
            ),
            vae: (
                "runwayml/stable-diffusion-v1-5".to_owned(),
                "vae/diffusion_pytorch_model.fp16.safetensors".to_owned(),
            ),
            model: (
                "runwayml/stable-diffusion-v1-5".to_owned(),
                "unet/diffusion_pytorch_model.fp16.safetensors".to_owned(),
            ),
            device,
            dtype: DType::F16,
            vae_scale: 0.18215,
        }
    }
    
    /// creates a new configuration set for SDXL Turbo use. This model is tuned for 512x512.
    pub fn new_sdxl_turbo(height: usize, width: usize, steps: usize, device: Device) -> Self {
        Self {
            height,
            width,
            inner_config: StableDiffusionConfig::sdxl_turbo(None, Some(height), Some(width)),
            sd_ver: StableDiffusionVersion::Turbo,
            steps,
            tokenizer1: (
                "openai/clip-vit-large-patch14".to_owned(),
                "tokenizer.json".to_owned(),
            ),
            tokenizer2: (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_owned(),
                "tokenizer.json".to_owned(),
            ),
            clip1: (
                "stabilityai/sdxl-turbo".to_owned(),
                "text_encoder/model.fp16.safetensors".to_owned(),
            ),
            clip2: (
                "stabilityai/sdxl-turbo".to_owned(),
                "text_encoder_2/model.fp16.safetensors".to_owned(),
            ),
            vae: (
                "madebyollin/sdxl-vae-fp16-fix".to_owned(),
                "diffusion_pytorch_model.safetensors".to_owned(),
            ),
            model: (
                "stabilityai/sdxl-turbo".to_owned(),
                "unet/diffusion_pytorch_model.fp16.safetensors".to_owned(),
            ),
            device,
            dtype: DType::F16,
            vae_scale: 0.13025,
        }
    }

    pub fn build_scheduler(&self) -> Result<Box<dyn stable_diffusion::schedulers::Scheduler>> {
        Ok(self.inner_config.build_scheduler(self.steps)?)
    }

    pub fn build_text_encoders(&self, prompt: &str, uncond_prompt: &str, cfg: f64) -> Result<Tensor> {
        let embeddings1 = self.text_embeddings(
            prompt,
            uncond_prompt,
            cfg,
            &self.tokenizer1,
            &self.clip1,
            Some(&self.inner_config.clip),
        )?;

        if self.sd_ver == StableDiffusionVersion::V1_5 {
            Ok(embeddings1)
         } else {
            let embeddings2 = self.text_embeddings(
                prompt,
                uncond_prompt,
                cfg,
                &self.tokenizer2,
                &self.clip2,
                self.inner_config.clip2.as_ref(),
            )?;
            Ok(Tensor::cat(&[embeddings1, embeddings2], D::Minus1)?)
        } 
    }

    pub fn build_vae(&self) -> Result<AutoEncoderKL> {
        let vae_weights_filename = Api::new()?.model(self.vae.0.to_string()).get(&self.vae.1)?;
        Ok(self
            .inner_config
            .build_vae(vae_weights_filename, &self.device, self.dtype)?)
    }

    pub fn build_model(&self) -> Result<UNet2DConditionModel> {
        // TODO: not supported yet...
        const USE_FLASH_ATTENTION: bool = false;
        let unet_weights_filename = Api::new()?
            .model(self.model.0.to_string())
            .get(&self.model.1)?;
        Ok(self.inner_config.build_unet(
            unet_weights_filename,
            &self.device,
            4,
            USE_FLASH_ATTENTION,
            self.dtype,
        )?)
    }

    pub fn build_init_latents(
        &self,
        scheduler: &Box<dyn stable_diffusion::schedulers::Scheduler>,
    ) -> Result<Tensor> {
        const BATCH_SIZE: usize = 1;

        let latents = Tensor::randn(
            0f32,
            1f32,
            (BATCH_SIZE, 4, self.height / 8, self.width / 8),
            &self.device,
        )?;
        let latents = (latents * scheduler.init_noise_sigma())?;
        Ok(latents.to_dtype(self.dtype)?)
    }

    pub fn build_img2img_latents(
        &self,
        source_filepath: &str,
        vae: &AutoEncoderKL,
        scheduler: &Box<dyn stable_diffusion::schedulers::Scheduler>,
        noise_timestep: usize,
    ) -> Result<Tensor> {
        // turn the source image into a latent tensor
        let img = image::io::Reader::open(source_filepath)?.decode()?;

        let h = img.height();
        let w = img.width();

        let h = h - h % 32;
        let w = w - w % 32;

        let img = img.resize_to_fill(w, h, image::imageops::FilterType::CatmullRom);
        let img = img.to_rgb8();
        let img = img.into_raw();

        let source_image = Tensor::from_vec(img, (h as usize, w as usize, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2.0 / 255.0, -1.0)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;

        let latent_dist = vae.encode(&source_image)?;
        let latents = (latent_dist.sample()? * self.vae_scale)?.to_device(&self.device)?;

        let noise = latents.randn_like(0f64, 1f64)?;
        let noisey_latents = scheduler.add_noise(&latents, noise, noise_timestep)?;
        Ok(noisey_latents.to_dtype(self.dtype)?)
    }

    fn text_embeddings(
        &self,
        prompt: &str,
        uncond_prompt: &str,
        cfg: f64,
        tokenizer_id_and_file: &RepoIdAndFile,
        clip_id_and_file: &RepoIdAndFile,
        clip_config: Option<&clip::Config>,
    ) -> Result<Tensor> {
        let tokenizer_filename = Api::new()?
            .model(tokenizer_id_and_file.0.to_string())
            .get(&tokenizer_id_and_file.1)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        let pad_id = match &self.inner_config.clip.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };

        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        if tokens.len() > self.inner_config.clip.max_position_embeddings {
            return Err(anyhow!(
                "the prompt is too long, {} > max-tokens ({})",
                tokens.len(),
                self.inner_config.clip.max_position_embeddings
            ));
        }
        while tokens.len() < self.inner_config.clip.max_position_embeddings {
            tokens.push(pad_id)
        }
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let clip_weights_filename = Api::new()?
            .model(clip_id_and_file.0.to_string())
            .get(&clip_id_and_file.1)?;
        let text_model = stable_diffusion::build_clip_transformer(
            clip_config.as_ref().unwrap(),
            clip_weights_filename,
            &self.device,
            DType::F32,
        )?;
        let text_embeddings = text_model.forward(&tokens)?;
        if cfg > 0.0 {
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            if uncond_tokens.len() > self.inner_config.clip.max_position_embeddings {
                return Err(anyhow!(
                    "the negative prompt is too long, {} > max-tokens ({})",
                    uncond_tokens.len(),
                    self.inner_config.clip.max_position_embeddings
                ));
            }
            while uncond_tokens.len() < self.inner_config.clip.max_position_embeddings {
                uncond_tokens.push(pad_id);
            }

            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let uncond_embeddings = text_model.forward(&uncond_tokens)?;

            Ok(Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.dtype)?)
        } else {
            Ok(text_embeddings.to_dtype(self.dtype)?)
        } 
    }
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
        if args.sd_ver == StableDiffusionVersion::Turbo {
            0.0
        } else {
            7.5
        }
    };

    // default steps value to 1 for Tubo and 30 for everything else, or use specified parameter
    let steps = if let Some(s) = args.steps {
        s
    } else {
        if args.sd_ver == StableDiffusionVersion::Turbo {
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
        StableDiffusionVersion::V1_5 => DiffusionConfig::new_sd15(args.height, args.width, steps, device.clone()),
        StableDiffusionVersion::Turbo => DiffusionConfig::new_sdxl_turbo(args.height, args.width, steps, device.clone())
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
