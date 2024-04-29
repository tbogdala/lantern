use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::Module;
use candle_transformers::models::stable_diffusion::{
    self, clip, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL, StableDiffusionConfig,
};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    V1_5,
    Turbo,
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

    pub fn build_text_encoders(
        &self,
        prompt: &str,
        uncond_prompt: &str,
        cfg: f64,
    ) -> Result<Tensor> {
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

            let uncond_tokens =
                Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let uncond_embeddings = text_model.forward(&uncond_tokens)?;

            Ok(Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.dtype)?)
        } else {
            Ok(text_embeddings.to_dtype(self.dtype)?)
        }
    }

    pub fn decode_latents(&self, latents: &Tensor, vae: &AutoEncoderKL) -> Result<Vec<u8>> {
        let images = vae.decode(&(latents / self.vae_scale)?)?;
        let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
        let img = images.i(0)?;

        let (channel, _height, _width) = img.dims3()?;
        if channel != 3 {
            return Err(anyhow!(
                "save_image expects an input of shape (3, height, width)"
            ));
        }

        let img = img.permute((1, 2, 0))?.flatten_all()?;
        let pixels = img.to_vec1::<u8>()?;
        Ok(pixels)
    }
}
