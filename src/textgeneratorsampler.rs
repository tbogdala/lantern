use anyhow::{Error, Result};
use candle_core::{DType, Tensor};
use rand::{distributions::Distribution, SeedableRng};

use crate::TextGenerationParams;

/// This is the sampler that is used during text generation. When created, it uses the
/// RNG seed provided and then `sample` can be invoked to pick the next logit based on
/// all of the parameters provided.
pub struct TextGeneratorSampler {
    rng: rand::rngs::StdRng,
}

impl TextGeneratorSampler {
    /// Create a new sampler with the seed provided. Client code must provide a seed as the library
    /// does not generate a 'random' value under any circumstances and will just use what's provided.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// a greedy sampler that just gets the most likely token id.
    pub fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    /// based on the provided probabilities, samples the next token id
    pub fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::msg)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of
    /// tokens that exceed probability top_p. This way we never sample tokens that
    /// have very low probabilities and are less likely to go "off the rails".
    /// Disabled with top_p >= 1.0.
    pub fn sample_top_p(&mut self, prs: &mut Vec<f32>, argsort_indices: &Vec<usize>, top_p: f32) {
        if top_p >= 1.0 {
            return;
        }

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
    }

    /// top-k sampling clamps all but the first top_k number of tokens to zero.
    /// Disabled with top_k == 0.
    pub fn sample_top_k(&mut self, prs: &mut Vec<f32>, argsort_indices: &Vec<usize>, top_k: usize) {
        if top_k == 0 {
            return;
        }
        // clamp everything but the first top_k number of tokens to a probability of zero.
        let mut total_visited = 0;
        for index in argsort_indices {
            if total_visited >= top_k {
                prs[*index] = 0.0;
            } else {
                total_visited += 1;
            }
        }
    }

    /// min-p sampling clamps everything that is not within a minimum percentage of the most likely
    /// token's probability to zero.  So if the most likely probability is 0.5 and min_p is 0.1, every
    /// probability is less than 0.05 gets clamped to zero.
    /// Disabled with min <= 0.0.
    pub fn sample_min_p(&mut self, prs: &mut Vec<f32>, argsort_indices: &Vec<usize>, min_p: f32) {
        if min_p <= 0.0 {
            return;
        }

        let min_probability = prs[argsort_indices[0]] * min_p;
        for index in argsort_indices {
            if prs[*index] < min_probability {
                prs[*index] = 0.0;
            }
        }
    }

    /// samples the next token id based on the params passed in. If temperature is <= 0.0,
    /// only greedy sampling is performed. Otherwise the repetition penalty was already performed
    /// in the text generator so samplers are called in the following order: top_k, top_p, min_p, temperature
    pub fn sample(&mut self, params: &TextGenerationParams, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;

        // if temperature is <= 0.0, then just do greedy sampling only
        let next_token = if params.temperature <= 0.0 {
            self.sample_argmax(logits)?
        } else {
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs: Vec<f32> = prs.to_vec1()?;

            // Sort by descending probability.
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());

            self.sample_top_k(&mut prs, &argsort_indices, params.top_k);

            self.sample_top_p(&mut prs, &argsort_indices, params.top_p);

            self.sample_min_p(&mut prs, &argsort_indices, params.min_p);

            // apply temperature
            let prs = prs.iter().map(|f| f / params.temperature).collect();

            // Sample with clamped probabilities.
            self.sample_multinomial(&prs)?
        };

        Ok(next_token)
    }
}
