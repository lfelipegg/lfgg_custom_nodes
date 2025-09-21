"""Configurable KSampler nodes with integrated checkpoint loading."""

from __future__ import annotations

from typing import Optional, Tuple

import comfy.samplers
import comfy.sd
import folder_paths

from comfy.comfy_types import IO

from nodes import common_ksampler


def _load_checkpoint(ckpt_name: str) -> Tuple[object, object, object]:
    """Load and return the model, clip, and vae for the given checkpoint name."""
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:3]
    return model, clip, vae


def _resolve_sampler(sampler_name: str):
    """Return a k-diffusion sampler callable for downstream nodes."""
    return comfy.samplers.ksampler(sampler_name)


def _prepare_assets(
    ckpt_name: Optional[str],
    model_input,
    clip_input,
    vae_input,
) -> Tuple[object, Optional[object], Optional[object], str]:
    """Return the model/clip/vae objects and the checkpoint label to expose."""
    if model_input is not None:
        checkpoint_label = ckpt_name or getattr(model_input, "ckpt_name", None)
        if checkpoint_label is None:
            checkpoint_label = getattr(model_input, "ckpt_path", None)
        checkpoint_label = checkpoint_label or "external_model"
        return model_input, clip_input, vae_input, str(checkpoint_label)

    if not ckpt_name:
        raise ValueError("ckpt_name must be provided when no model input is connected")

    model, clip, vae = _load_checkpoint(ckpt_name)
    return model, clip_input or clip, vae_input or vae, ckpt_name


class LFGGKSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint to load before sampling (ignored if a model input is connected)."},
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "Random seed used to create the initial noise field.",
                    },
                ),
                "steps": (
                    IO.INT,
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of denoising steps to perform.",
                    },
                ),
                "cfg": (
                    IO.FLOAT,
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Classifier-free guidance scale controlling adherence to prompts.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler algorithm to drive the diffusion process."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Noise schedule shaping how each step is taken."},
                ),
                "positive": (
                    IO.CONDITIONING,
                    {"tooltip": "Conditioning describing desired attributes."},
                ),
                "negative": (
                    IO.CONDITIONING,
                    {"tooltip": "Conditioning describing attributes to avoid."},
                ),
                "latent_image": (
                    IO.LATENT,
                    {"tooltip": "Latent tensor to denoise."},
                ),
                "denoise": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoising strength (lower preserves more of the input).",
                    },
                ),
            },
            "optional": {
                "model": (
                    IO.MODEL,
                    {"tooltip": "Optional preloaded model to use instead of loading from a checkpoint."},
                ),
                "clip": (
                    IO.CLIP,
                    {"tooltip": "Optional CLIP to forward when supplying an external model."},
                ),
                "vae": (
                    IO.VAE,
                    {"tooltip": "Optional VAE to forward when supplying an external model."},
                ),
            },
        }

    RETURN_TYPES = (
        IO.FLOAT,  # cfg
        IO.STRING,  # checkpoint_name
        IO.CLIP,
        IO.FLOAT,  # denoise
        IO.LATENT,
        IO.MODEL,
        IO.SAMPLER,  # sampler_name (callable)
        IO.STRING,  # sampler_string
        comfy.samplers.KSampler.SCHEDULERS,  # scheduler combo
        IO.STRING,  # scheduler_name
        IO.INT,  # seed
        IO.INT,  # steps
        IO.VAE,
    )
    RETURN_NAMES = (
        "cfg",
        "checkpoint_name",
        "clip",
        "denoise",
        "latent",
        "model",
        "sampler_name",
        "sampler_string",
        "scheduler",
        "scheduler_name",
        "seed",
        "steps",
        "vae",
    )
    OUTPUT_TOOLTIPS = (
        "CFG scale applied during sampling.",
        "Name of the checkpoint used for loading.",
        "CLIP model bundled with the checkpoint or supplied externally.",
        "Denoising strength value.",
        "Denoised latent produced with the configured sampler.",
        "Model used for denoising (loaded or supplied).",
        "Sampler callable suitable for SAMPLER inputs.",
        "Sampler choice that was applied (string).",
        "Scheduler choice that was applied (combo).",
        "Scheduler name provided as a string output.",
        "Seed applied for sampling.",
        "Number of diffusion steps used.",
        "VAE model bundled with the checkpoint or supplied externally.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/LFGG"

    def sample(
        self,
        ckpt_name,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        model=None,
        clip=None,
        vae=None,
    ):
        model_obj, clip_obj, vae_obj, checkpoint_label = _prepare_assets(
            ckpt_name,
            model,
            clip,
            vae,
        )
        sampler_obj = _resolve_sampler(sampler_name)
        latent_out, = common_ksampler(
            model_obj,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )
        return (
            cfg,
            checkpoint_label,
            clip_obj,
            denoise,
            latent_out,
            model_obj,
            sampler_obj,
            sampler_name,
            scheduler,
            scheduler,
            seed,
            steps,
            vae_obj,
        )


class LFGGKSamplerAdvancedConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint to load before sampling (ignored if a model input is connected)."},
                ),
                "add_noise": (
                    ["enable", "disable"],
                    {"tooltip": "Enable to inject fresh noise before denoising."},
                ),
                "noise_seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "Seed used when generating noise (if enabled).",
                    },
                ),
                "steps": (
                    IO.INT,
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of denoising steps to perform.",
                    },
                ),
                "cfg": (
                    IO.FLOAT,
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Classifier-free guidance scale controlling adherence to prompts.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler algorithm to drive the diffusion process."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Noise schedule shaping how each step is taken."},
                ),
                "positive": (
                    IO.CONDITIONING,
                    {"tooltip": "Conditioning describing desired attributes."},
                ),
                "negative": (
                    IO.CONDITIONING,
                    {"tooltip": "Conditioning describing attributes to avoid."},
                ),
                "latent_image": (
                    IO.LATENT,
                    {"tooltip": "Latent tensor to denoise."},
                ),
                "start_at_step": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Step index to start sampling from.",
                    },
                ),
                "end_at_step": (
                    IO.INT,
                    {
                        "default": 10000,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Step index at which to stop sampling.",
                    },
                ),
                "return_with_leftover_noise": (
                    ["disable", "enable"],
                    {"tooltip": "Enable to preserve leftover noise in the latent output."},
                ),
            },
            "optional": {
                "model": (
                    IO.MODEL,
                    {"tooltip": "Optional preloaded model to use instead of loading from a checkpoint."},
                ),
                "clip": (
                    IO.CLIP,
                    {"tooltip": "Optional CLIP to forward when supplying an external model."},
                ),
                "vae": (
                    IO.VAE,
                    {"tooltip": "Optional VAE to forward when supplying an external model."},
                ),
            },
        }

    RETURN_TYPES = (
        IO.STRING,  # add_noise
        IO.FLOAT,  # cfg
        IO.STRING,  # checkpoint_name
        IO.CLIP,
        IO.INT,  # end_at_step
        IO.LATENT,
        IO.MODEL,
        IO.INT,  # noise_seed
        IO.STRING,  # return_with_leftover_noise
        IO.SAMPLER,  # sampler_name (callable)
        IO.STRING,  # sampler_string
        comfy.samplers.KSampler.SCHEDULERS,  # scheduler combo
        IO.STRING,  # scheduler_name
        IO.INT,  # start_at_step
        IO.INT,  # steps
        IO.VAE,
    )
    RETURN_NAMES = (
        "add_noise",
        "cfg",
        "checkpoint_name",
        "clip",
        "end_at_step",
        "latent",
        "model",
        "noise_seed",
        "return_with_leftover_noise",
        "sampler_name",
        "sampler_string",
        "scheduler",
        "scheduler_name",
        "start_at_step",
        "steps",
        "vae",
    )
    OUTPUT_TOOLTIPS = (
        "Whether fresh noise was added before denoising.",
        "CFG scale applied during sampling.",
        "Name of the checkpoint used for loading.",
        "CLIP model bundled with the checkpoint or supplied externally.",
        "Final step index used for denoising.",
        "Denoised latent produced with the configured sampler.",
        "Model used for denoising (loaded or supplied).",
        "Seed used to initialise noise (when enabled).",
        "Flag indicating if leftover noise should be kept.",
        "Sampler callable suitable for SAMPLER inputs.",
        "Sampler choice that was applied (string).",
        "Scheduler choice that was applied (combo).",
        "Scheduler name provided as a string output.",
        "First step index used for denoising.",
        "Number of diffusion steps used.",
        "VAE model bundled with the checkpoint or supplied externally.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/LFGG"

    def sample(
        self,
        ckpt_name,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        model=None,
        clip=None,
        vae=None,
    ):
        model_obj, clip_obj, vae_obj, checkpoint_label = _prepare_assets(
            ckpt_name,
            model,
            clip,
            vae,
        )
        sampler_obj = _resolve_sampler(sampler_name)
        disable_noise = add_noise == "disable"
        force_full_denoise = return_with_leftover_noise != "enable"
        latent_out, = common_ksampler(
            model_obj,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )
        return (
            add_noise,
            cfg,
            checkpoint_label,
            clip_obj,
            end_at_step,
            latent_out,
            model_obj,
            noise_seed,
            return_with_leftover_noise,
            sampler_obj,
            sampler_name,
            scheduler,
            scheduler,
            start_at_step,
            steps,
            vae_obj,
        )


NODE_CLASS_MAPPINGS = {
    "LFGG KSampler - Config": LFGGKSamplerConfig,
    "LFGG KSampler (advanced) - Config": LFGGKSamplerAdvancedConfig,
}