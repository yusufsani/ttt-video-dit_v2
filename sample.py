import os
from typing import List

import imageio
import numpy as np
import torch
import wandb
from einops import rearrange
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from ttt.infra.config_manager import JobConfig
from ttt.infra.parallelisms import TORCH_DTYPE_MAP, end_distributed, get_world_info, init_distributed
from ttt.infra.utils import display_logo, set_random_seed
from ttt.models.cogvideo.model import CogVideoX
from ttt.models.cogvideo.sampler import DenoiserSampler, ModelLoader, PromptManager, TextEncoder
from ttt.models.vae.autoencoder import VideoAutoencoderInferenceWrapper


class VideoGenerator:
    """Handle generation of videos from text embeddings."""

    def __init__(
        self,
        model: CogVideoX,
        vae_model: VideoAutoencoderInferenceWrapper,
        tokenizer: T5Tokenizer,
        t5_encoder: T5EncoderModel,
        job_config: JobConfig,
        device: str,
        use_wandb: bool = False,
    ):
        """Initialize the video generator with models and configuration."""
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.t5_encoder = t5_encoder
        self.job_config = job_config
        self.device = device
        self.use_wandb = use_wandb

        effective_rank, _ = get_world_info(job_config)

        self.sampler = DenoiserSampler(
            model=model.dit,
            config=job_config,
            device=device,
            use_wandb=use_wandb,
            effective_rank=effective_rank,
            seed=job_config.job.seed,
            dtype=TORCH_DTYPE_MAP[job_config.eval.dtype],
        )

        self.dtype = TORCH_DTYPE_MAP[job_config.eval.dtype]

    def generate_video(self, prompts: List[str], neg_prompts: List[str]) -> torch.Tensor:
        """
        Generate a video based on the given text prompts.

        Args:
            prompts: Positive text prompts
            neg_prompts: Negative text prompts (optional)

        Returns:
            Tensor containing the generated video frames
        """
        # Encode text prompts.
        text_emb = TextEncoder.encode_text(
            self.tokenizer, self.t5_encoder, prompts, self.device, self.job_config.eval.txt_maxlen, self.dtype
        )
        print("smaple: text_emb.  ",text_emb.shape)
        # Empty string is used to represent None in the prompt list for classifier free guidance.
        neg_prompts = [txt or "" for txt in neg_prompts]
        neg_emb = TextEncoder.encode_text(
            self.tokenizer, self.t5_encoder, neg_prompts, self.device, self.job_config.eval.txt_maxlen, self.dtype
        )

        assert text_emb.ndim == neg_emb.ndim, "Positive and negative prompts must have the same shape"

        T, C, H, W, F = (
            self.job_config.eval.sampling_num_frames,
            self.job_config.eval.latent_channels,
            self.job_config.eval.image_height,
            self.job_config.eval.image_width,
            8,  # Factor for height and width scaling.
        )

        print("T, C, H, W, F ",T, C, H, W, F )
        latents = self.sampler.sample(text_emb=text_emb, neg_emb=neg_emb, shape=(T, C, H // F, W // F), batch_size=1)

        # Decode latents to frames.
        latents = rearrange(latents, "b t c h w -> b c t h w")
        decoded_frames = self.vae_model.decode_first_stage(latents)
        decoded_frames = rearrange(decoded_frames, "b c t h w -> b t c h w")

        # Normalize frames.
        samples = torch.clamp((decoded_frames + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        return samples


class VideoSaver:
    """Handle saving of generated videos."""

    @staticmethod
    def save_video(video_batch: torch.Tensor, save_path: str, fps: int, prompts: List[str]) -> None:
        """
        Save a batch of videos to disk.

        Args:
            video_batch: Tensor of video frames
            save_path: Directory to save videos
            fps: Frames per second
            prompts: Text prompts used to generate the videos
        """
        os.makedirs(save_path, exist_ok=True)

        # Save each video in the batch
        for i, video_tensor in enumerate(video_batch):
            gif_frames = [VideoSaver._prepare_frame(frame) for frame in video_tensor]

            video_file_path = os.path.join(save_path, f"{i:06d}.mp4")

            with imageio.get_writer(video_file_path, fps=fps) as writer:
                for frame in gif_frames:
                    writer.append_data(frame)  # type: ignore

        # Save prompts
        prompt_file_path = os.path.join(save_path, "prompt.txt")
        prompt_text = "\n\n".join(prompts) if len(prompts) > 1 else prompts[0]

        with open(prompt_file_path, "w", encoding="utf-8") as prompt_file:
            prompt_file.write(prompt_text)

    @staticmethod
    def _prepare_frame(frame: torch.Tensor) -> np.ndarray:
        """Convert a tensor frame to a numpy array suitable for saving."""
        frame = frame.to(torch.float32)
        frame = rearrange(frame, "c h w -> h w c")
        return (255.0 * frame).cpu().numpy().astype(np.uint8)


class VideoGenerationRunner:
    """Main runner for video generation pipeline."""

    def __init__(self, job_config: JobConfig):
        """Initialize the runner with the given job configuration."""
        assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."
        self.device = "cuda"

        self.job_config = job_config
        # --- Teacache config: set these as needed or expose via job_config ---
        self.enable_teacache = getattr(job_config, "enable_teacache", False)
        self.rel_l1_thresh = getattr(job_config, "rel_l1_thresh", 0.0004)
        self.teacache_coefficients = getattr(job_config, "teacache_coefficients", [-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02])
        self.teacache_num_steps = getattr(job_config, "teacache_num_steps", 50)
        # ---------------------------------------------------------------


        init_distributed(job_config)
        self.effective_rank, self.effective_world_size = get_world_info(job_config)
        set_random_seed(job_config.job.seed + self.effective_rank)

        self.tokenizer, self.t5_encoder = ModelLoader.load_t5_encoder(job_config, device=self.device)
        self.model = ModelLoader.load_cogvideox_model(
            job_config,
            self.effective_rank,
            self.effective_world_size,
            self.device,
            enable_teacache=self.enable_teacache,
            rel_l1_thresh=self.rel_l1_thresh,
            teacache_coefficients=self.teacache_coefficients,
            teacache_num_steps=self.teacache_num_steps,
        )

        
        self.vae_model = ModelLoader.load_vae_model(job_config, self.device)

        

        self.use_wandb = not job_config.wandb.disable and torch.distributed.get_rank() == 0

        self.video_generator = VideoGenerator(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            t5_encoder=self.t5_encoder,
            job_config=job_config,
            device=self.device,
            use_wandb=self.use_wandb,
        )
        

    def run(self):
        """Run the video generation pipeline."""
        assert self.job_config.eval.input_file, "Input file is required for evaluation."

        prompts = PromptManager.get_prompts(input_file=self.job_config.eval.input_file)

        per_rank_prompts = prompts[self.effective_rank :: self.effective_world_size]

        save_path = os.path.join(self.job_config.eval.output_dir, self.job_config.job.exp_name)

        if self.use_wandb:
            self._init_wandb()

        iterator = self._create_progress_iterator(per_rank_prompts)

        for i, (pos_prompts, neg_prompts) in enumerate(iterator):
            # Log progress if using W&B
            if self.use_wandb and (i + 1) % self.job_config.wandb.log_interval == 0:
                wandb.log(
                    {
                        "progress/step": i,
                        "progress/remaining": len(per_rank_prompts) - i,
                    }
                )
            import time

            start_time = time.time()
            # Generate video
            samples = self.video_generator.generate_video(pos_prompts, neg_prompts)
            end_time = time.time()
            print(f"**************Time taken: {end_time - start_time:.4f} seconds")

            # Save video if this rank is responsible for output
            if torch.distributed.get_rank() % self.job_config.parallelism.tp_sharding == 0:
                prompt_idx = i * self.effective_world_size + self.effective_rank
                output_path = os.path.join(save_path, f"prompt-{prompt_idx}")

                VideoSaver.save_video(
                    video_batch=samples,
                    save_path=output_path,
                    fps=self.job_config.eval.sampling_fps,
                    prompts=pos_prompts,
                )

            if self.use_wandb and self.job_config.wandb.alert and wandb.run:
                wandb.run.alert(
                    title="Prompt processed",
                    text=f"Prompt {i} processed for sampling job '{self.job_config.job.exp_name}'.",
                )

        if wandb.run:
            wandb.finish()

        end_distributed()

    def _init_wandb(self):
        """Initialize W&B logging."""
        wandb.init(
            project=self.job_config.wandb.project,
            entity=self.job_config.wandb.entity,
            name=self.job_config.job.exp_name,
            config=self.job_config.to_dict(),
        )

    def _create_progress_iterator(self, prompts):
        """Create an iterator with progress tracking if this is the main process."""
        if torch.distributed.get_rank() == 0:
            return tqdm(prompts, desc="Processing prompts", dynamic_ncols=True)
        else:
            return prompts


@record
def main(job_config: JobConfig):
    """Main entry point for the video generation script."""
    display_logo()

    runner = VideoGenerationRunner(job_config)
    runner.run()


if __name__ == "__main__":
    config = JobConfig(eval_mode=True)

    config.parse_args()

    main(config)
