from collections import OrderedDict
from typing import Optional, Union

from torch.utils.data import DataLoader

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import ModelConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds, PromptEmbeds
from toolkit.sampler import get_sampler
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork, UNET_IN_CHANNELS
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
from diffusers import PixArtTransformer2DModel
import gc
import torch
from library.model_util import load_vae
import copy


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class KnowledgeTransferChannelAdder(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        model_source_config = self.config.get('model_source', None)
        self.model_source_config = ModelConfig(**model_source_config)
        self.device2 = self.get_conf('device_2', self.device)
        self.device2_dtype = self.get_conf('device2_dtype', self.train_config.dtype)
        self.device2_torch = torch.device(self.device2)
        self.do_prior_prediction = True
        self.step_prediction = self.get_conf('step_prediction', False)
        self.caption_list = []
        self.alternate = self.get_conf('alternate', False)
        self.num_channels = self.get_conf('num_channels', 16)
        self.vae_path = self.get_conf('vae_path', None)

        sampler = get_sampler(
            self.train_config.noise_scheduler,
            {
                "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
            },
            'sd' if not self.model_config.is_pixart else 'pixart'
        )

        self.sd_source = StableDiffusion(
            device=self.device2,
            model_config=self.model_source_config,
            dtype=self.device2_dtype,
            noise_scheduler=sampler
        )

    def hook_after_model_load(self):
        # patch the model
        transformer: PixArtTransformer2DModel = self.sd.unet
        # cat pos_embed.proj.weight 4x on idx 1
        # cat proj_out.weight 4x on idx 0
        # cat proj_out.bias 4x on idx 0
        if transformer.config.in_channels != self.num_channels:
            num_cats = self.num_channels // transformer.config.in_channels

            transformer.pos_embed.proj.weight.data = torch.cat([transformer.pos_embed.proj.weight.data] * num_cats, dim=1)
            transformer.proj_out.weight.data = torch.cat([transformer.proj_out.weight.data] * num_cats, dim=0)
            transformer.proj_out.bias.data = torch.cat([transformer.proj_out.bias.data] * num_cats, dim=0)

            transformer.config.in_channels = self.num_channels
            transformer.config['in_channels'] = self.num_channels
            transformer.config.out_channels = self.num_channels * 2
            transformer.config['out_channels'] = self.num_channels * 2

        # hack in the vae after model loads or it will fail
        if self.vae_path is not None:
            vae = load_vae(self.vae_path, dtype=get_torch_dtype(self.sd.dtype))
            vae.to(self.sd.device_torch).eval()
            self.sd.vae = vae
            self.sd.pipeline.vae = vae

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)

        # setup source model
        self.sd_source.load_model()
        te_list = [self.sd_source.text_encoder]
        if isinstance(self.sd_source.text_encoder, list):
            te_list = self.sd_source.text_encoder
        for te in te_list:
            te.eval()
            te.to(self.device2_torch)
        self.sd_source.unet.eval()
        self.sd_source.unet.to(self.device2_torch)
        self.sd_source.vae.to(self.device2_torch)
        self.sd_source.vae.eval()

        if self.train_config.xformers:
            self.sd_source.vae.set_use_memory_efficient_attention_xformers(True)
            self.sd_source.unet.enable_xformers_memory_efficient_attention()
        if self.train_config.gradient_checkpointing:
            self.sd_source.unet.enable_gradient_checkpointing()

    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        is_reg = any(batch.get_is_reg_list())
        if is_reg:
            return None
        if self.alternate and self.step % 2 == 0:
            return None
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False

        with torch.no_grad():
            # slice the noisy latents
            noisy_latents = noisy_latents[:, :self.sd_source.unet.config.in_channels]
            dtype = get_torch_dtype(self.device2_dtype)
            device_torch = self.device2_torch

            embeds_to_use = conditional_embeds.clone().detach()
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            embeds = self.sd_source.encode_prompt(
                conditioned_prompts,
                long_prompts=self.do_long_prompts).to(
                self.device_torch,
                dtype=dtype).detach()

            unconditional_to_use = None
            if unconditional_embeds is not None:
                unconditional_to_use = self.sd_source.encode_prompt(
                    self.batch_negative_prompt,
                    long_prompts=self.do_long_prompts).to(
                    device_torch,
                    dtype=dtype
                ).detach()

            prior_pred = self.sd_source.predict_noise(
                latents=noisy_latents.to(device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds.to(device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_to_use,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                **pred_kwargs  # adapter residuals in here
            )

            if self.step_prediction:
                prior_pred = self.sd_source.step_scheduler(prior_pred, noisy_latents, timesteps)

            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active

            # cat the noisy latents
            num_cats = self.num_channels // self.sd_source.unet.config.in_channels
            prior_pred = torch.cat([prior_pred] * num_cats, dim=1)
        return prior_pred.to(self.sd.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        if self.step_prediction:
            noise_pred = self.sd.step_scheduler(noise_pred, noisy_latents, timesteps)

        # we scale this prediction to match the parent model
        if self.model_source_config.is_v_pred and not self.model_config.is_v_pred:
            noise_pred = self.sd_source.noise_scheduler.get_velocity(batch.latents, noise_pred, timesteps)

        # match the noist latents in the prior pred
        noisy_latents = noisy_latents[:, :self.sd_source.unet.config.in_channels]
        noise = noise[:, :self.sd_source.unet.config.in_channels]

        num_cats = self.num_channels // self.sd_source.unet.config.in_channels
        noisy_latents = torch.cat([noisy_latents] * num_cats, dim=1)
        noise = torch.cat([noise] * num_cats, dim=1)

        return super().calculate_loss(
            noise_pred=noise_pred,
            noise=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            batch=batch,
            mask_multiplier=mask_multiplier,
            prior_pred=prior_pred,
            **kwargs
        )

    def predict_noise(
            self,
            noisy_latents: torch.Tensor,
            timesteps: Union[int, torch.Tensor] = 1,
            conditional_embeds: Union[PromptEmbeds, None] = None,
            unconditional_embeds: Union[PromptEmbeds, None] = None,
            **kwargs,
    ):
        with torch.no_grad():
            # make them so they are concatenated
            noisy_latents = noisy_latents[:, :self.sd_source.unet.config.in_channels]
            num_cats = self.num_channels // self.sd_source.unet.config.in_channels
            noisy_latents = torch.cat([noisy_latents] * num_cats, dim=1)
            noisy_latents = noisy_latents.detach()

        dtype = get_torch_dtype(self.train_config.dtype)
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            **kwargs
        )