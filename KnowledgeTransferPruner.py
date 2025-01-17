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
from diffusers import AuraFlowTransformer2DModel, FluxTransformer2DModel
import gc
import torch
import copy


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class KnowledgeTransferPruner(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        model_source_config = self.config.get('model_source', None)
        self.model_source_config = ModelConfig(**model_source_config)
        self.device2 = self.get_conf('device2', self.device)
        self.device2_dtype = self.get_conf('device2_dtype', self.train_config.dtype)
        self.device2_torch = torch.device(self.device2)
        self.load_device2 = self.get_conf('load_device2', self.device2)
        self.load_device2_torch = torch.device(self.load_device2)
        self.do_prior_prediction = True
        self.step_prediction = self.get_conf('step_prediction', False)
        self.caption_list = []
        self.alternate = self.get_conf('alternate', False)
        self.num_layers = self.get_conf('num_layers', 28)
        self.num_single_layers = self.get_conf('num_single_layers', 28)
        self.random_noise_shift = self.get_conf('random_noise_shift', False)

        sampler = get_sampler(
            self.train_config.noise_scheduler,
            {
                "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
            },
            'sd' if not self.model_config.is_pixart else 'pixart'
        )

        self.sd_source = StableDiffusion(
            device=self.load_device2,
            model_config=self.model_source_config,
            dtype=self.device2_dtype,
            noise_scheduler=sampler,
            quantize_device=self.device
        )

        # setup source model
        self.sd_source.load_model()
        flush()
        self.sd_source.pipeline.to(self.device2_torch)
        flush()

    def hook_after_model_load(self):
        if self.sd.is_auraflow:
            # remove layers if we have more
            transformer: AuraFlowTransformer2DModel = self.sd.unet
            transformer_blocks = [x for x in transformer.single_transformer_blocks]
            if len(transformer_blocks) > self.num_layers:
                # keep last layer but remove from -2
                transformer_blocks = transformer_blocks[:self.num_layers - 1] + transformer_blocks[-1:]
            transformer.single_transformer_blocks = torch.nn.ModuleList(transformer_blocks)
            transformer.config.num_single_dit_layers = self.num_layers
            transformer.config['num_single_dit_layers'] = self.num_layers
        elif self.sd.is_flux:
            # remove layers if we have more
            transformer: FluxTransformer2DModel = self.sd.unet
            transformer_blocks = [x for x in transformer.transformer_blocks]
            if len(transformer_blocks) > self.num_layers:
                # keep last layer but remove from -2
                transformer_blocks = transformer_blocks[:self.num_layers - 1] + transformer_blocks[-1:]
            transformer.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
            transformer.config.num_layers = self.num_layers
            transformer.config['num_layers'] = self.num_layers

            # single layers
            single_transformer_blocks = [x for x in transformer.single_transformer_blocks]
            if len(single_transformer_blocks) > self.num_single_layers:
                # keep last layer but remove from -2
                single_transformer_blocks = single_transformer_blocks[:self.num_single_layers - 1] + single_transformer_blocks[-1:]
            transformer.single_transformer_blocks = torch.nn.ModuleList(single_transformer_blocks)
            transformer.config.num_single_layers = self.num_single_layers
            transformer.config['num_single_layers'] = self.num_single_layers

        else:
            # remove layers if we have more
            transformer: PixArtTransformer2DModel = self.sd.unet
            transformer_blocks = [x for x in transformer.transformer_blocks]
            if len(transformer_blocks) > self.num_layers:
                # keep last layer but remove from -2
                transformer_blocks = transformer_blocks[:self.num_layers - 1] + transformer_blocks[-1:]
            if len(transformer_blocks) < self.num_layers:
                num_blocks_to_add = self.num_layers - len(transformer_blocks)
                # clone the middle number of blocks
                center_block_idx = len(transformer_blocks) // 2
                start_clone_idx = center_block_idx - num_blocks_to_add // 2
                if start_clone_idx < 0:
                    while True:
                        # copy all blocks into center to expand it
                        transformer_blocks = transformer_blocks + copy.deepcopy(transformer_blocks)
                        center_block_idx = len(transformer_blocks) // 2
                        start_clone_idx = center_block_idx - num_blocks_to_add // 2
                        if start_clone_idx >= 0:
                            break

                cloned_blocks = transformer_blocks[start_clone_idx:start_clone_idx + num_blocks_to_add]
                cloned_blocks = [copy.deepcopy(x) for x in cloned_blocks]
                # put them in the middle
                transformer_blocks = transformer_blocks[:center_block_idx] + cloned_blocks + transformer_blocks[center_block_idx:]
            transformer.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
            transformer.config.num_layers = self.num_layers
            transformer.config['num_layers'] = self.num_layers

        flush()


    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)

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
        # if self.train_config.gradient_checkpointing:
        #     self.sd_source.unet.enable_gradient_checkpointing()

        flush()

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
        return prior_pred.to(self.sd.device_torch, dtype=get_torch_dtype(self.train_config.dtype)).detach()

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