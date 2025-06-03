# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from functools import partial

import torch
from dinov2.fsdp import (
    ShardedGradScaler,
    get_fsdp_modules,
    get_fsdp_wrapper,
    reshard_fsdp_model,
)
from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from dinov2.models import build_model_from_cfg
from dinov2.models.vision_transformer import BlockChunk
from dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay
from dinov2.utils.utils import has_batchnorms
from torch import nn

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # This cfg will be updated by setup_config_with_grad_accum in the main script
        self.fp16_scaler = (
            ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        )

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights, weights_only=False)
            logger.info(
                f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}"
            )
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(
                f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}"
            )
            self.dino_loss_weight = cfg.dino.loss_weight
            # dino_head needs to be defined here for the case where only do_dino is true
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")
            # Define dino_head even if not used by DINO, in case iBOT needs it (if not separate_head)
            if self.do_ibot and not self.ibot_separate_head:
                dino_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.dino.head_n_prototypes,  # iBOT will use DINO's prototype count
                    hidden_dim=cfg.dino.head_hidden_dim,
                    bottleneck_dim=cfg.dino.head_bottleneck_dim,
                    nlayers=cfg.dino.head_nlayers,
                )

        if self.do_dino or (
            self.do_ibot and not self.ibot_separate_head
        ):  # Ensure dino_head is defined if needed
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}"
        )
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}"
        )
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert (
                max(cfg.ibot.mask_ratio_min_max) > 0
            ), "please provide a positive mask ratio tuple for ibot"
            assert (
                cfg.ibot.mask_sample_probability > 0
            ), "please provide a positive mask probability for ibot"
            self.ibot_out_dim = (
                cfg.ibot.head_n_prototypes
                if self.ibot_separate_head
                else cfg.dino.head_n_prototypes
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(
                    f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}"
                )
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(
            f"Student and Teacher are built: they are both {cfg.student.arch} network."
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        # Retrieve gradient_accumulation_steps from the config object
        # This config is updated by the main training script
        gradient_accumulation_steps = self.cfg.train.get(
            "gradient_accumulation_steps", 1
        )

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps  # Scale the loss

        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat(
                (teacher_cls_tokens[1], teacher_cls_tokens[0])
            )
            ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens, _dim
                )
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound, _dim
                )
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                    buffer_tensor_teacher
                )[:n_masked_patches]
            else:  # Only DINO or iBOT with shared head (but do_ibot is false)
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_ibot_softmaxed_centered = None  # Ensure it's defined

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:  # This condition is important
                    masked_teacher_patch_tokens_after_head = (
                        masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_after_head[
                                :, :n_masked_patches
                            ],
                            teacher_temp=teacher_temp,
                        )
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    )
                    self.ibot_patch_loss.update_center(
                        masked_teacher_patch_tokens_after_head[:n_masked_patches]
                    )

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )

                if do_ibot:  # This condition is important
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            masked_teacher_patch_tokens_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor=n_masked_patches_tensor,
                        )
                    )
            else:
                raise NotImplementedError

            return (
                teacher_dino_softmaxed_centered_list,
                masked_teacher_ibot_softmaxed_centered,
            )

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            get_teacher_output()
        )
        reshard_fsdp_model(self.teacher)

        loss_dict = {}
        loss_accumulator = 0  # for backprop

        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        )

        inputs_for_student_head_list = []

        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        loss_scales = 2

        if do_dino:
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens
            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / loss_scales

        if do_ibot:
            ibot_patch_loss_val = (  # Renamed to avoid conflict with self.ibot_patch_loss
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )
            loss_dict["ibot_loss"] = (
                ibot_patch_loss_val / 2
            )  # Original scaling for display
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss_val

        # This is where the .backward() happens, scaled by gradient_accumulation_steps
        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()
        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # if hasattr(self.student, "dino_head") and self.student.dino_head is not None and hasattr(self.student.dino_head, "_streams"):
            #     self.student.dino_head._streams = self.teacher.dino_head._streams = \
            #     self.student.backbone._streams = self.teacher.backbone._streams
            # elif hasattr(self.student, "ibot_head") and self.student.ibot_head is not None and hasattr(self.student.ibot_head, "_streams"): # if only ibot_head exists
            #      self.student.ibot_head._streams = self.teacher.ibot_head._streams = \
            #      self.student.backbone._streams = self.teacher.backbone._streams
            # else: # Fallback if heads are not present or don't have _streams (should not happen with FSDP)
            #      self.student.backbone._streams = self.teacher.backbone._streams

            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(
                    get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])
                ):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        # logger.info("fusing param groups") # Moved to main script for less verbosity

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for (
            m_name,
            m_module,
        ) in self.student.items():  # Iterate through student's submodules
            logger.info(f"Getting params for student submodule: {m_name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m_module)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError(
                "FSDP with BatchNorm not fully supported/tested here, DINOv2 uses LayerNorm."
            )

        # Synchronize student and teacher before FSDP wrapping for consistency
        for k, student_module in self.student.items():
            if k in self.teacher:
                self.teacher[k].load_state_dict(student_module.state_dict())
            else:
                logger.warning(
                    f"Module {k} found in student but not in teacher during FSDP preparation."
                )

        # Apply FSDP wrapper
        for k in list(self.student.keys()):  # Use list to avoid issues if dict changes
            if k not in self.cfg.compute_precision.student:
                logger.warning(
                    f"No FSDP student config for module {k}, skipping FSDP wrapping for this module."
                )
                continue
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(
                student_model_cfg, modules_to_wrap={BlockChunk}
            )(self.student[k])

            if k not in self.teacher:
                logger.warning(
                    f"Module {k} wrapped for student but not present in teacher."
                )
                continue
            if k not in self.cfg.compute_precision.teacher:
                logger.warning(
                    f"No FSDP teacher config for module {k}, skipping FSDP wrapping for this teacher module."
                )
                continue
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(
                teacher_model_cfg, modules_to_wrap={BlockChunk}
            )(self.teacher[k])

    def load_state_dict_from_checkpoint_custom(
        self,
        checkpoint_data_or_state_dict,
        source_key=None,
        target_component_name="student",
        target_sub_component_name="backbone",
        strict=False,
    ):
        """
        Loads a state_dict from a checkpoint into a specified sub-component (e.g., student's backbone) of this SSLMetaArch model.
        Handles raw state_dicts or state_dicts nested under a key.

        Args:
            checkpoint_data_or_state_dict (dict): The loaded checkpoint dictionary OR the raw state_dict itself.
            source_key (str, optional): The key in checkpoint_data_or_state_dict if the state_dict is nested.
                                        If None, assumes checkpoint_data_or_state_dict *is* the state_dict.
            target_component_name (str): "student" or "teacher".
            target_sub_component_name (str): The name of the module within the student/teacher component
                                             to load into (e.g., "backbone", "head").
            strict (bool): Whether to strictly enforce key matching.

        Returns:
            NamedTuple with missing_keys and unexpected_keys, or None on critical error.
        """
        logger.info(
            f"Custom loading: source_key='{source_key}', target_component='{target_component_name}', target_sub_component='{target_sub_component_name}'"
        )

        state_dict_to_load = None
        if source_key is None:
            if not isinstance(checkpoint_data_or_state_dict, dict):
                logger.error(
                    f"Checkpoint data provided as raw state_dict is not a dictionary. Type: {type(checkpoint_data_or_state_dict)}"
                )
                return None
            state_dict_to_load = checkpoint_data_or_state_dict
            logger.info(
                f"  Interpreting checkpoint data as a raw state_dict for '{target_sub_component_name}'."
            )
        elif source_key in checkpoint_data_or_state_dict:
            state_dict_to_load = checkpoint_data_or_state_dict[source_key]
            if not isinstance(state_dict_to_load, dict):
                logger.error(
                    f"  Data under source key '{source_key}' is not a state_dict. Type: {type(state_dict_to_load)}"
                )
                return None
            logger.info(f"  Extracted state_dict from checkpoint key '{source_key}'.")
        else:
            logger.error(
                f"  Source key '{source_key}' not found in checkpoint. Available keys: {list(checkpoint_data_or_state_dict.keys())}"
            )
            return None

        # Get the target component (e.g., self.student, which is a dict of modules)
        target_component_dict = getattr(self, target_component_name, None)
        if not isinstance(target_component_dict, dict):
            logger.error(
                f"Target component '{target_component_name}' is not a valid dictionary of modules in SSLMetaArch."
            )
            return None

        # Get the specific sub-component module instance (e.g., self.student["backbone"])
        module_instance_to_load = target_component_dict.get(target_sub_component_name)
        if not isinstance(module_instance_to_load, nn.Module):
            logger.error(
                f"Target sub-component '{target_component_name}.{target_sub_component_name}' is not a valid nn.Module."
            )
            return None

        # Handle FSDP/DDP unwrapping if necessary
        # For loading into an FSDP module, it's generally expected that the state_dict is full (non-sharded)
        # if the checkpoint itself is not from an FSDP save.
        actual_model_to_load_into = module_instance_to_load
        if isinstance(module_instance_to_load, FSDP):
            # FSDP's load_state_dict should handle loading a full state_dict.
            # Ensure the model is on the correct device or use FSDP utils if loading sharded state.
            logger.info(
                f"  Target module '{target_sub_component_name}' is FSDP wrapped."
            )
        elif isinstance(module_instance_to_load, DDP):
            actual_model_to_load_into = module_instance_to_load.module
            logger.info(
                f"  Target module '{target_sub_component_name}' is DDP wrapped. Loading into underlying module."
            )

        logger.info(
            f"  Attempting load_state_dict for '{target_component_name}.{target_sub_component_name}' with {len(state_dict_to_load)} keys."
        )

        try:
            # PyTorch's load_state_dict can sometimes handle minor mismatches if keys are prefixed
            # (e.g. loading a DDP-saved model into a non-DDP model if keys start with 'module.')
            # but for raw state_dicts for a backbone, it should be fairly direct.
            incompatible = actual_model_to_load_into.load_state_dict(
                state_dict_to_load, strict=strict
            )

            logger.info(
                f"    Successfully attempted load for '{target_component_name}.{target_sub_component_name}'."
            )
            if incompatible.missing_keys:
                logger.warning(
                    f"    Missing keys for '{target_sub_component_name}': {incompatible.missing_keys}"
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    f"    Unexpected keys for '{target_sub_component_name}': {incompatible.unexpected_keys}"
                )
            return incompatible

        except Exception as module_load_e:
            logger.error(
                f"    Error loading state_dict for '{target_component_name}.{target_sub_component_name}': {module_load_e}",
                exc_info=True,
            )
            from torch.nn.modules.module import (
                _IncompatibleKeys,
            )  # Import locally for safety

            return _IncompatibleKeys(
                missing_keys=list(actual_model_to_load_into.state_dict().keys()),
                unexpected_keys=[],
            )
