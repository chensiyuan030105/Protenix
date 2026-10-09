# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114,C0301
from protenix.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

basic_configs = {
    "project": RequiredValue(str),
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    # training
    "eval_interval": RequiredValue(int),
    "log_interval": RequiredValue(int),
    "checkpoint_interval": -1,
    "eval_first": False,  # run evaluate() before training steps
    "iters_to_accumulate": 1,
    "finetune_params_with_substring": [
        ""
    ],  # params with substring will be finetuned with different learning rate: finetune_optim_configs["lr"]
    "eval_only": False,
    "load_checkpoint_path": "",
    "load_ema_checkpoint_path": "",
    "load_strict": True,
    "load_params_only": True,
    "skip_load_step": False,
    "skip_load_optimizer": False,
    "skip_load_scheduler": False,
    "load_step_for_scheduler": False,
    "train_confidence_only": False,
    # Kineidos (P004): do not run the confidence / distogram heads at all when
    # their losses are off, rather than running them and weighting them to zero.
    # See protenix/model/protenix.py's __init__ for why zero weights are not
    # enough.  False reproduces upstream.
    "skip_confidence": False,
    "skip_distogram": False,
    # Separate from skip_confidence on purpose: the mini-rollout block also
    # permutes the label, and whether that permutation ever changes anything on
    # GAGU is a measurement we have not made.  Requires skip_confidence.
    "skip_mini_rollout": False,
    "use_wandb": True,
    "wandb_id": "",
    "seed": 42,
    "deterministic": False,
    "deterministic_seed": False,
    "ema_decay": -1.0,
    "eval_ema_only": False,  # whether wandb only tracking ema checkpoint metrics
    "ema_mutable_param_keywords": [""],
    "model_name": "protenix_base_default_v1.0.0",  # train model name
}
data_configs = {
    # Data
    "train_crop_size": 256,
    "test_max_n_token": -1,
    "train_lig_atom_rename": False,
    "train_shuffle_mols": False,
    "train_shuffle_sym_ids": False,
    "test_lig_atom_rename": False,
    "test_shuffle_mols": False,
    "test_shuffle_sym_ids": False,
    "esm": {
        "enable": False,
        "model_name": "esm2-3b",
        "embedding_dim": 2560,
    },
}
optim_configs = {
    # Optim
    "lr": 0.0018,
    "lr_scheduler": "af3",
    "warmup_steps": 10,
    "max_steps": RequiredValue(int),
    "min_lr_ratio": 0.1,
    "decay_every_n_steps": 50000,
    "grad_clip_norm": 10,
    # Optim - Adam
    "adam": {
        "beta1": 0.9,
        "beta2": 0.95,
        # P010 D16, 2026-10-08: 0 and not 1e-8, which P009 section 8 item 8
        # handed to this plan to fix.
        #
        # The optimizer is plain Adam (use_adamw below is False), so weight
        # decay is *coupled* -- added to the gradient and then put through
        # Adam's normalisation.  `eps` is not passed, so it is torch's default
        # 1e-8, which is exactly `weight_decay`.  For a parameter whose data
        # gradient is ~0 the whole gradient is wd*theta, and
        #
        #     update = lr * g/(sqrt(g^2) + eps)
        #            = lr * (1e-8*theta)/(1e-8*theta + 1e-8)
        #            = lr * theta/(theta + 1)        <- wd cancels out
        #
        # which for theta << 1 is exponential decay at rate = lr.  The size of
        # wd does not matter; it appears in numerator and denominator.
        #
        # Measured with these exact hyperparameters (theta0 = 1, grad = 0):
        #
        #   wd=1e-8, eps=1e-8   500 steps 0.6037   2000 0.0678   5000 0.000299
        #   wd=0                          1.000000        1.000000      1.000000
        #   eps=1e-6                      0.9911          0.9650        0.9147
        #   use_adamw=True                1.000000        1.000000      1.000000
        #
        # The victim class is narrow -- theta != 0 *and* data gradient
        # <~ 1e-8*theta -- and in this codebase it is one parameter:
        # wp_layernorm.weight, initialised to 1.  P009 measured the `zero`
        # arm's gamma decaying at 1.83e-3/step, equal to lr, with a provably
        # zero data gradient; and `random`'s at 0.33e-3/step, 5.5x slower,
        # which is how it established that the data gradient *resists* the
        # artefact rather than causing the decay.
        #
        # Why 0 rather than use_adamw=True, which also fixes it: this is the
        # smaller change.  1e-8 is four to six orders of magnitude below any
        # real weight decay (1e-2 to 1e-4), so it was never regularising
        # anything; setting it to 0 removes a force that was only ever an
        # artefact.  use_adamw=True would additionally switch 442M parameters
        # to decoupled decay and regroup every 1-D parameter.
        #
        # Why this matters past this one parameter, which is P009's point and
        # the reason it is fixed rather than annotated: the experiment this
        # project keeps running is "add a gated side branch and see whether the
        # model uses it".  If the gate starts non-zero and gets a weak
        # gradient, Adam closes it whatever the data does -- and the gate's
        # value is exactly the quantity being read.
        "weight_decay": 0.0,
        "lr": GlobalConfigValue("lr"),
        "use_adamw": False,
    },
    # Optim - LRScheduler
    "af3_lr_scheduler": {
        "warmup_steps": GlobalConfigValue("warmup_steps"),
        "decay_every_n_steps": GlobalConfigValue("decay_every_n_steps"),
        "decay_factor": 0.95,
        "lr": GlobalConfigValue("lr"),
    },
}
# Fine-tuned optimizer settings.
# For models supporting structural constraints and ESM embeddings.
finetune_optim_configs = {
    # Optim
    "lr": 0.0018,
    "lr_scheduler": "cosine_annealing",
    "warmup_steps": 1000,
    "max_steps": 20000,
    "min_lr_ratio": 0.1,
    "decay_every_n_steps": 50000,
}
model_configs = {
    "mc_dropout_apply_rate": 0.4,
    "mc_dropout_rate": 0.4,
    # Model
    "c_s": 384,
    "c_z": 128,
    "c_s_inputs": 449,  # c_s_inputs == c_token + 32 + 32 + 1
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,
    "max_atoms_per_token": 24,  # DNA G max_atoms = 23
    "no_bins": 64,
    "sigma_data": 16.0,
    "diffusion_batch_size": 48,
    "diffusion_chunk_size": ValueMaybeNone(4),  # chunksize of diffusion_batch_size
    "blocks_per_ckpt": ValueMaybeNone(
        1
    ),  # NOTE: Number of blocks in each activation checkpoint, if None, no checkpointing is performed.
    "hidden_scale_up": False,  # whether to scale up hidden dim in pairformer and confidence head
    # switch of kernels
    "triangle_multiplicative": "cuequivariance",  # cuequivariance, torch
    "triangle_attention": "cuequivariance",  # triattention, cuequivariance, deepspeed, torch
    "enable_diffusion_shared_vars_cache": False,
    "enable_efficient_fusion": False,
    "enable_tf32": False,
    "find_unused_parameters": False,
    "dtype": "bf16",  # default training dtype: bf16
    "loss_metrics_sparse_enable": True,  # the swicth for both sparse lddt metrics and sparse bond/smooth lddt loss
    "skip_amp": {
        "sample_diffusion": True,
        # If confidence_head (below) set to True and triangle_attention set to cuequivariance,
        # RuntimeError: ERROR: Full precision FP32 backward pass for triangle attention is not
        # implemented yet! Please set torch.backends.cuda.matmul.allow_tf32=True.
        "confidence_head": False,
        "sample_diffusion_training": True,
        "loss": True,
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(
            256
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "dynamic_chunk_size": True,
        "chunk_size_thresholds": {
            "1024": -1,  # -1 means no chunking (equivalent to None)
            "1536": 512,
            "2048": 256,
            "2560": 128,
        },
        "sample_diffusion_chunk_size": ValueMaybeNone(
            5
        ),  # should set to null for normal training and small dataset eval [for efficiency]
        "lddt_metrics_sparse_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "lddt_metrics_chunk_size": ValueMaybeNone(
            1
        ),  # only works if loss_metrics_sparse_enable, can set as default 1
    },
    "train_noise_sampler": {
        "p_mean": -1.2,
        "p_std": 1.5,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "inference_noise_scheduler": {
        "s_max": 160.0,
        "s_min": 4e-4,
        "rho": 7,
        "sigma_data": 16.0,  # NOTE: in EDM, this is 1.0
    },
    "sample_diffusion": {
        "gamma0": 0.8,
        "gamma_min": 1.0,
        "noise_scale_lambda": 1.003,
        "step_scale_eta": 1.5,
        "N_step": 200,
        "N_sample": 5,
        "N_step_mini_rollout": 20,
        "N_sample_mini_rollout": 1,
        "guidance": {
            # config for Training-Free Guidance (TFG).
            "enable": False,
            "log_last_step_energy": True,
            "rho": 0.0,
            "mu": 0.1,
            "mc": {
                "std": 0.0,
                "batch": 1,
            },
            "steps": {
                "tfg_outer": 1,
                "tfg_inner": 20,
                "projection_outer": 2,
                "projection_inner": 10,
            },
            "terms": {
                "VinaStericPotential": {
                    "interval": 1,
                    "weight": 0.1,
                    "buffer": 0.225,
                },
                "ExperimentalTorsionPotential": {
                    "interval": 1,
                    "weight": 0.0015,
                },
                "InterchainBondPotential": {
                    "interval": 1,
                    "weight": 0.15,
                    "buffer": 2.0,
                },
                "PairwiseDistancePotential": {
                    "interval": 1,
                    "weight": 0.5,
                    "enable_projection": True,
                    "bond_buffer": 0.00,
                    "angle_buffer": 0.00,
                    "clash_buffer": 0.00,
                },
                "ChiralAtomPotential": {
                    "interval": 1,
                    "weight": 0.0,
                    "enable_projection": True,
                    "buffer": 0.6155,
                },
                "StereoBondPotential": {
                    "interval": 1,
                    "weight": 0.25,
                    "buffer": 0.52360,
                },
                "PlanarImproperPotential": {
                    "interval": 1,
                    "weight": 0.12,
                },
                "LinearBondPotential": {
                    "interval": 1,
                    "weight": 0.25,
                    "buffer": 0.08726646259,
                },
            },
        },
    },
    "model": {
        "N_model_seed": 1,  # for inference
        "N_cycle": 4,
        "condition_embedding_drop_rate": 0.0,
        "confidence_embedding_drop_rate": 0.0,
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        "template_embedder": {
            "c": 64,
            "c_z": GlobalConfigValue("c_z"),
            "n_blocks": 0,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
        },
        "msa_module": {
            "c_m": 64,
            "c_z": GlobalConfigValue("c_z"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "msa_chunk_size": ValueMaybeNone(2048),
            "msa_max_size": 16384,
        },
        # Optional constraint embedder, only used when constraint is enabled.
        "constraint_embedder": {
            "pocket_embedder": {
                "enable": False,
                "c_s_input": 3,
                "c_z_input": 1,
            },
            "contact_embedder": {
                "enable": False,
                "c_z_input": 2,
            },
            "substructure_embedder": {
                "enable": False,
                "n_classes": 4,
                "architecture": "transformer",
                "hidden_dim": 128,
                "n_layers": 1,
            },
            "contact_atom_embedder": {
                "enable": False,
                "c_z_input": 2,
            },
            "c_constraint_z": GlobalConfigValue("c_z"),
            "c_constraint_s": GlobalConfigValue("c_s_inputs"),
            "c_constraint_atom_pair": GlobalConfigValue("c_atompair"),
            "initialize_method": "zero",  # zero, kaiming
        },
        "pairformer": {
            "n_blocks": GlobalConfigValue("n_blocks"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "n_heads": 16,
            "dropout": 0.25,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
        },
        "diffusion_module": {
            "use_fine_grained_checkpoint": True,
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_token": 768,
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "atom_encoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "transformer": {
                "n_blocks": 24,
                "n_heads": 16,
            },
            "atom_decoder": {
                "n_blocks": 3,
                "n_heads": 4,
            },
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        "confidence_head": {
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "max_atoms_per_token": GlobalConfigValue("max_atoms_per_token"),
            "pairformer_dropout": 0.0,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "distance_bin_start": 3.25,
            "distance_bin_end": 52.0,
            "distance_bin_step": 1.25,
            "stop_gradient": True,
        },
        "distogram_head": {
            "c_z": GlobalConfigValue("c_z"),
            "no_bins": GlobalConfigValue("no_bins"),
        },
    },
}
perm_configs = {
    # Chain and Atom Permutation
    "chain_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": False,
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "configs": {
            "use_center_rmsd": False,
            "find_gt_anchor_first": False,
            "accept_it_as_it_is": False,
            "enumerate_all_anchor_pairs": False,
            "selection_metric": "aligned_rmsd",
        },
    },
    "atom_permutation": {
        "train": {
            "mini_rollout": True,
            "diffusion_sample": False,
        },
        "test": {
            "diffusion_sample": True,
        },
        "permute_by_pocket": True,
        "global_align_wo_symmetric_atom": False,
    },
}
loss_configs = {
    "loss": {
        "diffusion_lddt_chunk_size": ValueMaybeNone(1),
        "diffusion_bond_chunk_size": ValueMaybeNone(1),
        "diffusion_chunk_size_outer": ValueMaybeNone(1),
        "diffusion_sparse_loss_enable": GlobalConfigValue("loss_metrics_sparse_enable"),
        "diffusion_lddt_loss_dense": True,  # only set true in initial training for training speed
        "resolution": {"min": 0.1, "max": 4.0},
        "weight": {
            "alpha_confidence": 1e-4,
            "alpha_pae": 0.0,  # or 1 in finetuning stage 3
            "alpha_except_pae": 1.0,
            "alpha_diffusion": 4.0,
            "alpha_distogram": 3e-2,
            "alpha_bond": 0.0,  # or 1 in finetuning stages
            "smooth_lddt": 1.0,  # or 0 in finetuning stages
        },
        "plddt": {
            "min_bin": 0,
            "max_bin": 1.0,
            "no_bins": 50,
            "normalize": True,
            "eps": 1e-6,
        },
        "pde": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "resolved": {
            "eps": 1e-6,
        },
        "pae": {
            "min_bin": 0,
            "max_bin": 32,
            "no_bins": 64,
            "eps": 1e-6,
        },
        "diffusion": {
            "mse": {
                "weight_mse": 1 / 3,
                "weight_dna": 5.0,
                "weight_rna": 5.0,
                "weight_ligand": 10.0,
                "eps": 1e-6,
            },
            "bond": {
                "eps": 1e-6,
            },
            "smooth_lddt": {
                "eps": 1e-6,
            },
        },
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": 1e-6,
        },
    },
    "metrics": {
        "lddt": {
            "eps": 1e-6,
        },
        "complex_ranker_keys": ListValue(["plddt", "gpde", "ranking_score"]),
        "chain_ranker_keys": ListValue(["chain_ptm", "chain_plddt"]),
        "interface_ranker_keys": ListValue(
            ["chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt"]
        ),
        "clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75},
    },
}

# Kineidos (P004).  Two blocks: `wp` is the ablation's arm, `kineidos` is where
# the GAGU data comes from.  Both are inert unless kineidos/train/trainer.py is
# the entry point -- upstream's runner never reads them.
kineidos_configs = {
    "wp": {
        # none | zero | random | pretrained (plan section 4).  `none` builds no
        # fusion at all, which is what makes it the baseline rather than a
        # fusion carrying zeros -- that is `zero`.
        "mode": "none",
        # Calibrated in plan section 2.10: 0.35 nm neighbourhood.  0.26 nm is
        # the ablation alternative; changing this changes what h means, so it
        # belongs in the config rather than in a default argument.
        "particle_radius_nm": 0.0778,
        # mode=pretrained only; Stage 1 has not produced one yet.
        "checkpoint": "",
        # Seeds WorldParticle's initialisation. -1 means "do not touch the
        # global RNG", which is what a run that wants the global seed to govern
        # everything should use.
        "seed": -1,
        # P010's oracle probe, and only on research/kineidos-v3-diag-oracle.
        #
        # "" is off and is the default, so every arm that does not ask for the
        # probe builds the same windows and runs the same code as p010-base.
        # "target" feeds the model the frame it is being asked to predict;
        # "decoy" feeds a random frame of the same trajectory at least 50 ns
        # away, which has the same statistics and no information (D2 item 6).
        # mode='oracle' requires one of the two -- there is no default, because
        # a default would decide for the reader which experiment ran.
        "oracle_source": "",
        # The projection is generated from this rather than stored, so this
        # number plus the sha256 in env.lock is what reproduces h exactly
        # (D2 item 7).
        "oracle_seed": 20261008,
        # D-a / section 1's nuance: whether the *target frame* gets AF3's random
        # rotation and translation before noise is added.  True is current
        # behaviour and the default.  The `oracle-noaug` arm sets it false, to
        # ask whether the mid-sigma band is blocked by the skip connection and
        # the network output being in different frames -- which is a real
        # obstacle there and not one at high sigma, where c_skip is small.
        "target_augmentation": True,
        # P010 section 8.  "linear" is the original probe, h = x_target @ P:
        # a rank-3 linear map, which wp_layernorm then strips of its scale --
        # and since the canonical frame's origin sits 5.17 nm from a molecule
        # 1.47 nm across, the lost radial direction is one fixed global axis,
        # so the network receives a two-dimensional shadow of the answer
        # (measured: the perpendicular components come back at R^2 = 0.97, the
        # parallel one at 0.11).  The gate's registered justification -- "the
        # target is fully recoverable" -- was computed for P alone and did not
        # account for the LayerNorm immediately after it.
        #
        # "fourier_anchor" is the replacement: each atom's distances to eight
        # anchor atoms, each distance expanded in 48 log-spaced sine/cosine
        # features (8 x 48 x 2 = 768).  Invariant by construction, so no frame
        # has to be inferred; and sin^2 + cos^2 = 1 makes every atom's
        # LayerNorm normaliser the same constant, so the layer cannot take the
        # signal.  Default stays "linear" -- changing it would silently change
        # what every oracle arm means.
        "oracle_encoding": "linear",
        # readout.md section 13.5.  Permute the finished encoding along the
        # atom axis, with a fixed seed.  Everything about the injected signal
        # survives -- its marginal distribution, its magnitude, its rank, how
        # wp_layernorm treats it -- except which atom each row describes.  It
        # is the null that `zero` and `random` cannot be, because both of those
        # have no per-atom structure at all while the open question is whether
        # per-atom structure suffices without being correct.
        "oracle_shuffle": False,
        # readout.md section 15.4.  "fourier_coord" is the frame-BEARING
        # counterpart of "fourier_anchor": same LayerNorm property, same
        # frequency floor, but it encodes the coordinates rather than the
        # distances, so a rotation changes it.  It exists to ask whether a
        # non-invariant encoder works once the frames are made to agree.
        "oracle_shared_rotation": False,
    },
    "kineidos": {
        "gagu_root": (
            "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/"
            "datasets/processed/gagu_internal_loop_v0_1"
        ),
        # Named, not globbed: a run's config should say which trajectories it
        # saw, so that a later run can be compared to it.
        "train_samples": ListValue(["gagu_100mM_K_agaguu_startI_r1"]),
        # Held out, scored with the training objective during training.  Named
        # rather than derived, for the same reason as train_samples.
        "held_out_samples": ListValue([""]),
        # How many held-out windows one evaluation round scores.  Fixed, drawn
        # once, iterated in order.
        "eval_windows": 64,
        # Seeds both the held-out window draw and -- reset at the start of every
        # round -- the noise and augmentation inside the scored forward, so that
        # round k of one arm scores bit-identical noise to round k of another.
        "eval_seed": 1234,
        "window_k": 8,
        # Nominal: an epoch is this many draws, not an enumeration (see
        # GAGUWindowDataset).
        "epoch_length": 10000,
        "num_workers": 0,
        # Where a requeued job looks for its predecessor.  Empty disables
        # resume.  Must be per-arm: resuming a `zero` checkpoint into a
        # `random` model would load every shared key and leave the bridge at
        # initialisation, giving a run that is neither arm.
        "resume_dir": "",
        # Kineidos (P010 D1): the per-sigma held-out reading.  All five default
        # to off or empty, so a run that does not ask for them behaves exactly
        # as it did under P009 -- the protocol is the defaults (P010 D11).
        #
        # `sigma_grid` makes every evaluation round write the per-sigma file
        # beside the per-window one.  It is not on by default because it costs
        # a second pass over the held-out set, and P009's four arms must keep
        # scoring what they have been scoring.
        "sigma_grid": False,
        # Noise samples per sigma per window.  Four is what section 4 sizes;
        # with eleven sigmas that is a 44-sample diffusion batch, close enough
        # to the 48 of training that memory behaves the same.
        "sigma_grid_noise": 4,
        # How often the grid runs, in steps.  0 means "every evaluation round",
        # which is the behaviour before this key existed and so the default.
        #
        # It exists because the grid turned out to cost as much as the training
        # it is measuring: 3.28 s a window, measured, times 256 windows is 14
        # minutes a round, and at eval_interval 250 over 2000 steps that is
        # eight rounds -- 1.9 GPU-hours against the arm's own 2.5.  The
        # held-out rounds must stay at 250, because the oracle gate reads the
        # first 500 steps and arm B reads the 500 after its release (section 6
        # items 2 and 3); the grid does not need that cadence, since what it
        # tracks moves on a scale of a thousand steps.  Rounded up to a
        # multiple of eval_interval, because it can only run where a round does.
        "sigma_grid_interval": 0,
        # How many of the held-out windows the grid scores.  0 means all of
        # them (kineidos.eval_windows), which is what section 4 fixes for
        # scoring a checkpoint.  A smaller number is a prefix of the same fixed
        # ordered set, so the rows still carry global window_ids and still pair
        # against a 256-window scoring on the windows they share.
        "sigma_grid_windows": 0,
        # The name written into every row's `arm` field, and into the file
        # name.  Empty means "derive it", which is right inside a training run
        # and wrong when scoring someone else's checkpoint.
        "sigma_grid_arm": "",
        # Where the jsonl goes.  Empty means run_dir/sigma_grid; the standalone
        # scorer points it at runs/p010/sigma_grid so that several arms' files
        # land together.
        "sigma_grid_out": "",
        # The trained checkpoint kineidos.score_sigma_grid scores.  Only that
        # entry point reads it; a training run leaves it empty.
        "score_checkpoint": "",
        # Resume across a code change.  False refuses, which is the default:
        # see KineidosTrainer.check_resume_provenance.  A config key rather
        # than an environment variable, so that using it is recorded in the
        # run's own env.lock instead of only in a shell history.
        "allow_resume_across_code": False,
    },
}

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    **perm_configs,
    **loss_configs,
    **kineidos_configs,
}
configs["finetune"] = finetune_optim_configs
