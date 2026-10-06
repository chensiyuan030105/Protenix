"""The P004 trainer: Protenix's loop, with the trunk frozen and WP injected.

A subclass rather than a fork of runner/train.py.  Four things differ, and
each of them is a place where getting it wrong is silent:

- **init_data** reads GAGU windows instead of mmCIF, through our collate.
- **after_model_built** attaches the WorldParticle bridge and freezes the
  trunk, in that order and at that moment -- see the hook's docstring in
  runner/train.py for why the moment matters.
- **try_load_checkpoint** uses kineidos/checkpoint.py, which requires the
  missing key set to *equal* the fusion's three, where upstream passes
  load_strict straight to load_state_dict.
- **train_step** records the injection-strength observables alongside the
  losses, so a run that is not actually injecting says so while it runs rather
  than when someone reads the checkpoint afterwards (plan section 5).

Evaluation is deliberately not the inference path; see `evaluate`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from kineidos import observables
from kineidos.checkpoint import FUSION_KEYS, load_checkpoint
from kineidos.train.batch import collate_fn_window
from kineidos.train.freezing import format_report, freeze_trunk
from kineidos.wp_bridge import TOKEN_DIM, WorldParticleBridge

from runner.train import AF3Trainer


def wp_token_dim_for(mode: str) -> int | None:
    """What the fusion's width has to be for a given arm.

    `none` is the only arm where the fusion does not exist at all: with
    wp_token_dim=None the AtomAttentionEncoder builds no wp_fusion and no
    wp_layernorm, so the `none` arm is Protenix with nothing added rather than
    Protenix plus a pathway that happens to carry zeros.  That distinction is
    what `zero` is for, and keeping them separate is the point of having both.
    """
    return None if mode == "none" else TOKEN_DIM


class KineidosTrainer(AF3Trainer):
    def __init__(self, configs: Any) -> None:
        # wp_token_dim cannot be set here.  DiffusionModule is built as
        # DiffusionModule(**configs.model.diffusion_module), so the key has to
        # be in that dict before parse_configs runs -- kineidos/train/main.py
        # puts it there.  All this can do is refuse a config where the arm and
        # the architecture disagree, which would otherwise train the baseline
        # under an ablation arm's name and look entirely normal.
        want = wp_token_dim_for(configs.wp.mode)
        got = configs.model.diffusion_module.get("wp_token_dim", None)
        if got != want:
            raise ValueError(
                f"wp.mode={configs.wp.mode!r} needs "
                f"model.diffusion_module.wp_token_dim={want!r}, but the parsed "
                f"config has {got!r}. Launch through kineidos.train.main, which "
                f"sets it before parse_configs; setting it afterwards is too "
                f"late to reach the constructor."
            )
        super().__init__(configs)

    # -------------------------------------------------------------- log

    def init_log(self) -> None:
        super().init_log()
        self.write_env_lock()

    def write_env_lock(self) -> None:
        """Pin what this run actually ran, inside the run's own directory.

        AGENTS.md asks for an env.lock per run.  It is written here rather than
        by the submission script because only the trainer knows the directory:
        init_basics appends a timestamp to run_name, so a lock written before
        launch lands beside the run rather than in it, and the next run to the
        same name would overwrite it.
        """
        import json

        from protenix.utils.distributed import DIST_WRAPPER

        if DIST_WRAPPER.rank != 0:
            return
        from kineidos.env_lock import collect, find_workspace_root

        record = collect(find_workspace_root())
        record["seeds"] = {"configs.seed": self.configs.seed}
        record["wp"] = {
            "mode": self.configs.wp.mode,
            "particle_radius_nm": self.configs.wp.particle_radius_nm,
            "checkpoint": self.configs.wp.checkpoint,
            "wp_token_dim": self.configs.model.diffusion_module.get(
                "wp_token_dim", None),
        }
        record["loss_switches"] = {
            "skip_confidence": self.configs.skip_confidence,
            "skip_distogram": self.configs.skip_distogram,
            "skip_mini_rollout": self.configs.skip_mini_rollout,
            "alpha_confidence": self.configs.loss.weight.alpha_confidence,
            "alpha_distogram": self.configs.loss.weight.alpha_distogram,
            "alpha_diffusion": self.configs.loss.weight.alpha_diffusion,
        }
        path = os.path.join(self.run_dir, "env.lock")
        with open(path, "w") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        self.print(f"env.lock -> {path}")
        for label, worktree in record["worktrees"].items():
            dirty = " [DIRTY]" if worktree["dirty"] else ""
            self.print(f"  {label:12s} {str(worktree['commit'])[:12]} "
                       f"({worktree['branch']}){dirty}")

    # ------------------------------------------------------------- model

    def after_model_built(self) -> None:
        """Attach the bridge, then freeze -- in that order.

        The bridge is assigned onto the model so that it is a submodule: DDP
        reduces its gradients, the optimizer sees its parameters, EMA tracks
        them and the checkpoint carries them.  Attaching it after freezing
        would leave its parameters with requires_grad=True by accident rather
        than by decision; attaching it before means freeze_trunk's "everything
        except these" covers it explicitly.
        """
        cfg = self.configs.wp
        bridge = WorldParticleBridge(
            cfg.mode,
            particle_radius_nm=cfg.particle_radius_nm,
            checkpoint=cfg.checkpoint or None,
            seed=cfg.seed if cfg.seed >= 0 else None,
        )
        self.raw_model.wp_bridge = bridge.to(self.device)

        report = freeze_trunk(self.raw_model,
                              trainable=("diffusion_module", "wp_bridge"))
        self.print(f"WorldParticle arm: {cfg.mode}")
        self.print(format_report(report))
        if report["parameter_free"]:
            self.print(f"  (empty, frozen vacuously: "
                       f"{', '.join(report['parameter_free'])})")
        self.freeze_report = report

    # -------------------------------------------------------------- data

    def init_data(self) -> None:
        from protenix.utils.distributed import DIST_WRAPPER

        from kineidos.data.gagu import GAGUProtenixAdapter
        from kineidos.data.windows import GAGUWindowDataset

        cfg = self.configs.kineidos
        root = Path(cfg.gagu_root)
        names = list(cfg.train_samples)
        if not names:
            raise ValueError(
                "kineidos.train_samples is empty; name the GAGU trajectories to "
                "train on rather than globbing, so a run's config says which "
                "data it saw"
            )
        self.print(f"Loading {len(names)} GAGU samples from {root}")
        samples = [GAGUProtenixAdapter(root / n).load() for n in names]

        dataset = GAGUWindowDataset(
            samples, k=cfg.window_k, length=cfg.epoch_length,
            seed=self.configs.seed, canonicalize=True,
        )
        # The dataset's index *is* the draw (see its docstring), so shuffling
        # would only permute which seeded window comes when.  What does matter
        # on more than one device is that the ranks draw *different* indices:
        # without a sampler every rank iterates 0..length-1 and they all train
        # on the same windows, which costs N times the compute for one device's
        # worth of data and looks exactly like a working multi-GPU run.
        sampler = None
        if DIST_WRAPPER.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank, shuffle=False, drop_last=False,
            )
        self.train_dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,          # the loss calls feat_dict["resolution"].item()
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_window,
            drop_last=False,
        )
        self.test_dls = {}

    # -------------------------------------------------------- checkpoint

    def try_load_checkpoint(self) -> None:
        """Load the pretrained trunk, allowing exactly the fusion to be absent.

        Upstream hands `load_strict` to load_state_dict.  Either value is wrong
        here: True raises because the fusion's three parameters are new, and
        False would equally tolerate a renamed module quietly not loading.
        kineidos/checkpoint.py asks for equality instead.
        """
        path = self.configs.load_checkpoint_path
        if not path:
            raise ValueError(
                "load_checkpoint_path is empty. The ablation's premise is a "
                "frozen *pretrained* trunk (plan section 1); training from a "
                "random trunk would answer a different question."
            )
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model"]
        if next(iter(state)).startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}

        expect = set() if self.configs.wp.mode == "none" else set(FUSION_KEYS)
        # The bridge's own weights are never in a Protenix checkpoint: `random`
        # means randomly initialised by definition, and `pretrained` loads them
        # from WorldParticle's own file inside the bridge's constructor.
        expect |= {k for k in self.raw_model.state_dict() if k.startswith("wp_bridge.")}

        info = load_checkpoint(self.raw_model, state, expect_new=expect)
        self.print(
            f"Loaded {info['tensors_loaded']} tensors "
            f"({info['parameters'] / 1e6:.2f}M); left at initialisation: "
            f"{len(info['left_at_initialisation'])}"
        )

    # -------------------------------------------------------------- loop

    def train_step(self, batch: dict[str, Any]) -> None:
        super().train_step(batch)
        stats = observables.read(self.raw_model)
        if not stats:
            return
        for key, value in stats.items():
            self.train_metric_wrapper.add(f"wp/{key}", torch.tensor(float(value)),
                                          namespace="train")
        for message in observables.warnings(stats, mode=self.configs.wp.mode):
            self.print(f"[injection] step {self.step}: {message}")

    def evaluate(self, mode: str = "eval") -> None:
        """Not implemented -- skipped loudly, not silently, and not fatally.

        Upstream's evaluate runs main_inference_loop and then reads
        pred_dict["summary_confidence"], which the confidence head produces and
        this configuration does not run.  So held-out evaluation needs a loop
        of its own, scoring the *training* objective on unseen windows.  Two
        decisions have to be made first and neither should be improvised here:
        whether to accept dropout noise in the metric (Protenix.forward asserts
        self.training for mode="train", so scoring the training objective with
        dropout off means relaxing that assert in this fork), and which GAGU
        trajectories are held out.

        This raised NotImplementedError at first, which was wrong: run() calls
        evaluate() whenever `is_last_step`, regardless of eval_interval, so
        raising threw away every completed run at its final step.  It prints
        instead -- once per run with eval_interval=-1 -- because a silent `pass`
        would let someone read the absence of eval metrics as "they were fine".

        Until it exists, the instruments are section 5's observables and the
        training losses.
        """
        self.print(
            f"[eval] skipped at step {self.step}: held-out evaluation is not "
            f"implemented for P004. Upstream's evaluate needs the confidence "
            f"head that skip_confidence removes; this run's instruments are the "
            f"training losses and the train/wp/ observables."
        )
