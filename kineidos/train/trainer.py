"""The Kineidos trainer: Protenix's loop, with the trunk frozen and WP injected.

Built for P004; carried into P009 unchanged except for `write_heldout_rows`.

A subclass rather than a fork of runner/train.py.  Five things differ, and
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
  than when someone reads the checkpoint afterwards (P004 section 5).
- **write_heldout_rows** keeps the per-window held-out losses, which the mean
  alone cannot substitute for: on this data the difficulty is set by dt, so a
  difference between arms lives inside a dt bin (P009 section 4, item 6).

Evaluation is deliberately not the inference path; see `evaluate`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from protenix.utils.distributed import DIST_WRAPPER

from kineidos import atomic, observables
from kineidos.seeding import set_all_seeds
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
        # Atomic, for the same reason as everything else this project writes:
        # `background` preempts with no grace period.  A half-written env.lock
        # is worse than an absent one -- it is invalid JSON, so every reader
        # that parses it (read_heldout's oracle refusal, the resume provenance
        # gate) falls through its JSONDecodeError handler and reports the run
        # as unverifiable, which is indistinguishable from a run that was
        # never locked.  See kineidos/atomic.py.
        path = Path(self.run_dir) / "env.lock"
        atomic.write_json(path, record)
        self.print(f"env.lock -> {path}")
        for label, worktree in record["worktrees"].items():
            dirty = " [DIRTY]" if worktree["dirty"] else ""
            # The tag as well as the hash.  P010 compares nine arms across
            # three branches and the tag is what says an arm is comparable
            # (p010-base / p010-oracle / p010-gamma); a hash in a log is not
            # something a reader can place.
            tags = worktree.get("tags") or []
            tag = f" @{','.join(tags)}" if tags else ""
            self.print(f"  {label:12s} {str(worktree['commit'])[:12]} "
                       f"({worktree['branch']}){tag}{dirty}")

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
        names = [n for n in cfg.train_samples if n]
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
        # ListValue's default is [""]; an empty name is "no held-out set", not a
        # trajectory called "".
        held = [n for n in cfg.held_out_samples if n]
        self.eval_windows = None
        if held:
            self.print(f"Held out {len(held)} GAGU samples")
            eval_samples = [GAGUProtenixAdapter(root / n).load() for n in held]
            # A fixed set, iterated in order, with its own seed.  Fixed is the
            # whole point: the number only means something if every arm, and
            # every evaluation round, scores the same windows.
            eval_ds = GAGUWindowDataset(
                eval_samples, k=cfg.window_k, length=cfg.eval_windows,
                seed=cfg.eval_seed, canonicalize=True,
            )
            self.eval_windows = [eval_ds[i] for i in range(len(eval_ds))]

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

    def latest_link(self) -> Optional[Path]:
        """The stable path a requeued job looks for, or None if not configured.

        A requeued job gets a fresh run directory -- init_basics appends a
        timestamp -- so it cannot find its predecessor's checkpoints by
        construction.  This is the fixed point that survives the rename.
        """
        if not self.configs.kineidos.resume_dir:
            return None
        return Path(self.configs.kineidos.resume_dir) / "latest.pt"

    def save_checkpoint(self, ema_suffix: str = "") -> None:
        """Save as upstream does, then point `latest.pt` at it.

        The arm goes into the file as well.  Resuming a `zero` checkpoint into a
        `random` model would load cleanly for every shared key and leave the
        bridge at its initialisation, producing a run that is neither arm and
        says nothing about it -- so it is refused on the way back in.
        """
        super().save_checkpoint(ema_suffix=ema_suffix)
        link = self.latest_link()
        if link is None or DIST_WRAPPER.rank != 0 or ema_suffix:
            return
        saved = Path(self.checkpoint_dir) / f"{self.step}{ema_suffix}.pt"
        if not saved.exists():
            return
        link.parent.mkdir(parents=True, exist_ok=True)
        # Replace the symlink atomically: a requeue can land between the unlink
        # and the re-link, and a resume that finds no latest.pt silently starts
        # the run over.  os.replace on a symlink is one rename syscall.
        tmp = link.with_name(link.name + ".tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(saved.resolve())
        os.replace(tmp, link)
        # The arm goes beside the checkpoint rather than inside it: rewriting a
        # 3 GB file to add one string is not worth it, and a sidecar can be read
        # without torch.load.
        (link.parent / "latest.arm").write_text(self.configs.wp.mode + "\n")
        self.write_resume_provenance(link.parent)
        self.print(f"latest.pt -> {saved}")

    def write_resume_provenance(self, where: Path) -> None:
        """Which code wrote this checkpoint, beside the checkpoint.

        The input-fingerprint gate the P006 session arrived at from four
        separate silent-corruption incidents, applied to the one place P010
        reuses an upstream artefact: `latest.pt`.
        
        The shape of the hazard here.  The diagnostic arms are 2000 steps on
        `background`, which preempts with no grace period, and resume() reloads
        latest.pt.  Development happens in the same worktree -- 22 commits to
        this tree in one session.  So an arm can be preempted, the tree can
        move, and the requeued arm resumes an optimizer state written by code
        that no longer exists, while its env.lock records the *new* commit.
        Every existing gate is green: latest.arm matches, the state_dict loads
        with nothing missing, the loss curve continues smoothly.  Nothing says
        the run is now two code versions stitched together.
        """
        import json

        from kineidos.env_lock import code_trees, find_workspace_root

        try:
            trees = code_trees(find_workspace_root())
        except Exception as exc:                            # noqa: BLE001
            # Never let provenance bookkeeping lose a checkpoint.  A missing
            # sidecar makes the gate say "cannot verify", which is the right
            # answer and is not silence.
            self.print(f"[resume] could not record provenance: {exc}")
            return
        record = {
            "wp_mode": self.configs.wp.mode,
            "step": self.step,
            "trees": {name: {"commit": t.get("commit"), "tags": t.get("tags"),
                             "dirty": t.get("dirty")}
                      for name, t in trees.items()},
        }
        (where / "latest.provenance").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n")

    def resume(self, path: Path) -> bool:
        """Continue a run that was interrupted.  True if it actually resumed.

        Needed because the only partition with working GPUs and free capacity is
        `background`, which sits at PriorityTier=5 under PreemptMode=REQUEUE
        with no grace period: a job there is killed and restarted from scratch
        whenever a tier-10 job wants the node.  Without this, "train for a long
        time" and "use background" are mutually exclusive.
        """
        arm_file = path.parent / "latest.arm"
        if arm_file.exists():
            arm = arm_file.read_text().strip()
            if arm != self.configs.wp.mode:
                raise ValueError(
                    f"{path} was written by the {arm!r} arm and this run is "
                    f"{self.configs.wp.mode!r}. Resuming across arms would load "
                    f"every shared key and leave the bridge at initialisation, "
                    f"giving a run that is neither arm. Use a resume_dir per arm."
                )
        self.check_resume_provenance(path.parent)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model"]
        if next(iter(state)).startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
        # Nothing may be missing here, unlike the pretrained load: this file was
        # written by this architecture, so a gap means the architecture changed
        # under the run and the comparison is void.
        load_checkpoint(self.raw_model, state, expect_new=())
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"] + 1
        self.start_step = self.step
        self.global_step = self.step * self.iters_to_accumulate
        self.print(f"Resumed {self.configs.wp.mode!r} from {path} at step {self.step}")
        return True

    def check_resume_provenance(self, where: Path) -> None:
        """Refuse to resume a checkpoint that different code wrote.

        Not a warning.  A warning in a slurm log that nobody reads until the
        run is over is the same as no check: the arm would finish, look
        normal, and be two code versions stitched at whatever step the
        preemption happened.  There is no way to tell afterwards which half of
        the curve came from which.

        `kineidos.allow_resume_across_code` is the escape hatch, and it is a
        config key rather than an environment variable so that using it is
        recorded in the run's own env.lock.

        A dirty tree on either side also refuses: "same commit" says nothing
        when there are uncommitted edits, which is the normal state of a tree
        somebody is working in.
        """
        import json

        from kineidos.env_lock import code_trees, find_workspace_root

        sidecar = where / "latest.provenance"
        if not sidecar.is_file():
            # Written by a run from before this check existed.  Say so rather
            # than passing silently -- "cannot verify" is a different state
            # from "verified".
            self.print(
                f"[resume] {sidecar} is absent, so which code wrote this "
                f"checkpoint cannot be verified. Written before this check "
                f"existed, or removed."
            )
            return
        try:
            was = json.loads(sidecar.read_text())
            now_trees = code_trees(find_workspace_root())
        except Exception as exc:                            # noqa: BLE001
            self.print(f"[resume] could not check provenance: {exc}")
            return

        problems: list[str] = []
        for name, then in (was.get("trees") or {}).items():
            now = now_trees.get(name)
            if now is None:
                problems.append(f"{name}: was on PYTHONPATH, now is not")
                continue
            if then.get("commit") != now.get("commit"):
                problems.append(
                    f"{name}: checkpoint written at "
                    f"{str(then.get('commit'))[:12]}"
                    f"{' @' + ','.join(then.get('tags') or []) if then.get('tags') else ''}"
                    f", now {str(now.get('commit'))[:12]}"
                    f"{' @' + ','.join(now.get('tags') or []) if now.get('tags') else ''}")
            if then.get("dirty") or now.get("dirty"):
                problems.append(
                    f"{name}: dirty tree ({'then' if then.get('dirty') else ''}"
                    f"{' and ' if then.get('dirty') and now.get('dirty') else ''}"
                    f"{'now' if now.get('dirty') else ''}), so the commit does "
                    f"not identify the code")
        if not problems:
            self.print("[resume] provenance checks out: same code wrote this "
                       "checkpoint")
            return
        if self.configs.kineidos.allow_resume_across_code:
            self.print("[resume] CODE CHANGED UNDER THIS RUN, and "
                       "kineidos.allow_resume_across_code says to continue:")
            for p_ in problems:
                self.print(f"  {p_}")
            self.print("  this arm is two code versions stitched at the step "
                       "the preemption happened; nothing downstream can tell "
                       "which half is which")
            return
        raise ValueError(
            "refusing to resume: the code changed since this checkpoint was "
            "written.\n  " + "\n  ".join(problems) + "\n"
            "The arms are 2000 steps on a preemptible partition and "
            "development happens in the same worktree, so this is the normal "
            "way a run becomes two code versions stitched together -- with "
            "every other gate green: latest.arm matches, the state_dict loads "
            "with nothing missing, the loss curve continues smoothly. Start "
            "the arm again on one commit, or set "
            "--kineidos.allow_resume_across_code true to say in the run's own "
            "env.lock that you meant it."
        )

    def try_load_checkpoint(self) -> None:
        """Resume if there is a run to resume, otherwise load the pretrained trunk.

        For the pretrained load, upstream hands `load_strict` to
        load_state_dict.  Either value is wrong here: True raises because the
        fusion's three parameters are new, and False would equally tolerate a
        renamed module quietly not loading.  kineidos/checkpoint.py asks for
        equality instead.
        """
        link = self.latest_link()
        if link is not None and (link.exists() or link.is_symlink()):
            self.resume(link)
            return

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

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """The training objective on held-out windows, with dropout off.

        Not upstream's evaluate: that goes through main_inference_loop and reads
        pred_dict["summary_confidence"], which the confidence head produces and
        this configuration does not run.

        Three choices make the number comparable across arms, and each of them
        matters more than it looks:

        **Dropout off.**  Not for a cleaner number -- for pairing.  The noise
        level and the augmentation rotation are drawn inside
        sample_diffusion_training from the global RNG.  With dropout active the
        arms consume different amounts of that stream, the draws stop matching,
        and the comparison quietly becomes unpaired.  With dropout off and the
        same seed, every arm sees bit-identical noise and rotations, so the only
        thing that differs between arms is the thing under test.  This is why
        Protenix.forward gained eval_training_objective.

        **A fixed window set.**  Drawn once in init_data from the held-out
        trajectories with its own seed, and scored in the same order every time.

        **The seed reset per round.**  Reset before the loop so that round k
        scores the same noise as round k of any other arm, which makes the
        curves comparable point by point rather than only in trend.

        The loss is Protenix's own, so the rigid alignment inside MSELoss and
        SmoothLDDTLoss is the one the training objective uses -- writing a second
        alignment here is how a metric ends up measuring something else (8).
        """
        if not self.eval_windows:
            self.print(
                f"[eval] step {self.step}: no held-out windows configured "
                f"(kineidos.held_out_samples is empty), nothing to score"
            )
            return

        from protenix.utils.torch_utils import to_device

        from kineidos.train.batch import collate_window

        # raw_model, not the DDP wrapper.  Scoring needs no gradient sync, and
        # going through DDP would entangle an eval forward with static_graph's
        # one-shot graph construction for no benefit.
        was_training = self.raw_model.training
        self.raw_model.eval()
        # Reset per round, not once per run: round k must score the same noise
        # as round k of every other arm.
        set_all_seeds(self.configs.kineidos.eval_seed)

        # Shard the fixed set across ranks and sum afterwards, rather than every
        # rank scoring all of it.  The mean is the same either way -- the set and
        # its order are fixed -- but the duplicated version costs world_size
        # times the work and makes rank 0 a straggler the others wait on.
        shard = self.eval_windows[DIST_WRAPPER.rank::DIST_WRAPPER.world_size]
        totals: dict[str, float] = {}
        n = 0
        rows: list[dict[str, Any]] = []
        for i, window in enumerate(shard):
            batch = to_device(collate_window(window), self.device)
            pred, label, _ = self.raw_model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode="train",
                current_step=self.step,
                symmetric_permutation=self.symmetric_permutation,
                eval_training_objective=True,
                # Pin the recycling depth.  Upstream draws it from
                # RandomState(current_step), which made every round use a
                # different trunk depth -- rounds at steps 499/999/1499/1999
                # drew 8/1/2/4 -- so the curve measured a depth lottery as much
                # as it measured learning.  Full depth is what inference uses
                # (main_inference_loop passes self.N_cycle), so it is the depth
                # the number should describe.
                n_cycle=self.configs.model.N_cycle,
            )
            total, loss_dict = self.loss(
                feat_dict=batch["input_feature_dict"],
                pred_dict=pred,
                label_dict=label,
                mode="train",
            )
            per_window = {k: float(v) for k, v in loss_dict.items()
                          if "loss" in k}
            for key, value in per_window.items():
                totals[key] = totals.get(key, 0.0) + value
            rows.append({
                # The index into the fixed set, not the position in this rank's
                # shard.  The shard is eval_windows[rank::world_size], so rank
                # r's i-th window is global r + i*world_size; numbering by i
                # alone would give three different windows the same id and the
                # per-dt bins would be built from a scrambled set.
                "window_id": DIST_WRAPPER.rank + i * DIST_WRAPPER.world_size,
                "sample_id": window.sample_id,
                "target_frame": int(window.target_frame),
                "stride": int(window.stride),
                "delta_t_ns": float(window.delta_t_ns),
                **per_window,
                # Last, so it is the total that backward would have used rather
                # than whatever the loss happens to call "loss" in its dict.
                "loss": float(total),
            })
            n += 1

        if DIST_WRAPPER.world_size > 1:
            import torch.distributed as dist

            # Sum the per-rank partial sums and counts, so the reported mean is
            # over the whole fixed set and does not depend on world_size.
            keys = sorted(totals)
            packed = torch.tensor([totals[k] for k in keys] + [float(n)],
                                  device=self.device, dtype=torch.float64)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {k: packed[i].item() for i, k in enumerate(keys)}
            n = int(packed[-1].item())

        # rows survives the reduction above -- it reduces `totals`, not the
        # per-window values -- but those values exist nowhere else, so they go
        # to disk before anything else can be reported.
        self.write_heldout_rows(rows)

        means = {k: v / n for k, v in totals.items()}
        for key, value in means.items():
            self.train_metric_wrapper.add(f"heldout/{key}",
                                          torch.tensor(value), namespace="train")
        # Injection strength on unseen windows.  High on training windows and
        # low here would mean WorldParticle is memorising rather than
        # generalising, which no training-side observable can tell you.
        stats = observables.read(self.raw_model)
        for key, value in stats.items():
            self.train_metric_wrapper.add(f"heldout_wp/{key}",
                                          torch.tensor(float(value)),
                                          namespace="train")
        headline = ", ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())
                             if not k.startswith("weighted_"))
        # N_cycle in the line, because it changes the numbers and a reader
        # comparing two logs has no other way to know it was held fixed.
        self.print(f"[eval] step {self.step} over {n} held-out windows "
                   f"(N_cycle={self.configs.model.N_cycle}): {headline}")

        # The per-sigma reading, on the same windows in the same round (P010
        # D1).  Off by default, and after the per-window rows are already on
        # disk: this is the new and less proven of the two, and a failure in it
        # must not cost the round's held-out numbers.
        if self.configs.kineidos.sigma_grid and self.sigma_grid_due():
            from kineidos import score_sigma_grid

            out = self.configs.kineidos.sigma_grid_out
            score_sigma_grid.score(
                self,
                step=self.step,
                arm=self.configs.kineidos.sigma_grid_arm
                or self.configs.run_name,
                out_dir=Path(out) if out else Path(self.run_dir) / "sigma_grid",
                n_noise=int(self.configs.kineidos.sigma_grid_noise),
                n_windows=int(self.configs.kineidos.sigma_grid_windows),
            )

        if was_training:
            self.raw_model.train()

    def sigma_grid_due(self) -> bool:
        """Is this one of the rounds the per-sigma grid runs on.

        0 means every round, which is what the key defaulted to before it
        existed.  Otherwise the grid runs on rounds whose step is at or past
        the next multiple of the interval -- rounded up from eval_interval,
        since the grid can only run where a round does, and the last step is
        always included because run() evaluates there whatever the interval
        says and that round is the arm's final state.
        """
        every = int(self.configs.kineidos.sigma_grid_interval)
        if every <= 0:
            return True
        if self.step >= self.configs.max_steps - 1:
            return True
        interval = max(int(self.configs.eval_interval), 1)
        # The round index, so the test does not depend on whether the step
        # numbering is 0- or 1-based at this point in the loop.
        rounds_per_grid = max(int(round(every / interval)), 1)
        return ((self.step + 1) // interval) % rounds_per_grid == 0

    def write_heldout_rows(self, rows: list[dict[str, Any]]) -> None:
        """One line per held-out window, per rank, per evaluation round.

        The mean cannot answer the question P009 asks.  Difficulty on this data
        is almost entirely set by dt -- the history can remove 27.8% of the
        no-history error at 0.1 ns and 5.2% at 1.0 ns (P009 section 1.1) -- so
        an arm-to-arm difference lives inside a dt bin and is diluted by the
        windows where no history could have helped.  Averaged over the set, a
        real effect in the short-dt windows and no effect at all are the same
        number.  P004 section 2.17 named this gap; filling it costs the lines
        below, and the all_reduce after the scoring loop is where these values
        used to disappear.

        Per rank, not gathered.  all_gather_object on 256 rows every 500 steps
        buys nothing, and separate files cannot interleave.  The reader joins
        them: the window_id is global, so `cat step_<N>.rank*.jsonl` is the
        whole set exactly once.
        """
        import json

        out = Path(self.run_dir) / "heldout"
        # exist_ok because init_basics creates run_dir on rank 0 only, and every
        # rank writes here.
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"step_{self.step}.rank{DIST_WRAPPER.rank}.jsonl"
        # Atomic.  These rows are what P009 section 6.2's verdict is computed
        # from, and read_heldout.load_rows reads whatever lines are present
        # with no completeness check -- so a round cut short by a preemption
        # would read as a complete round over fewer windows.  Today that is
        # caught only because read_heldout compares window sets across arms
        # and raises when they differ, which needs at least one arm to have
        # survived the same preemption.  Writing aside and renaming removes
        # the condition rather than relying on a sibling.
        with atomic.atomic_open(path) as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
