r"""
scripts/training/train.py

StructureLlama section-level pair training -- similarity-bucket conditioning.

Each training sample is one (context_section, sim_bucket, target_section) pair.
Loss is masked to zero on the context prefix; only target section tokens contribute.

condition_mode:
    m1 -- correct sim_bucket token (sim:high/mid/low) + PROTO tokens at bar boundaries
    m2 -- no sim_bucket token, no PROTO tokens (unconditional baseline)
    m3 -- randomly wrong sim_bucket token + PROTO tokens (ablation)

Training data: pairs.csv (built by build_section_pairs.py).
Bucket-balanced sampling is handled by the DataLoader (WeightedRandomSampler).

Example usage:
    SEC=<SECTIONS_DIR>

    # M1 -- full conditioning
    python scripts/training/train.py \
        --config_path   scripts/configs/model_config_structure_llama_839M.json \
        --sections_dir  $SEC \
        --pairs_csv     $SEC/pairs.csv \
        --checkpoint    <CHECKPOINT_PATH>/moonbeam_839M.pt \
        --condition_mode m1 --use_lora \
        --context_length 1024 --batch_size 2 --num_epochs 200 --lr 3e-5 \
        --save_dir runs/m1

    # M2 -- unconditional baseline (same checkpoint, no sim_bucket token)
    # change: --condition_mode m2  --save_dir runs/m2

    # M3 -- wrong bucket labels (ablation)
    # change: --condition_mode m3  --save_dir runs/m3
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.config            import load_config
from scripts.model.structure_llama   import StructureLlama
from scripts.model.tokenizer         import MusicTokenizer
from scripts.utils.checkpoint_loader import load_moonbeam_checkpoint
from scripts.training.dataset        import SECTION_TOKEN_MAP, build_pair_dataloader


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune StructureLlama (section-level pairs)")

    # Model
    p.add_argument("--config_path", required=True,
                   help="Path to model config JSON")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to Moonbeam pretrained checkpoint (.pt). Skipped if omitted.")

    # Data
    p.add_argument("--sections_dir", required=True,
                   help="Directory containing per-section .npy files (output of "
                        "build_section_pairs.py)")
    p.add_argument("--pairs_csv",    required=True,
                   help="Path to pairs.csv (output of build_section_pairs.py)")
    p.add_argument("--condition_mode", default="m1", choices=["m1", "m2", "m3"],
                   help="m1=correct sim_bucket+PROTO, m2=no conditioning, m3=wrong bucket")

    # Training
    p.add_argument("--context_length", type=int,   default=1024)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--num_epochs",     type=int,   default=50)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="Cap training pairs for smoke tests (0 = use all).")
    p.add_argument("--constant_lr", action="store_true",
                   help="Disable cosine LR decay -- hold LR constant. Use for overfit tests.")

    # LoRA (Run 9+)
    p.add_argument("--use_lora", action="store_true",
                   help="Freeze backbone and inject LoRA adapters on Q/V projections. "
                        "Trainable: supplementary_embedding, proto_proj (if used), LoRA A/B.")
    p.add_argument("--lora_rank",  type=int, default=8,
                   help="LoRA rank r (default: 8).")
    p.add_argument("--lora_alpha", type=int, default=16,
                   help="LoRA alpha scaling (default: 16; effective scale = alpha/r).")

    # Ablation options
    p.add_argument("--no_proto", action="store_true",
                   help="Disable PROTO tokens. Sequence becomes [SOS][ctx][SOC][bucket][tgt][EOC]. "
                        "Forces model to use bucket label + attention to context.")
    p.add_argument("--min_buckets", type=int, default=0,
                   help="Filter to contexts with >= N distinct sim_bucket values. "
                        "0 = no filter (default).")

    # Checkpointing
    p.add_argument("--save_dir",   default=None,
                   help="Directory to save epoch checkpoints. Skipped if omitted.")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save a checkpoint every N epochs.")
    p.add_argument("--resume",     default=None,
                   help="Path to a saved epoch checkpoint to resume training from.")
    p.add_argument("--patience",   type=int, default=0,
                   help="Early stopping: stop if val_loss does not improve for this many "
                        "consecutive epochs. 0 = disabled (run all --num_epochs).")

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Training / validation loops
# ──────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    grad_clip: float = 1.0,
    is_train:  bool  = True,
) -> tuple[float, float, float]:
    """Run one epoch. Returns (mean_loss, perplexity, mean_ce_gap).

    mean_ce_gap = mean(CE_wrong - CE_real) across batches.
    CE_wrong: loss when context tokens are replaced by the next sample's context
              (batch rolled by 1 along batch dim).
    A growing CE_gap during training indicates the model is learning to use context.
    For batch_size=1, CE_gap is always 0 (same-sample roll is a no-op).
    """
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss   = 0.0
    total_ce_gap = 0.0
    total_steps  = 0

    with context:
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            condition_mask = batch["condition_mask"].to(device)
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)

            out_real = model(
                input_ids,
                labels=labels,
                condition_mask=condition_mask,
                additional_token_map=SECTION_TOKEN_MAP,
                sample_weights=sample_weights,
            )
            ce_real = out_real.loss

            # CE_wrong diagnostic: swap context prefixes within the batch
            with torch.no_grad():
                ctx_mask        = condition_mask.unsqueeze(-1).expand_as(input_ids)
                input_ids_wrong = torch.where(ctx_mask, input_ids.roll(1, dims=0), input_ids)
            out_wrong = model(
                input_ids_wrong,
                labels=labels,
                condition_mask=condition_mask,
                additional_token_map=SECTION_TOKEN_MAP,
            )
            ce_gap = (out_wrong.loss.detach() - ce_real.detach()).item()
            total_ce_gap += ce_gap

            if is_train:
                optimizer.zero_grad()
                ce_real.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss  += ce_real.item()
            total_steps += 1

    mean_loss   = total_loss   / max(total_steps, 1)
    mean_ce_gap = total_ce_gap / max(total_steps, 1)
    return mean_loss, math.exp(min(mean_loss, 20)), mean_ce_gap


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────
    use_proto = not args.no_proto
    config = load_config(args.config_path)
    model  = StructureLlama(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if args.use_lora:
        print(f"LoRA: r={args.lora_rank}  alpha={args.lora_alpha}")

    if args.checkpoint:
        load_moonbeam_checkpoint(model, args.checkpoint, map_location="cpu", verbose=True)

    # Run 9: apply LoRA AFTER loading pretrained checkpoint so frozen weights are correct
    if args.use_lora:
        model.apply_lora(r=args.lora_rank, alpha=args.lora_alpha,
                         use_proto=use_proto)

    # Resume from a saved training checkpoint (overrides --checkpoint weights)
    resume_ckpt = None
    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {resume_ckpt['epoch']}), "
              f"starting at epoch {start_epoch}")

    model = model.to(device)

    # ── Tokenizer (vocab sizes only -- no token dict needed for pair training) ──
    tokenizer = MusicTokenizer.from_config(config)

    # ── Data ─────────────────────────────────────────────────
    print(f"Condition mode: {args.condition_mode.upper()}")
    print(f"PROTO tokens: {'enabled' if use_proto else 'DISABLED'}")
    if args.min_buckets > 0:
        print(f"min_buckets filter: {args.min_buckets}")
    train_loader = build_pair_dataloader(
        sections_dir   = args.sections_dir,
        pairs_csv      = args.pairs_csv,
        tokenizer      = tokenizer,
        partition      = "train",
        condition_mode = args.condition_mode,
        max_seq_len    = args.context_length,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        max_samples    = args.max_train_samples,
        min_buckets    = args.min_buckets,
        use_proto      = use_proto,
    )
    val_loader = build_pair_dataloader(
        sections_dir   = args.sections_dir,
        pairs_csv      = args.pairs_csv,
        tokenizer      = tokenizer,
        partition      = "val",
        condition_mode = args.condition_mode,
        max_seq_len    = args.context_length,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        min_buckets    = args.min_buckets,
        use_proto      = use_proto,
    )
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch:   {len(val_loader)}")
    if len(val_loader) == 0:
        raise RuntimeError(
            "Validation DataLoader is empty -- no rows with split='val' in pairs.csv."
        )

    # ── Optimiser ────────────────────────────────────────────
    # Only optimise parameters with requires_grad=True.
    # When --use_lora, this excludes the frozen backbone weights.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = args.num_epochs,
        eta_min = args.lr if args.constant_lr else args.lr * 0.05,
    )

    # ── Save dir ─────────────────────────────────────────────
    if args.save_dir:
        pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────
    best_loss      = float("inf")
    no_improve     = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        t0 = time.time()

        train_loss, train_ppl, train_ce_gap = run_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, is_train=True,
        )
        scheduler.step()

        val_loss, val_ppl, val_ce_gap = run_epoch(
            model, val_loader, None, device,
            grad_clip=args.grad_clip, is_train=False,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_ppl={train_ppl:.1f}  val_ppl={val_ppl:.1f}  "
            f"val_ce_gap={val_ce_gap:+.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if args.save_dir:
            ckpt_payload = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss":           train_loss,
                "val_loss":             val_loss,
                "condition_mode":       args.condition_mode,
            }
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = pathlib.Path(args.save_dir) / f"epoch_{epoch:03d}.pt"
                torch.save(ckpt_payload, str(ckpt_path))
                print(f"  Saved checkpoint -> {ckpt_path}")
                # Delete the previous epoch checkpoint (keep only the latest + best.pt)
                for old in pathlib.Path(args.save_dir).glob("epoch_*.pt"):
                    if old != ckpt_path:
                        old.unlink()
            if val_loss < best_loss:
                best_loss  = val_loss
                no_improve = 0
                best_path  = pathlib.Path(args.save_dir) / "best.pt"
                torch.save(ckpt_payload, str(best_path))
                print(f"  New best val_loss {best_loss:.4f} -> {best_path}")
            else:
                no_improve += 1

        elif val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stop: val_loss did not improve for {args.patience} epochs.")
            break

    print("\nTraining complete.")
    if args.save_dir:
        print(f"Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
