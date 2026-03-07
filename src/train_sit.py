"""
SiT (Scalable Interpolant Transformers) training script.

SiT uses the same DiT architecture but with a continuous-time interpolant
framework instead of discrete-time DDPM. Key choices:
- Path type: linear (flow matching), gvp (trigonometric), vp
- Prediction: velocity (simplest), score, noise
- Loss weighting: none, velocity, likelihood

Reference: https://arxiv.org/abs/2401.08740
"""

import argparse
import copy
from datetime import datetime
from pathlib import Path
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

try:
    import wandb
except ImportError:
    pass

from nanodiffusion.models.factory import create_model, choices
from nanodiffusion.optimizers.lr_schedule import get_cosine_schedule_with_warmup
from nanodiffusion.datasets import load_data
from nanodiffusion.sit.transport import Transport, euler_ode_sample
from nanodiffusion.sit.interpolant import create_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="SiT training for images")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--net", type=str, choices=choices(), default="dit_t1")
    parser.add_argument("--path_type", type=str, choices=["linear", "gvp", "vp"], default="linear",
                        help="Interpolant path type")
    parser.add_argument("--prediction", type=str, choices=["velocity", "score", "noise"], default="velocity",
                        help="What the model predicts")
    parser.add_argument("--loss_weight", type=str, choices=["none", "velocity", "likelihood"], default="none",
                        help="Loss weighting scheme")
    parser.add_argument("--sample_steps", type=int, default=100, help="ODE integration steps for sampling")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_min", type=float, default=2e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--validate_every", type=int, default=2000)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/sit")
    parser.add_argument("--num_samples_for_logging", type=int, default=16)
    # Patch masking (Micro-Diffusion)
    parser.add_argument("--mask_ratio", type=float, default=0.0,
                        help="Fraction of patches to mask during training (0 = no masking, 0.75 = mask 75%%)")
    parser.add_argument("--patch_mixer_depth", type=int, default=0,
                        help="Depth of patch-mixer for deferred masking (0 = no mixer)")
    # Evaluation
    parser.add_argument("--fid_every", type=int, default=0,
                        help="Compute FID and IS every N steps (0 = disabled). Requires torch-fidelity.")
    parser.add_argument("--num_samples_for_fid", type=int, default=2048,
                        help="Number of generated samples for FID computation")
    args = parser.parse_args()
    return args


def update_ema(ema_model, model, beta):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1 - beta)


def compute_val_metrics(model, val_dataloader, transport, device):
    """Compute validation loss + data-prediction MSE (comparable across prediction types/paths)."""
    total_loss = 0
    total_dp_mse = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            total_loss += transport.training_losses(model, x).item()
            total_dp_mse += transport.data_prediction_mse(model, x).item()
            n += 1
    return total_loss / max(n, 1), total_dp_mse / max(n, 1)


def compute_fid_and_is(model, args, device, step):
    """Compute FID and Inception Score using torch-fidelity.

    Both metrics are scale-independent and directly comparable across
    different prediction types, path types, and loss weightings.
    """
    try:
        import torch_fidelity
    except ImportError:
        print("torch-fidelity not installed. Skipping FID/IS computation.")
        return None, None

    model.eval()
    n = args.num_samples_for_fid
    batch = 128
    shape = (args.in_channels, args.resolution, args.resolution)

    # Collect generated images as uint8 tensors
    all_images = []
    with torch.no_grad():
        generated = 0
        seed = step  # deterministic per step
        while generated < n:
            cur = min(batch, n - generated)
            samples = euler_ode_sample(
                model,
                shape=(cur, *shape),
                num_steps=args.sample_steps,
                device=device,
                seed=seed + generated,
                prediction=args.prediction,
                path_type=args.path_type,
            )
            # Convert [0,1] float to uint8
            imgs = (samples.clamp(0, 1) * 255).to(torch.uint8).cpu()
            all_images.append(imgs)
            generated += cur

    all_images = torch.cat(all_images, dim=0)[:n]

    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, t): self.t = t
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return self.t[i]

    ds = TensorDataset(all_images)
    if args.dataset == "cifar10":
        ref = "cifar10-train"
    else:
        ref = None  # skip FID for unknown datasets

    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=ds,
            input2=ref,
            cuda=True,
            isc=True,
            fid=(ref is not None),
            verbose=False,
            datasets_root=os.path.expanduser("~/.cache/torch_fidelity"),
            datasets_download=True,
        )
        fid_score = metrics.get("frechet_inception_distance")
        is_mean = metrics.get("inception_score_mean")
        return fid_score, is_mean
    except Exception as e:
        print(f"FID/IS computation failed: {e}")
        return None, None


def generate_samples(model, device, args, seed=0):
    model.eval()
    with torch.no_grad():
        samples = euler_ode_sample(
            model,
            shape=(args.num_samples_for_logging, args.in_channels, args.resolution, args.resolution),
            num_steps=args.sample_steps,
            device=device,
            seed=seed,
            prediction=args.prediction,
            path_type=args.path_type,
        )
    return samples


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data
    class Config:
        pass
    config = Config()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.val_split = 0.1
    train_dataloader, val_dataloader = load_data(config)

    # Model - pass patch_mixer_depth for masking support
    model_kwargs = {}
    if args.mask_ratio > 0 and args.patch_mixer_depth > 0:
        model_kwargs["patch_mixer_depth"] = args.patch_mixer_depth
    model = create_model(
        net=args.net,
        in_channels=args.in_channels,
        resolution=args.resolution,
        **model_kwargs,
    ).to(device)
    ema_model = copy.deepcopy(model) if args.use_ema else None

    # Transport (loss + sampling framework)
    transport = Transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps, lr_min=args.lr_min,
    )

    # Logging
    num_params = sum(p.numel() for p in model.parameters())
    if args.logger == "wandb":
        project = os.getenv("WANDB_PROJECT") or "nano-diffusion"
        run_name = f"sit-{args.path_type}-{args.prediction}-{args.net}"
        if args.mask_ratio > 0:
            run_name += f"-mask{args.mask_ratio}"
        run_params = vars(args).copy()
        run_params["model_parameters"] = num_params
        run_params["method"] = "sit"
        wandb.init(project=project, config=run_params, name=run_name)

    print(f"SiT training: path={args.path_type}, prediction={args.prediction}, "
          f"loss_weight={args.loss_weight}, mask_ratio={args.mask_ratio}")
    print(f"Model params: {num_params / 1e6:.2f}M")

    # Training loop
    step = 0
    train_start = time.time()
    while step < args.total_steps:
        for x, _ in train_dataloader:
            if step >= args.total_steps:
                break

            x = x.to(device)
            model.train()
            optimizer.zero_grad()

            if args.mask_ratio > 0:
                loss = _masked_train_step(model, x, transport, args)
            else:
                loss = transport.training_losses(model, x)

            loss.backward()
            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            optimizer.step()
            lr_scheduler.step()

            if args.use_ema:
                with torch.no_grad():
                    update_ema(ema_model, model, args.ema_beta)

            # Logging
            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - train_start
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                msg = f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} | {steps_per_sec:.1f} steps/s"
                print(msg)
                if args.logger == "wandb":
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/num_examples": step * args.batch_size,
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm
                    wandb.log(log_dict, step=step)

            # Validation
            if step % args.validate_every == 0 and step > 0 and val_dataloader:
                val_loss, val_dp_mse = compute_val_metrics(model, val_dataloader, transport, device)
                log_dict_val = {"val/loss": val_loss, "val/data_pred_mse": val_dp_mse}
                print(f"step {step} | val_loss {val_loss:.4f} | val_dp_mse {val_dp_mse:.4f}", end="")
                if args.use_ema and ema_model is not None:
                    ema_val_loss, ema_dp_mse = compute_val_metrics(ema_model, val_dataloader, transport, device)
                    log_dict_val["val/ema_loss"] = ema_val_loss
                    log_dict_val["val/ema_data_pred_mse"] = ema_dp_mse
                    print(f" | ema_dp_mse {ema_dp_mse:.4f}", end="")
                print()
                if args.logger == "wandb":
                    wandb.log(log_dict_val, step=step)

            # FID and Inception Score
            if args.fid_every > 0 and step > 0 and step % args.fid_every == 0:
                model_to_eval = ema_model if args.use_ema else model
                print(f"step {step} | computing FID and IS ({args.num_samples_for_fid} samples)...")
                fid_score, is_score = compute_fid_and_is(model_to_eval, args, device, step)
                if fid_score is not None:
                    print(f"step {step} | FID={fid_score:.2f} | IS={is_score:.2f}")
                    if args.logger == "wandb":
                        wandb.log({"eval/fid": fid_score, "eval/inception_score": is_score}, step=step)

            # Sampling
            if step > 0 and step % args.sample_every == 0:
                model_to_sample = ema_model if args.use_ema else model
                samples = generate_samples(model_to_sample, device, args)
                grid = make_grid(samples, nrow=4, padding=2)
                save_image(grid, checkpoint_dir / f"samples_step_{step}.png")
                if args.logger == "wandb":
                    images = (samples.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
                    wandb.log({
                        "samples": [wandb.Image(img, caption=f"step {step} #{i}") for i, img in enumerate(images)],
                        "sample_grid": wandb.Image(
                            (grid.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().round().astype("uint8"),
                            caption=f"Grid at step {step}",
                        ),
                    }, step=step)

            # Save
            if step % args.save_every == 0 and step > 0:
                ckpt_path = checkpoint_dir / f"model_step_{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                if args.use_ema:
                    torch.save(ema_model.state_dict(), checkpoint_dir / f"ema_model_step_{step}.pt")
                if args.logger == "wandb":
                    wandb.save(str(ckpt_path))

            step += 1

    # Final save
    if step > 100:
        final_path = checkpoint_dir / "model_final.pt"
        torch.save(model.state_dict(), final_path)
        if args.use_ema:
            torch.save(ema_model.state_dict(), checkpoint_dir / "ema_model_final.pt")
        if args.logger == "wandb":
            wandb.save(str(final_path))

    total_time = time.time() - train_start
    print(f"SiT training complete in {total_time/60:.1f}min. Checkpoints at {checkpoint_dir}")
    if args.logger == "wandb":
        wandb.log({"train/total_time_min": total_time / 60}, step=step)
        wandb.finish()


def _masked_train_step(model, x1, transport, args):
    """Training step with patch masking."""
    from nanodiffusion.models.patch_masking import compute_masked_loss

    batch_size = x1.shape[0]
    device = x1.device
    t = torch.rand(batch_size, device=device)
    x0 = torch.randn_like(x1)
    x_t, u_t = transport.path.plan(t, x0, x1)

    # Forward with masking
    output = model(t=t, x=x_t, mask_ratio=args.mask_ratio)

    if isinstance(output, tuple):
        pred, mask = output
    else:
        pred = output
        mask = None

    if hasattr(pred, "sample"):
        pred = pred.sample

    if mask is not None:
        # Compute target based on prediction type
        if transport.prediction == "velocity":
            target = u_t
        elif transport.prediction == "noise":
            target = x0
        elif transport.prediction == "score":
            _, sigma_t, _, _ = transport.path.coefficients(t)
            sigma_t_expanded = sigma_t.reshape(-1, *([1] * (x0.dim() - 1)))
            target = -x0 / sigma_t_expanded.clamp(min=1e-6)
        loss = compute_masked_loss(pred, target, mask, model.patch_size)
    else:
        loss = ((pred - u_t) ** 2).mean()

    return loss


if __name__ == "__main__":
    main()
