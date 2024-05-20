import argparse

import lightning as L
import torch
from diffusers.models import AutoencoderKL
from lightning.pytorch.callbacks import *
from torchvision.transforms import transforms

from modules.diffusion import create_diffusion
from modules.dit_builder import DiT_models
from modules.training_utils import center_crop_arr


def train(args):
    print("Starting training..")

    latent_size = args.image_size // 8 # 256 // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        compile_components=False
    )  # define the model

    # load weights
    # state_dict = find_model("ckpts/dit6_512_2_8.ckpt")
    # if 'pytorch-lightning_version' in state_dict.keys():
    #     state_dict = state_dict["state_dict"]
    # model.load_state_dict(state_dict, strict=False)

    model.batch_size = args.global_batch_size
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    # training only
    model.diffusion = diffusion
    model.vae = vae
    model.train()
    model.vae.eval()  # not to train

    # epochs-- 500 epochs (4, 9, 14)
    model_ckpt = ModelCheckpoint(dirpath="ckpts/", monitor="train_loss", save_top_k=5, save_last=True,
                                 every_n_epochs=5)  # every_n_train_steps=200 // grad_batch_accum

    trainer = L.Trainer(
        enable_checkpointing=True,
        detect_anomaly=False,
        log_every_n_steps=5,
        accelerator="auto",
        devices="auto",
        max_epochs=args.epochs,
        precision="16-mixed" if args.precision == "fp16" else "32-true",
        callbacks=[model_ckpt,
                   StochasticWeightAveraging(swa_lrs=1e-2)],
        accumulate_grad_batches=1,
    )

    trainer.fit(model)  # updating of weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_Clipped")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parsed_args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, parsed_args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train(parsed_args)
