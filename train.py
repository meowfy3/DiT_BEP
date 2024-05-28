import argparse
import matplotlib.pyplot as plt
import lightning as L
import torch
import csv
from diffusers.models import AutoencoderKL
from lightning.pytorch.callbacks import *
from torchvision.transforms import transforms
import sys  

from modules.diffusion import create_diffusion
from modules.dit_builder import DiT_models



class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')



class PrintLossCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.csv_file = 'losses.csv'

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')
        if train_loss is not None and val_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
            self.val_losses.append(val_loss.cpu().item())
            print(f"Epoch {trainer.current_epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Write to CSV
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([trainer.current_epoch, train_loss.cpu().item(), val_loss.cpu().item()])

    def on_train_end(self, trainer, pl_module):
        # Plot the losses
        epochs = range(len(self.train_losses))
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('loss_plot.png')
        plt.show()


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
        callbacks=[model_ckpt,  PrintLossCallback(), LitProgressBar()],
        accumulate_grad_batches=1,
        strategy='ddp_find_unused_parameters_true',
    )

    trainer.fit(model)  # updating of weights
    trainer.validate(model);


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
