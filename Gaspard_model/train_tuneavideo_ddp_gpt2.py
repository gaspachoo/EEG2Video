import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, PNDMScheduler
import torch.multiprocessing as mp
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

class EEGVideoDataset(Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks, e_blocks = [], []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith('.npy'):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith('.npy'):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)
        self.e_t   = np.concatenate(e_blocks, axis=0)
        assert self.z_hat.shape[0] == self.e_t.shape[0], (
            f"Mismatch: {self.z_hat.shape[0]} vs {self.e_t.shape[0]}"
        )

    def __len__(self):
        return len(self.z_hat)

    def __getitem__(self, idx):
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et

class TuneAVideoTrainer:
    def __init__(self, args, rank: int, local_rank: int, world_size: int):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')

        # --- Dataset & DataLoaders ---
        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        n_train = int(0.8 * len(dataset))
        n_val   = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_sampler = DistributedSampler(train_ds, world_size, rank)
        val_sampler   = DistributedSampler(val_ds, world_size, rank, shuffle=False)
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True)
        self.val_loader   = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)

        # --- Load the 4-bit quantized SD 3.5 pipeline ---
        # we load everything on CPU first, then move submodules appropriately
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 quant
            bnb_4bit_compute_dtype=torch.float16 # compute in fp16
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="cpu",        # start on CPU
            local_files_only=False,  # fetch from HF if not cached
        )

        # --- VAE on CPU, frozen ---
        self.vae = pipe.vae.to("cpu").eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # --- UNet 3D on GPU (4-bit) ---
        # Note: our UNet3DConditionModel inherits from diffusers' UNet2DConditional,
        # so we can directly swap in pipe.unet, which is already 4-bit quantized.
        print(f"[Rank{rank}] GPU memory before UNet → {torch.cuda.memory_allocated()/1e9:.2f} GB")
        self.unet = pipe.unet.to(self.device)
        print(f"[Rank{rank}] GPU memory after  UNet → {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # --- Scheduler (no device movement needed) ---
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # --- Tokenizer ---
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')

        # --- Wrap in your custom TuneAVideoPipeline ---
        self.pipeline = TuneAVideoPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler
        )
        self.pipeline.unet.train()

        # --- EEG projection head ---
        sem_dim   = dataset[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        # --- DDP wrapping ---
        self.pipeline.unet = DDP(
            self.pipeline.unet,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100
        )
        self.proj_eeg = DDP(self.proj_eeg, device_ids=[local_rank], output_device=local_rank)

        # --- Optimizer & AMP scaler ---
        self.optimizer = torch.optim.Adam(
            list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters()),
            lr=args.lr
        )
        self.scaler = GradScaler()

        # --- WandB logging (only on rank 0) ---
        if args.use_wandb and rank == 0:
            wandb.init(project='eeg2video-tuneavideo', config=vars(args))

        self.args = args
        self.train_sampler = train_sampler

    def _train_epoch(self, epoch:int):
        self.pipeline.unet.train()
        self.train_sampler.set_epoch(epoch)
        total_loss = 0.0
        for z0, et in tqdm(self.train_loader, desc=f"Rank{self.rank} Train {epoch}/{self.args.epochs}"):
            # prepare inputs
            z0 = z0.to(self.device).permute(0,2,1,3,4)
            et = et.to(self.device).unsqueeze(1)
            et = self.proj_eeg(et)

            # diffuse + forward
            noise = torch.randn_like(z0, device=self.device)
            ts = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
            z_t = self.scheduler.add_noise(z0, noise, ts)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = self.pipeline.unet(z_t, ts, encoder_hidden_states=et)
                loss = F.mse_loss(out.sample, noise)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            # free memory
            del z0, et, noise, ts, z_t, out, loss
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        total_loss = 0.0
        with torch.no_grad(), autocast():
            for z0, et in tqdm(self.val_loader, desc=f"Rank{self.rank} Validate"):
                z0 = z0.to(self.device).permute(0,2,1,3,4)
                et = et.to(self.device).unsqueeze(1)
                et = self.proj_eeg(et)

                noise = torch.randn_like(z0, device=self.device)
                ts = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
                z_t = self.scheduler.add_noise(z0, noise, ts)

                out = self.pipeline.unet(z_t, ts, encoder_hidden_states=et)
                total_loss += F.mse_loss(out.sample, noise).item()

                del z0, et, noise, ts, z_t, out
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch:int):
        if self.rank == 0:
            ckpt_dir = os.path.join(self.args.root, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            self.pipeline.unet.module.save_pretrained(
                os.path.join(ckpt_dir, f'unet_epoch{epoch}.pt')
            )
            print(f"✅ Saved checkpoint epoch {epoch}")

    def train(self):
        for e in range(1, self.args.epochs + 1):
            tr = self._train_epoch(e)
            vl = self._validate_epoch()
            if self.rank == 0:
                print(f"Epoch {e}: train_loss={tr:.4f}, val_loss={vl:.4f}")
            if e % self.args.save_every == 0:
                self._save_checkpoint(e)

def ddp_init(rank, args):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    local_rank = rank
    os.environ['LOCAL_RANK'] = str(local_rank)

    master_addr = os.environ.get('MASTER_ADDR', '10.59.121.193')
    master_port = os.environ.get('MASTER_PORT', '12356')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print(f"[DDP Init] rank={rank}, local_rank={local_rank}, world_size={os.environ['WORLD_SIZE']}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

    torch.cuda.set_device(local_rank)
    init_process_group(backend='nccl', rank=rank, world_size=int(os.environ['WORLD_SIZE']))

    trainer = TuneAVideoTrainer(args, rank, local_rank, int(os.environ['WORLD_SIZE']))
    trainer.train()
    destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--zhat_dir',    type=str, default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir',     type=str, default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--use_wandb',   action='store_true')
    parser.add_argument('--save_every',  type=int, default=10)
    parser.add_argument('--root',        type=str, default=root)
    parser.add_argument(
        '--model_name',
        type=str,
        default='argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized',
        help='HF repo for 4-bit quantized SD 3.5 Large'
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(ddp_init, args=(args,), nprocs=world_size)
