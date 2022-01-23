import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, weights_init
import torchvision.datasets as datasets
from utils import load_checkpoint, save_checkpoint

# HyperParams
LOAD_MODEL = False
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
EPOCHS = 1478
FEATURES_CRIT = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

# Dataset
transformation = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)
dataset = datasets.ImageFolder(
    root="indexTorch\spiderWGAN\datasets", transform=transformation
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initializing Gen and Disc
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
crit = Critic(CHANNELS_IMG, FEATURES_CRIT).to(device)
weights_init(gen)
weights_init(crit)

# Optims
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_crit = optim.RMSprop(crit.parameters(), lr=LEARNING_RATE)

# TensorBoard
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logss/real")
writer_fake = SummaryWriter(f"logss/fake")
writer_lossC = SummaryWriter(f"logss/lossC")
writer_lossG = SummaryWriter(f"logss/lossG")

step = 0

# Training
gen.train()
crit.train()

if LOAD_MODEL:
    load_checkpoint("CHECKPOINT_GEN.pt", gen, opt_gen, LEARNING_RATE)
    load_checkpoint("CHECKPOINT_CRIT.pt", crit, opt_crit, LEARNING_RATE)

for epoch in range(EPOCHS):
    if epoch % 5 == 0:
        save_checkpoint(gen, opt_gen, filename="CHECKPOINT_GEN.pt")
        save_checkpoint(crit, opt_crit, filename="CHECKPOINT_CRIT.pt")

    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            crit_real = crit(real).reshape(-1)
            crit_fake = crit(fake).reshape(-1)
            loss_crit = -(
                torch.mean(crit_real) - torch.mean(crit_fake)
            )  # - turns minimizing int maximizing
            crit.zero_grad()
            loss_crit.backward(retain_graph=True)
            opt_crit.step()
            # clip critic weights between -0.01, 0.01            
            for p in crit.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        ## Train Gen: min -E[critic(gen_fake)]
        gen_fake = crit(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print to tensorboard
        if batch_idx % 100 == 0:
            gen.eval()
            crit.eval()
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss C: {loss_crit:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                writer_lossC.add_scalar('Loss C', loss_crit, epoch)
                writer_lossG.add_scalar('Loss G', loss_gen, epoch)
            step += 1
            gen.train()
            crit.train()