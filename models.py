import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import DiscriminatorLoss, visualize_results


class ColorizingNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.generator = UNET()
        self.discriminator = PatchDiscriminator()

        self.adversarial_loss = DiscriminatorLoss()
        self.pixelwise_loss = nn.L1Loss()
        self.register_buffer(
            "l1_lambda", torch.tensor(100)
        )  # the weight of the l1 loss. 100 as defined on the paper
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.generator(x)

    def train_model(self, train_loader, val_loader, num_epochs=100, lr=2e-4):
        writer = SummaryWriter()
        generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            discriminator_loss_sum = 0.0
            generator_loss_sum = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, (gray_imgs, color_imgs) in enumerate(pbar):
                if color_imgs is None:
                    continue

                gray_imgs = gray_imgs.to(self.device)
                color_imgs = color_imgs.to(self.device)

                # Train Discriminator
                discriminator_optimizer.zero_grad()
                generated_ab = self.generator(gray_imgs)

                # Real images
                real_combined = torch.cat([gray_imgs, color_imgs], dim=1)
                pred_real = self.discriminator(real_combined)
                loss_real = self.adversarial_loss(pred_real, True)

                # Fake images
                fake_combined = torch.cat([gray_imgs, generated_ab.detach()], dim=1)
                pred_fake = self.discriminator(fake_combined)
                loss_fake = self.adversarial_loss(pred_fake, False)

                # Total discriminator loss
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                discriminator_optimizer.step()

                # Train Generator
                generator_optimizer.zero_grad()
                # Generate again since we need the graph
                # generated_ab = self.generator(gray_imgs)
                fake_combined = torch.cat([gray_imgs, generated_ab], dim=1)
                pred_fake = self.discriminator(fake_combined)

                g_loss_gan = self.adversarial_loss(pred_fake, True)
                g_loss_l1 = (
                    self.pixelwise_loss(generated_ab, color_imgs) * self.l1_lambda
                )
                g_loss = g_loss_gan + g_loss_l1
                g_loss.backward()
                generator_optimizer.step()

                # Update running losses
                discriminator_loss_sum += d_loss.item()
                generator_loss_sum += g_loss.item()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "g_loss": generator_loss_sum / (step + 1),
                        "d_loss": discriminator_loss_sum / (step + 1),
                    }
                )

                # Log to tensorboard
                writer.add_scalar(
                    "Generator Loss", g_loss.item(), epoch * len(train_loader) + step
                )
                writer.add_scalar(
                    "Discriminator Loss",
                    d_loss.item(),
                    epoch * len(train_loader) + step,
                )

            # Calculate average losses for the epoch
            avg_g_loss = generator_loss_sum / len(train_loader)
            avg_d_loss = discriminator_loss_sum / len(train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}"
            )

            # Validation
            self.generator.eval()
            self.discriminator.eval()
            val_generator_loss = 0.0

            with torch.no_grad():
                for gray_imgs, color_imgs in val_loader:
                    gray_imgs = gray_imgs.to(self.device)
                    color_imgs = color_imgs.to(self.device)

                    generated_ab = self.generator(gray_imgs)
                    fake_combined = torch.cat([gray_imgs, generated_ab], dim=1)
                    pred_fake = self.discriminator(fake_combined)

                    val_g_loss = (
                        self.adversarial_loss(pred_fake, True).item()
                        + self.pixelwise_loss(generated_ab, color_imgs).item()
                        * self.l1_lambda
                    )
                    val_generator_loss += val_g_loss

                avg_val_g_loss = val_generator_loss / len(val_loader)
                writer.add_scalar("Validation Generator Loss", avg_val_g_loss, epoch)
                print(f"Validation Generator Loss: {avg_val_g_loss:.4f}")

                # Save best model
                if val_generator_loss < best_val_loss:
                    best_val_loss = val_generator_loss
                    torch.save(self.state_dict(), "best_colorization_model.pth")
                    print("Saved best model!")

            # Visualize results periodically
            if (epoch + 1) % 5 == 0:
                visualize_results(self, val_loader, self.device, epoch, writer)

        writer.close()


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, num_blocks=3):
        super().__init__()
        self.encoder = nn.ModuleList(
            [self.down_conv_block(in_channels, 64, norm=False)]
            + [self.down_conv_block(64 * 2**i, 64 * 2 ** (i + 1)) for i in range(num_blocks)]
            + [self.down_conv_block(64 * 2**num_blocks, 64 * 2**num_blocks) for _ in range(4)] # 4 times the last block as described on the paper
        )

        # the +1 on the exponent is due to the concatenation of the encoder ouputs
        self.decoder = nn.ModuleList(
            [
                self.up_conv_block(64 * 2**num_blocks, 64 * 2**num_blocks, dropout=False),
                *[
                    self.up_conv_block(64 * 2 ** (num_blocks + 1), 64 * 2**num_blocks)
                    for _ in range(3)
                ],
                *[
                    self.up_conv_block(
                        64 * 2 ** (num_blocks - i + 1), 64 * 2 ** (num_blocks - i - 1),
                        dropout=False
                    )
                    for i in range(num_blocks)
                ],
                self.up_conv_block(
                    64 * 2, out_channels, norm=False, activation="tanh", dropout=False
                ),  # times 2 due to concatenation of the encoder ouputs
            ]
        )

    def up_conv_block(self, in_channels, out_channels, norm=True, dropout=True, activation="relu"):
        post_conv_layers = []

        nn.Tanh() if activation == "tanh" else nn.ReLU(0.2)
        if activation != "tanh":
            if norm:
                post_conv_layers.append(nn.BatchNorm2d(out_channels)) 
            if dropout:
                post_conv_layers.append(nn.Dropout(0.5))
            post_conv_layers.append(nn.ReLU())
        else:
            post_conv_layers.append(nn.Tanh())

        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=not norm
            ),
            *post_conv_layers,
        )
        return block

    def down_conv_block(self, in_channels, out_channels, norm=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

        return block

    def forward(self, x):
        features_encoder = []
        # print(x.shape)
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i < len(self.encoder) - 1:
                features_encoder.append(x)
            # print(x.shape)

        for i, layer in enumerate(self.decoder):
            # it would be the same as concatenatinx x with x
            if i != 0:
                x = torch.cat(
                    [features_encoder.pop(), x], dim=1
                )  # dim = 1 due to the batch dimention
            x = layer(x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)