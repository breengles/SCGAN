import torch
import wandb
from torch import nn
from torch.optim import Adam
from tqdm.auto import trange

from src.SCDis import SCDis
from src.SCGen import SCGen
from src.log_utils import save_images
from src.losses import GANLoss, HistogramLoss
from src.utils import Norm, xavier_init
from src.vgg import VGG


class SCGAN(nn.Module):
    def __init__(
        self,
        img_size=128,
        lambda_idt=0.5,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_his_lip=1.0,
        lambda_his_skin=0.1,
        lambda_his_eye=1.0,
        lambda_vgg=5e-3,
        d_conv_dim=64,
        norm=Norm.SN,
        vgg_path="vgg/vgg.pth",
        vgg_conv_path="vgg/vgg_conv.pth",
        ngf=64,  # number of gen filters in first conv layer
        styledim=192,
        num_downsamplings=2,
        num_residuals=3,
        mlp_dim=256,
        n_components=3,
        input_nc=3,
    ):
        super().__init__()
        self.img_size = img_size
        self.d_conv_dim = d_conv_dim
        self.norm = norm
        self.mask_A = {}
        self.mask_B = {}
        self.vgg_path = vgg_path
        self.vgg_conv_path = vgg_conv_path
        self.lambdas = {
            "idt": lambda_idt,
            "A": lambda_A,
            "B": lambda_B,
            "hist_lip": lambda_his_lip,
            "hist_skin": lambda_his_skin,
            "hist_eye": lambda_his_eye,
            "vgg": lambda_vgg,
        }

        self.layers = ["r41"]

        self.SCGen = SCGen(
            dim=ngf,
            style_dim=styledim,
            num_downsamplings=num_downsamplings,
            num_residuals=num_residuals,
            mlp_dim=mlp_dim,
            n_components=n_components,
            inp_dim=input_nc,
        )

        self.D_A = SCDis(img_size, self.d_conv_dim, self.norm)
        self.D_B = SCDis(img_size, self.d_conv_dim, self.norm)

        self.vgg = VGG()

        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        self.criterionGAN = GANLoss()
        self.criterionHist = HistogramLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    def move_to_device(self):
        self.SCGen.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)
        self.vgg.to(self.device)

        self.criterionL1.to(self.device)
        self.criterionL2.to(self.device)
        self.criterionHist.to(self.device)
        self.criterionGAN.device = self.device
        self.criterionHist.device = self.device

    def disable_vgg_grad(self):
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.SCGen.PSEnc.disable_vgg_grad()

    def initialize_weights(self):
        self.D_A.apply(xavier_init)
        self.D_B.apply(xavier_init)
        self.SCGen.apply(xavier_init)

        self.vgg.load_state_dict(torch.load(self.vgg_conv_path))
        self.SCGen.PSEnc.load_vgg(self.vgg_path)

        # if self.resume:
        #     G_path = os.path.join(self.snapshot_path, "G.pth")
        #     D_A_path = os.path.join(self.snapshot_path, "D_A.pth")
        #     D_B_path = os.path.join(self.snapshot_path, "D_B.pth")
        #
        #     if not os.path.exists(G_path):
        #         raise ValueError(f"{G_path} does not exist")
        #     if not os.path.exists(D_A_path):
        #         raise ValueError(f"{D_A_path} does not exist")
        #     if not os.path.exists(D_B_path):
        #         raise ValueError(f"{D_B_path} does not exist")
        #
        #     self.SCGen.load_state_dict(torch.load(G_path))
        #     print(f"loaded trained generator {G_path}..!")
        #
        #     self.D_A.load_state_dict(torch.load(D_A_path))
        #     print(f"loaded trained discriminator A {D_A_path}..!")
        #
        #     self.D_B.load_state_dict(torch.load(D_B_path))
        #     print(f"loaded trained discriminator B {D_B_path}..!")

        self.disable_vgg_grad()

    def extract_batch(self, batch):
        valid = batch["valid"]
        batch.pop("valid")
        out = {k: v[valid].to(self.device) for k, v in batch.items()}
        return out

    def calc_disc_loss(self, disc, img_A, img_B, seg_B):
        out = disc(img_A)
        loss_real = self.criterionGAN(out, 1.0)

        with torch.no_grad():
            fake = self.SCGen(img_B, seg_B, img_A)

        out = disc(fake)
        loss_fake = self.criterionGAN(out, 0.0)

        return 0.5 * (loss_real.mean() + loss_fake.mean())

    def calc_idt_loss(self, makeup_img, makeup_seg, nonmakeup_img, nonmakeup_seg):
        idt_A = self.SCGen(makeup_img, makeup_seg, nonmakeup_img)
        idt_B = self.SCGen(nonmakeup_img, nonmakeup_seg, makeup_img)

        idt_A_loss = self.criterionL2(idt_A, makeup_img) * self.lambdas["A"] * self.lambdas["idt"]
        idt_B_loss = self.criterionL2(idt_B, nonmakeup_img) * self.lambdas["B"] * self.lambdas["idt"]
        return idt_A_loss, idt_B_loss

    def calc_hist_loss(
        self,
        fake_makeup,
        fake_nonmakeup,
        makeup_img,
        nonmakeup_img,
        makeup_mask_lip,
        makeup_mask_skin,
        makeup_mask_left_eye,
        makeup_mask_right_eye,
        nonmakeup_mask_lip,
        nonmakeup_mask_skin,
        nonmakeup_mask_left_eye,
        nonmakeup_mask_right_eye,
    ):
        # LIP HIST
        hist_loss_lip_A = (
            self.criterionHist(fake_makeup, makeup_img, nonmakeup_mask_lip, makeup_mask_lip, nonmakeup_img)
            * self.lambdas["hist_lip"]
        )

        hist_loss_lip_B = (
            self.criterionHist(fake_nonmakeup, nonmakeup_img, makeup_mask_lip, nonmakeup_mask_lip, makeup_img)
            * self.lambdas["hist_lip"]
        )

        # SKIN HIST
        hist_loss_skin_A = (
            self.criterionHist(fake_makeup, makeup_img, nonmakeup_mask_skin, makeup_mask_skin, nonmakeup_img)
            * self.lambdas["hist_skin"]
        )

        hist_loss_skin_B = (
            self.criterionHist(fake_nonmakeup, nonmakeup_img, makeup_mask_skin, nonmakeup_mask_skin, makeup_img)
            * self.lambdas["hist_skin"]
        )

        # EYES
        hist_loss_left_eye_A = (
            self.criterionHist(fake_makeup, makeup_img, nonmakeup_mask_left_eye, makeup_mask_left_eye, nonmakeup_img,)
            * self.lambdas["hist_eye"]
        )
        hist_loss_left_eye_B = (
            self.criterionHist(
                fake_nonmakeup, nonmakeup_img, makeup_mask_left_eye, nonmakeup_mask_left_eye, makeup_img,
            )
            * self.lambdas["hist_eye"]
        )
        hist_loss_right_eye_A = (
            self.criterionHist(fake_makeup, makeup_img, nonmakeup_mask_right_eye, makeup_mask_right_eye, nonmakeup_img,)
            * self.lambdas["hist_eye"]
        )
        hist_loss_right_eye_B = (
            self.criterionHist(
                fake_nonmakeup, nonmakeup_img, makeup_mask_right_eye, nonmakeup_mask_right_eye, makeup_img,
            )
            * self.lambdas["hist_eye"]
        )

        # TOTAL HIST LOSS
        hist_loss_A = hist_loss_lip_A + hist_loss_skin_A + hist_loss_left_eye_A + hist_loss_right_eye_A
        hist_loss_B = hist_loss_lip_B + hist_loss_skin_B + hist_loss_left_eye_B + hist_loss_right_eye_B

        return hist_loss_A, hist_loss_B

    def calc_cycle_loss(self, fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg):
        rec_A = self.SCGen(fake_makeup, nonmakeup_seg, nonmakeup_img)
        rec_B = self.SCGen(fake_nonmakeup, makeup_seg, makeup_img)

        loss_cycle_A = self.criterionL1(rec_A, nonmakeup_img) * self.lambdas["A"]
        loss_cycle_B = self.criterionL1(rec_B, makeup_img) * self.lambdas["B"]

        return loss_cycle_A, loss_cycle_B

    def calc_vgg_loss(self, fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img):
        with torch.no_grad():
            vgg_s = self.vgg(nonmakeup_img, self.layers)[0]
            vgg_r = self.vgg(makeup_img, self.layers)[0]

        vgg_fake_nonmakeup = self.vgg(fake_nonmakeup, self.layers)[0]
        vgg_fake_makeup = self.vgg(fake_makeup, self.layers)[0]
        loss_vgg_A = self.criterionL2(vgg_fake_nonmakeup, vgg_s) * self.lambdas["A"] * self.lambdas["vgg"]
        loss_vgg_B = self.criterionL2(vgg_fake_makeup, vgg_r) * self.lambdas["B"] * self.lambdas["vgg"]

        return loss_vgg_A, loss_vgg_B

    def calc_recon_loss(self, fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg):
        cycle_A, cycle_B = self.calc_cycle_loss(
            fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg
        )
        vgg_A, vgg_B = self.calc_vgg_loss(fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img)
        return cycle_A, cycle_B, vgg_A, vgg_B

    def calc_gan_loss(
        self,
        makeup_img,
        makeup_seg,
        nonmakeup_img,
        nonmakeup_seg,
        makeup_mask_lip,
        makeup_mask_skin,
        makeup_mask_left_eye,
        makeup_mask_right_eye,
        nonmakeup_mask_lip,
        nonmakeup_mask_skin,
        nonmakeup_mask_left_eye,
        nonmakeup_mask_right_eye,
    ):
        fake_makeup = self.SCGen(nonmakeup_img, nonmakeup_seg, makeup_img)
        pred = self.D_A(fake_makeup)
        adv_loss_A = self.criterionGAN(pred, 1.0)

        fake_nonmakeup = self.SCGen(makeup_img, makeup_seg, nonmakeup_img)
        pred = self.D_B(fake_nonmakeup)
        adv_loss_B = self.criterionGAN(pred, 1.0)

        hist_loss_A, hist_loss_B = self.calc_hist_loss(
            fake_makeup,
            fake_nonmakeup,
            makeup_img,
            nonmakeup_img,
            makeup_mask_lip,
            makeup_mask_skin,
            makeup_mask_left_eye,
            makeup_mask_right_eye,
            nonmakeup_mask_lip,
            nonmakeup_mask_skin,
            nonmakeup_mask_left_eye,
            nonmakeup_mask_right_eye,
        )

        cycle_A, cycle_B, vgg_A, vgg_B = self.calc_recon_loss(
            fake_makeup, fake_nonmakeup, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg
        )

        return (
            adv_loss_A,
            adv_loss_B,
            hist_loss_A,
            hist_loss_B,
            cycle_A,
            cycle_B,
            vgg_A,
            vgg_B,
            [nonmakeup_img, makeup_img, fake_makeup],
        )

    def fit(
        self, trainloader, epochs=1, g_step=1, g_lr=2e-4, d_lr=2e-4, betas=(0.5, 0.999), save_step=1,
    ):
        self.initialize_weights()
        self.move_to_device()

        g_optim = Adam(self.SCGen.parameters(), lr=g_lr, betas=betas)
        d_A_optim = Adam(self.D_A.parameters(), lr=d_lr, betas=betas)
        d_B_optim = Adam(self.D_B.parameters(), lr=d_lr, betas=betas)

        it = 0
        for _ in trange(epochs, desc="Training"):
            for batch in trainloader:
                if batch["valid"].sum() == 0:
                    continue

                it += 1

                batch = self.extract_batch(batch)

                d_A_loss = self.calc_disc_loss(
                    self.D_A, batch["makeup_img"], batch["nonmakeup_img"], batch["nonmakeup_seg"]
                )
                d_A_optim.zero_grad()
                d_A_loss.backward()
                d_A_optim.step()

                d_B_loss = self.calc_disc_loss(
                    self.D_B, batch["nonmakeup_img"], batch["makeup_img"], batch["makeup_seg"]
                )
                d_B_optim.zero_grad()
                d_B_loss.backward()
                d_B_optim.step()

                if (it + 1) % g_step == 0:
                    idt_A, idt_B = self.calc_idt_loss(
                        batch["makeup_img"], batch["makeup_seg"], batch["nonmakeup_img"], batch["nonmakeup_seg"]
                    )

                    (
                        adv_loss_A,
                        adv_loss_B,
                        hist_loss_A,
                        hist_loss_B,
                        cycle_A,
                        cycle_B,
                        vgg_A,
                        vgg_B,
                        imgs,
                    ) = self.calc_gan_loss(
                        batch["makeup_img"],
                        batch["makeup_seg"],
                        batch["nonmakeup_img"],
                        batch["nonmakeup_seg"],
                        batch["makeup_mask_lip"],
                        batch["makeup_mask_skin"],
                        batch["makeup_mask_left_eye"],
                        batch["makeup_mask_right_eye"],
                        batch["nonmakeup_mask_lip"],
                        batch["nonmakeup_mask_skin"],
                        batch["nonmakeup_mask_left_eye"],
                        batch["nonmakeup_mask_right_eye"],
                    )

                    idt_loss = 0.5 * (idt_A + idt_B)
                    recon_loss = 0.5 * (cycle_A + cycle_B + vgg_A + vgg_B)
                    g_loss = (adv_loss_A + adv_loss_B + recon_loss + idt_loss + hist_loss_A + hist_loss_B).mean()

                    g_optim.zero_grad()
                    g_loss.backward()
                    g_optim.step()

                    wandb.log(
                        {
                            "gen/adv-A": adv_loss_A.detach().cpu().numpy().mean(),
                            "gen/adv-B": adv_loss_B.detach().cpu().numpy().mean(),
                            "gen/cyc-A": cycle_A.detach().cpu().numpy().mean(),
                            "gen/cyc-B": cycle_B.detach().cpu().numpy().mean(),
                            "gen/idt": idt_loss.detach().cpu().numpy().mean(),
                            "gen/recon": (cycle_A + cycle_B).detach().cpu().numpy().mean(),
                            "gen/vgg": (vgg_A + vgg_B).detach().cpu().numpy().mean(),
                            "gen/A-hist": hist_loss_A.detach().cpu().numpy().mean(),
                            "gen/total": g_loss.detach().cpu().numpy().mean(),
                            "disc/A": d_A_loss.detach().cpu().numpy().mean(),
                            "disc/B": d_B_loss.detach().cpu().numpy().mean(),
                        }
                    )

                    if (it + 1) % save_step == 0:
                        save_images(imgs)
