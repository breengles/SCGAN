import os

import torch
import wandb
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from tqdm.auto import tqdm, trange

from src.SCGen import SCGen
from . import net_utils
from .SCDis import SCDis
from .losses import GANLoss, HistogramLoss
from .utils import wandb_save_images
from .vgg import VGG


class SCGAN(nn.Module):
    def __init__(
        self,
        img_size=128,
        phase="train",
        lambda_idt=0.5,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_his_lip=1.0,
        lambda_his_skin=0.1,
        lambda_his_eye=1.0,
        lambda_vgg=5e-3,
        d_conv_dim=64,
        d_repeat_num=3,
        ngf=64,
        norm1="SN",
        style_dim=192,
        n_downsampling=2,
        n_res=3,
        mlp_dim=256,
        n_components=3,
        input_nc=3,
        ispartial=False,
        isinterpolation=False,
        pretrained_path=None,
        vgg_root="vgg",
        hsv=False,
        fast_matching=False,
        l1=False,
    ):
        super().__init__()

        self.img_size = img_size
        self.phase = phase
        self.lambda_idt = lambda_idt
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_his_lip = lambda_his_lip
        self.lambda_his_skin = lambda_his_skin
        self.lambda_his_eye = lambda_his_eye
        self.lambda_vgg = lambda_vgg
        self.d_conv_dim = d_conv_dim
        self.d_repeat_num = d_repeat_num
        self.norm1 = norm1
        self.ispartial = ispartial
        self.isinterpolation = isinterpolation

        self.pretrained_path = pretrained_path

        self.layers = ["r41"]

        self.mask_A = {}
        self.mask_B = {}

        self.lips = True
        self.eye = True
        self.skin = True

        self.SCGen = SCGen(
            ngf,
            style_dim,
            n_downsampling,
            n_res,
            mlp_dim,
            n_components,
            input_nc,
            phase,
            ispartial=ispartial,
            isinterpolation=isinterpolation,
            vgg_root=vgg_root,
        )

        self.D_A = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)
        self.D_B = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)

        self.D_A.apply(net_utils.weights_init_xavier)
        self.D_B.apply(net_utils.weights_init_xavier)
        self.SCGen.apply(net_utils.weights_init_xavier)

        self.SCGen.PSEnc.load_vgg(os.path.join(vgg_root, "vgg.pth"))

        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.load_checkpoint()

        if self.phase == "train":
            self.vgg = VGG()
            self.vgg.load_state_dict(torch.load(os.path.join(vgg_root, "vgg_conv.pth")))
            self.vgg.cuda()

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss()
        self.criterionHis = HistogramLoss(hsv=hsv, fast_matching=fast_matching, l1=l1)

        # kinda rude
        self.SCGen.cuda()
        self.criterionHis.cuda()
        self.criterionGAN.cuda()
        self.criterionL1.cuda()
        self.criterionL2.cuda()
        self.D_A.cuda()
        self.D_B.cuda()

    @property
    def device(self):
        return next(self.parameters()).device

    def load_checkpoint(self):
        state_dict = torch.load(self.pretrained_path)

        if "epoch" in state_dict.keys():
            self.SCGEN.load_state_dict(state_dict["SCGEN_state_dict"])
            self.D_A.load_state_dict(state_dict["D_A_state_dict"])
            self.D_D.load_state_dict(state_dict["D_B_state_dict"])

            print(f"Loaded whole model from {self.pretrained_path}!")
        else:
            self.SCGen.load_state_dict(state_dict)
            print(f"Loaded generator model from {self.pretrained_path}!")

    def set_input(self, input):
        self.mask_A = input["mask_A"]
        self.mask_B = input["mask_B"]
        self.makeup = input["makeup_img"]
        self.nonmakeup = input["nonmakeup_img"]
        self.makeup_seg = input["makeup_seg"]
        self.nonmakeup_seg = input["nonmakeup_seg"]
        # self.makeup_unchanged = input["makeup_unchanged"]
        # self.nonmakeup_unchanged = input["nonmakeup_unchanged"]

    def to_var(self, x, requires_grad=False):
        if isinstance(x, list):
            return x

        if torch.cuda.is_available():
            x = x.cuda()

        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def extract_batch(self, batch):
        valids = batch["valid"]

        mask_A = {k: v[valids] for k, v in batch["mask_A"].items()}
        mask_B = {k: v[valids] for k, v in batch["mask_B"].items()}

        out = {
            "nonmakeup_seg": batch["nonmakeup_seg"][valids],
            "makeup_seg": batch["makeup_seg"][valids],
            "nonmakeup_img": batch["nonmakeup_img"][valids],
            "makeup_img": batch["makeup_img"][valids],
            "mask_A": mask_A,
            "mask_B": mask_B,
            # "makeup_unchanged": makeup_unchanged,
            # "nonmakeup_unchanged": nonmakeup_unchanged,
        }

        return out

    def fit(
        self,
        dataloader,
        epochs=1,
        beta1=0.5,
        beta2=0.999,
        g_lr=2e-4,
        d_lr=2e-4,
        g_step=2e-4,
        g_delay=1,
        log_step=1,
        checkpoint_rate=None,
    ):
        g_optimizer = Adam(self.SCGen.parameters(), g_lr, (beta1, beta2))
        d_A_optimizer = Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), d_lr, (beta1, beta2))
        d_B_optimizer = Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), d_lr, (beta1, beta2))

        checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if checkpoint_rate == -1:
            if epochs // 10 > 0:
                checkpoint_rate = epochs // 10
            else:
                checkpoint_rate = 1

        it = 0
        for epoch in trange(epochs, desc="Epoch: ", leave=False):
            for data in tqdm(dataloader, desc="Batch: ", leave=False):
                if sum(data["valid"]) == 0:
                    continue

                it += 1
                data = self.extract_batch(data)

                loss = {}

                self.set_input(data)

                makeup, nonmakeup = (
                    self.to_var(self.makeup),
                    self.to_var(self.nonmakeup),
                )

                makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
                # makeup_unchanged=self.to_var(self.makeup_unchanged)
                # nonmakeup_unchanged=self.to_var(self.nonmakeup_unchanged)
                mask_makeup = {key: self.to_var(self.mask_B[key]) for key in self.mask_B}
                mask_nonmakeup = {key: self.to_var(self.mask_A[key]) for key in self.mask_A}

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real makeup
                out = self.D_A(makeup)
                d_loss_real = self.criterionGAN(out, True)

                # Fake makeup
                with torch.no_grad():
                    fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                out = self.D_A(fake_makeup)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                d_A_optimizer.step()

                # Logging
                loss["D-A-loss_real"] = d_loss_real.mean().detach().cpu().numpy()

                # training D_B, D_B aims to distinguish class A
                # Real non-makeup
                out = self.D_B(nonmakeup)
                d_loss_real = self.criterionGAN(out, True)

                # Fake de-makeup
                with torch.no_grad():
                    fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                out = self.D_B(fake_nonmakeup)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                d_B_optimizer.step()

                # Logging
                loss["D-B-loss_real"] = d_loss_real.mean().detach().cpu().numpy()

                # ================== Train G ================== #
                if it >= g_delay and (it + 1) % g_step == 0:
                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.SCGen(makeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    idt_B = self.SCGen(nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    loss_idt_A = self.criterionL2(idt_A, makeup) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL2(idt_B, nonmakeup) * self.lambda_B * self.lambda_idt
                    # loss_idt_A = self.criterionL1(idt_A, makeup) * self.lambda_A * self.lambda_idt
                    # loss_idt_B = self.criterionL1(idt_B, nonmakeup) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5
                    # loss_idt = loss_idt_A * 0.5

                    # GAN loss D_A(G_A(A))
                    # fake_A in class B,
                    fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    pred_fake = self.D_A(fake_makeup)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    pred_fake = self.D_B(fake_nonmakeup)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    # histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0

                    if self.lips:
                        g_A_lip_loss_his = (
                            self.criterionHis(
                                fake_makeup, makeup, mask_nonmakeup["mask_A_lip"], mask_makeup["mask_B_lip"], nonmakeup,
                            )
                            * self.lambda_his_lip
                        )
                        g_B_lip_loss_his = (
                            self.criterionHis(
                                fake_nonmakeup,
                                nonmakeup,
                                mask_makeup["mask_B_lip"],
                                mask_nonmakeup["mask_A_lip"],
                                makeup,
                            )
                            * self.lambda_his_lip
                        )
                        g_A_loss_his += g_A_lip_loss_his
                        g_B_loss_his += g_B_lip_loss_his

                    if self.skin:
                        g_A_skin_loss_his = (
                            self.criterionHis(
                                fake_makeup,
                                makeup,
                                mask_nonmakeup["mask_A_skin"],
                                mask_makeup["mask_B_skin"],
                                nonmakeup,
                            )
                            * self.lambda_his_skin
                        )
                        g_B_skin_loss_his = (
                            self.criterionHis(
                                fake_nonmakeup,
                                nonmakeup,
                                mask_makeup["mask_B_skin"],
                                mask_nonmakeup["mask_A_skin"],
                                makeup,
                            )
                            * self.lambda_his_skin
                        )
                        g_A_loss_his += g_A_skin_loss_his
                        g_B_loss_his += g_B_skin_loss_his

                    if self.eye:
                        g_A_eye_left_loss_his = (
                            self.criterionHis(
                                fake_makeup,
                                makeup,
                                mask_nonmakeup["mask_A_eye_left"],
                                mask_makeup["mask_B_eye_left"],
                                nonmakeup,
                            )
                            * self.lambda_his_eye
                        )
                        g_B_eye_left_loss_his = (
                            self.criterionHis(
                                fake_nonmakeup,
                                nonmakeup,
                                mask_makeup["mask_B_eye_left"],
                                mask_nonmakeup["mask_A_eye_left"],
                                makeup,
                            )
                            * self.lambda_his_eye
                        )
                        g_A_eye_right_loss_his = (
                            self.criterionHis(
                                fake_makeup,
                                makeup,
                                mask_nonmakeup["mask_A_eye_right"],
                                mask_makeup["mask_B_eye_right"],
                                nonmakeup,
                            )
                            * self.lambda_his_eye
                        )
                        g_B_eye_right_loss_his = (
                            self.criterionHis(
                                fake_nonmakeup,
                                nonmakeup,
                                mask_makeup["mask_B_eye_right"],
                                mask_nonmakeup["mask_A_eye_right"],
                                makeup,
                            )
                            * self.lambda_his_eye
                        )
                        g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                        g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his

                    # cycle loss
                    rec_A = self.SCGen(fake_makeup, nonmakeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    rec_B = self.SCGen(fake_nonmakeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)

                    g_loss_rec_A = self.criterionL1(rec_A, nonmakeup) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, makeup) * self.lambda_B

                    # vgg loss
                    with torch.no_grad():
                        vgg_s = self.vgg(nonmakeup, self.layers)[0]
                        vgg_r = self.vgg(makeup, self.layers)[0]

                    vgg_fake_nonmakeup = self.vgg(fake_nonmakeup, self.layers)[0]
                    g_loss_A_vgg = self.criterionL2(vgg_fake_nonmakeup, vgg_s) * self.lambda_A * self.lambda_vgg

                    vgg_fake_makeup = self.vgg(fake_makeup, self.layers)[0]
                    g_loss_B_vgg = self.criterionL2(vgg_fake_makeup, vgg_r) * self.lambda_B * self.lambda_vgg
                    # local-per
                    # vgg_fake_makeup_unchanged=self.vgg(fake_makeup*nonmakeup_unchanged,self.layers)
                    # vgg_makeup_masked=self.vgg(makeup*makeup_unchanged,self.layers)
                    # vgg_nonmakeup_masked=self.vgg(nonmakeup*nonmakeup_unchanged,self.layers)
                    # vgg_fake_nonmakeup_unchanged=self.vgg(fake_nonmakeup*makeup_unchanged,self.layers)
                    # g_loss_unchanged=(self.criterionL2(vgg_fake_makeup_unchanged, vgg_nonmakeup_masked)+self.criterionL2(vgg_fake_nonmakeup_unchanged,vgg_makeup_masked))

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # Combined loss
                    g_loss = (g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his).mean()

                    g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    g_optimizer.step()
                    # self.track("Generator backward")

                    # Logging
                    # self.loss['G-loss-unchanged']=g_loss_unchanged.mean().item()
                    loss["G-A-loss-adv"] = g_A_loss_adv.mean().detach().cpu().numpy()
                    loss["G-B-loss-adv"] = g_B_loss_adv.mean().detach().cpu().numpy()
                    loss["G-loss-org"] = g_loss_rec_A.mean().detach().cpu().numpy()
                    loss["G-loss-ref"] = g_loss_rec_B.mean().detach().cpu().numpy()
                    loss["G-loss-idt"] = loss_idt.mean().detach().cpu().numpy()
                    loss["G-loss-img-rec"] = (g_loss_rec_A + g_loss_rec_B).mean().detach().cpu().numpy()
                    loss["G-loss-vgg-rec"] = (g_loss_A_vgg + g_loss_B_vgg).mean().detach().cpu().numpy()
                    loss["G-A-loss-his"] = torch.mean(g_A_loss_his).detach().cpu().numpy()

                if (it + 1) % log_step == 0:
                    wandb.log(loss)
                    wandb_save_images([nonmakeup, makeup, fake_makeup, fake_nonmakeup])

            if checkpoint_rate is not None and (epoch + 1) % checkpoint_rate == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{epoch + 1}.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "D_A_state_dict": self.D_A.state_dict(),
                        "D_B_state_dict": self.D_B.state_dict(),
                        "SCGen_state_dict": self.SCGen.state_dict(),
                    },
                    checkpoint_path,
                )
                wandb.save(checkpoint_path, base_path=checkpoint_dir, policy="end")

    @torch.no_grad()
    def test(self):
        self.SCGen.eval()
        self.D_A.eval()
        self.D_B.eval()

        makeups = []
        makeups_seg = []
        nonmakeups = []
        nonmakeups_seg = []

        for self.i, data in enumerate(self.dataloader):
            if len(data) == 0:
                print("No eyes!!")
                continue

            self.set_input(data)
            makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup)
            makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
            makeups.append(makeup)
            makeups_seg.append(makeup_seg)
            nonmakeups.append(nonmakeup)
            nonmakeups_seg.append(nonmakeup_seg)

        source, ref1, ref2 = nonmakeups[0], makeups[0], makeups[1]
        source_seg, ref1_seg, ref2_seg = nonmakeups_seg[0], makeups_seg[0], makeups_seg[1]

        transfered = self.SCGen(source, source_seg, ref1, ref1_seg, ref2, ref2_seg)

        if not self.ispartial and not self.isinterpolation:
            results = [
                [source, ref1],
                [source, ref2],
                [ref1, source],
                [ref2, source],
            ]

            for i, img in zip(range(0, len(results)), transfered):
                results[i].append(img)

            self.imgs_save(results)

        elif not self.ispartial and self.isinterpolation:
            results = [
                [source, ref1],
                [source, ref2],
                [ref1, source],
                [ref2, source],
                [ref2, ref1],
            ]

            for i, imgs in zip(range(0, len(results) - 1), transfered):
                for img in imgs:
                    results[i].append(img)

            for img in transfered[-1]:
                results[-1].insert(1, img)
            results[-1].reverse()

            self.imgs_save(results)

        elif self.ispartial and not self.isinterpolation:
            results = [
                [source, ref1],
                [source, ref2],
                [source, ref1, ref2],
            ]

            for i, imgs in zip(range(0, len(results)), transfered):
                for img in imgs:
                    results[i].append(img)

            self.imgs_save(results)

        elif self.ispartial and self.isinterpolation:
            results = [
                [source, ref1],
                [source, ref1],
                [source, ref1],
                [source, ref2],
                [source, ref2],
                [source, ref2],
                [ref2, ref1],
                [ref2, ref1],
                [ref2, ref1],
            ]

            for i, imgs in zip(range(0, len(results) - 3), transfered):
                for img in imgs:
                    results[i].append(img)

            for i, imgs in zip(range(len(results) - 3, len(results)), transfered[-3:]):
                for img in imgs:
                    results[i].insert(1, img)
                results[i].reverse()

            self.imgs_save(results)
