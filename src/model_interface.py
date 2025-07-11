import torch
import pytorch_lightning as pl
import time

from typing import Any
from omegaconf import DictConfig
from einops import rearrange

from models.IPOT import EncoderProcessorDecoder as IPOT, IPOTBasicPreprocessor, IPOTEncoder, IPOTProcessor, IPOTDecoder
from models.FNO import FNO2d
from models.Ours import OursModel
from models.Ours_irregular import OursIrregularModel
from models.Ours_lno import OursLNOModel
from models.MIONet import MIONet_periodic as MIONet
from models.OFormer import OFormer
from models.OFORMER_FILLGAP import OFormerFillGap
from tools import LpLoss, masked_loss_average, check_model_parameters_isnan, reshape2blocks, reshape2data, count_parameters, central_diff, rel_l2norm_loss
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingLR, MultiStepLR
import torch.nn.functional as F
import math


def random_false_shared(mask: torch.tensor, task: str, patch_size=4, patch_num=[16, 16]):
    if "task2" in task:
        B, N, T = mask.shape
        mask_patch = reshape2blocks(mask, patch_size=patch_size, patch_num=patch_num)
        num_observed = len(torch.nonzero(mask_patch[0, :, 0, 0, 0], as_tuple=False))
        num_to_flip = torch.randint(0, int(num_observed*0.5), (1,)).item()
        for b in range(B):
            true_indices = torch.nonzero(mask_patch[b, :, 0, 0, 0], as_tuple=False).squeeze(1)
            indices_to_flip = true_indices[torch.randperm(len(true_indices))[:num_to_flip]]
            mask_patch[b, indices_to_flip, :, :, :] = False
        mask = reshape2data(mask_patch, patch_size=patch_size, patch_num=patch_num)
    elif "task3" in task:
        B, N, T = mask.shape
        num_observed = len(torch.nonzero(mask[0, :, 0], as_tuple=False))
        num_to_flip = torch.randint(0, int(num_observed*0.5), (1,)).item()
        for b in range(B):
            true_indices = torch.nonzero(mask[b, :, 0], as_tuple=False).squeeze(1) 
            indices_to_flip = true_indices[torch.randperm(len(true_indices))[:num_to_flip]] 
            mask[b, indices_to_flip, :] = False
    elif "task4" in task:
        B, N, T = mask.shape
        mask_patch = reshape2blocks(mask, patch_size=patch_size, patch_num=patch_num)
        num_observed = len(torch.nonzero(mask_patch[0, :, 0, 0, 0], as_tuple=False))
        #num_to_flip = int(num_observed*0.2)
        num_to_flip = torch.randint(0, int(num_observed*0.5), (1,)).item()
        for b in range(B):
            true_indices = torch.nonzero(mask_patch[b, :, 0, 0, 0], as_tuple=False).squeeze(1)
            indices_to_flip = true_indices[torch.randperm(len(true_indices))[:num_to_flip]]
            mask_patch[b, indices_to_flip, :, :, :] = False
        mask = reshape2data(mask_patch, patch_size=patch_size, patch_num=patch_num)
    else:
        raise NotImplementedError
    
    return mask


def get_model(cfg):
    if cfg.name == "IPOT":
        preprocessor = IPOTBasicPreprocessor(
            position_encoding_type=cfg.position_encoding_type,
            in_channel=cfg.input_channel,
            pos_channel=cfg.pos_channel,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=cfg.num_bands,
                max_resolution=cfg.max_resolution,
            )
        )
        encoder = IPOTEncoder(
            input_channel=cfg.input_channel + (2 * sum(cfg.num_bands) + len(cfg.num_bands)),  # pos2fourier
            num_latents=cfg.num_latents,
            latent_channel=cfg.latent_channel,
            cross_heads_num=cfg.cross_heads_num,
            cross_heads_channel=cfg.cross_heads_channel,
            latent_init_scale=cfg.latent_init_scale
        )
        processor = IPOTProcessor(
            self_per_cross_attn=cfg.self_per_cross_attn,
            self_heads_channel=cfg.self_heads_channel,
            latent_channel=cfg.latent_channel,
            self_heads_num=cfg.self_heads_num,
            ff_mult=cfg.ff_mult,
        )
        decoder = IPOTDecoder(
            output_channel=cfg.output_channel,
            query_channel=2 * sum(cfg.num_bands) + len(cfg.num_bands),  # pos2fourier
            latent_channel=cfg.latent_channel,
            cross_heads_num=cfg.cross_heads_num,
            cross_heads_channel=cfg.cross_heads_channel,
            ff_mult=cfg.ff_mult,
            output_scale=cfg.output_scale,
            position_encoding_type=cfg.position_encoding_type,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=cfg.num_bands,
                max_resolution=cfg.max_resolution, )
        )
        model = IPOT(
            encoder=encoder,
            processor=processor,
            decoder=decoder,
            input_preprocessor=preprocessor
        )
    elif cfg.name == "FNO":
        model = FNO2d(cfg.modes, cfg.modes, cfg.latent_channel)
    elif cfg.name == "MIONet":
        H, W = cfg.space_size[0], cfg.space_size[1]
        h = int((H / cfg.downsample))
        w = int((W / cfg.downsample))
        if cfg.sensors is None:
            sensors = int((h*w) * (1-cfg.missing_rate))
        else:
            sensors = int(cfg.sensors)
        size = [sensors, 256, 256, 256, 256, 256, 256, 256]  # T slices as input functions
        sizes = []
        # for T history
        for i in range(cfg.input_size):
            sizes.append(size)
        # for 2D positions
        for _ in range(2): 
            sizes.append(size)
        # x,y
        sizes.append(['p', 256, 256, 256, 256])
        # t
        sizes.append([1, 256, 256, 256, 256])
        model = MIONet(sizes, cfg.activation, cfg.initializer)
    elif cfg.name == "OFormer":
        model = OFormer(cfg)
    elif cfg.name == "OFormer_fillgap":
        model = OFormerFillGap(
            cfg.in_channels,
            cfg.encoder_emb_dim,
            cfg.out_seq_emb_dim,
            cfg.encoder_heads,
            cfg.encoder_depth,
            cfg.decoder_emb_dim,
            cfg.out_channels,
            cfg.out_step,
            cfg.propagator_depth,
            cfg.fourier_frequency,
            is_fillGap=cfg.is_fillGap,
            scale_factor=cfg.scale_factor,
            r=cfg.r_size,
        )
    elif cfg.name == "Ours":
        model = OursModel(cfg)
    elif cfg.name == "Ours_irregular":
        model = OursIrregularModel(cfg)
    elif cfg.name == "Ours_lno":
        model = OursLNOModel(cfg)
    else:
        raise NotImplementedError
    return model


def get_optimizer(params, cfg):
    params = list(params)
    params = filter(lambda p: p.requires_grad, params)
    if cfg.optim_alg == "Adam":
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
    elif cfg.optim_alg == "AdamW":
        optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay, lr=cfg.lr)
    else:
        raise NotImplemented
    return optimizer


def get_scheduler(optimizer, cfg):
    batch_size = cfg.b_train_test[0]
    if cfg.name == "StepLR":
        cfg       = cfg.StepLR
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.name == "OneCycleLR":
        cfg              = cfg.OneCycleLR
        train_loader_len = (cfg.num_train // batch_size) + 1
        valid_keys = {"max_lr", "epochs", "div_factor", "pct_start", "final_div_factor"}
        scheduler_args = {k: cfg[k] for k in valid_keys if cfg[k] is not None}
        scheduler        = OneCycleLR(optimizer, steps_per_epoch=train_loader_len, **scheduler_args)
    elif cfg.name == 'CosineAnnealingLR':
        cfg = cfg.CosineAnnealingLR
        train_loader_len = (cfg.num_train // batch_size) + 1
        scheduler        = CosineAnnealingLR(optimizer, T_max=cfg.epochs*train_loader_len)
    elif cfg.name == 'milestones':
        cfg           = cfg.milestones
        num_steps     = cfg.epochs
        step_interval = cfg.step_interval
        milestones    = list(range(step_interval, num_steps, step_interval))
        scheduler     = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.gamma)
    else:
        raise NotImplementedError
    return scheduler


class ModuleTemplate(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        #count_parameters(self.model)
        self.criterion  = LpLoss(size_average=False)

        self.ntrain       = params_model.ntrain
        self.ntest        = params_model.ntest
        self.b_train_test = params_scheduler.b_train_test

        self.is_sync_dist = torch.cuda.device_count() > 1
        self.train_start_time = None
        self.epoch_times = []

        self.curriculum_steps = params_optim.curriculum_steps
        self.curriculum_ratio = params_optim.curriculum_ratio
        self.curriculum_init  = params_optim.curriculum_init
        self.max_epochs       = params_optim.max_epochs
        self.epoch_iterations = 0.
        self.T_output         = params_optim.T_all // 2

        self.use_grad         = params_optim.use_grad

    def on_train_epoch_start(self):
        self.train_start_time = time.time()

    def on_train_epoch_end(self):
        train_end_time = time.time()
        epoch_time = train_end_time - self.train_start_time
        self.epoch_times.append(epoch_time)
        self.epoch_iterations += 1.
        self.log(
            "train/epoch_time", epoch_time, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.mean
        )
        if self.use_grad:
            self.log(
                "train/curriculum_steps", self.get_curriculum_steps(),
                sync_dist=self.is_sync_dist, on_step=False, 
                on_epoch=True, reduce_fx=torch.mean
            )

    def on_train_end(self):
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f"Average epoch time: {avg_epoch_time:.2f} sec")

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step', 
                'frequency': 1,
            }
        }

    def training_step(self, batch: Any, batch_idx: int):
        full_loss, loss = self.step(batch)
        self.log(
            "train/full_loss", full_loss/self.ntrain,
            sync_dist=self.is_sync_dist, on_step=False,
            on_epoch=True, reduce_fx=torch.sum,
        )
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        full_loss = self.rollout(batch)
        self.log(
            "validation/full_loss", full_loss/self.ntest,
            sync_dist=self.is_sync_dist, on_step=False,
            on_epoch=True, reduce_fx=torch.sum,
        )
        return {"loss": full_loss}

    def test_step(self, batch: Any, batch_idx: int):
        full_loss      = self.rollout(batch[0])
        full_loss_high = self.rollout(batch[1])
        self.log(
            "test/full_loss", full_loss/self.ntest,
            sync_dist=self.is_sync_dist, on_step=False,
            on_epoch=True, reduce_fx=torch.sum,
        )
        self.log(
            "test/full_loss_high", full_loss_high/self.ntest,
            sync_dist=self.is_sync_dist, on_step=False,
            on_epoch=True, reduce_fx=torch.sum,
        )
        return {"loss": full_loss}
    
    def get_curriculum_steps(self):
        # curriculum_init = 1. / n
        if self.curriculum_steps > 0 and self.epoch_iterations < int(self.curriculum_ratio * self.max_epochs):
            progress = self.epoch_iterations / (self.curriculum_init * self.curriculum_ratio * self.max_epochs)
            curriculum_steps = self.curriculum_steps + \
                int(
                    max(0, progress - 1.) * \
                    ((self.T_output - self.curriculum_steps) / (int(1./self.curriculum_init) - 1))
                )
        else:
            curriculum_steps = self.T_output
        return curriculum_steps

    
class IPOTModule(ModuleTemplate):
    def step(self, batch: Any):
        mask, pos, xx, yy, _, task = batch
        B, HW, Ti = xx.shape
        _, _,  To = yy.shape
        mask_      = mask[..., 0].unsqueeze(dim=-1)
        #agent_mask = random_false_shared(mask_.clone(), task, patch_size=3, patch_num=[30, 60]) # ERA5 patch-wise
        #agent_mask = random_false_shared(mask_.clone(), task)
        agent_mask = random_false_shared(mask_.clone(), task, patch_size=2, patch_num=[18, 36]) # NS patch-wise
        x_had     = xx [agent_mask.repeat(1, 1, Ti).bool()].reshape(B, -1, Ti)
        pos_had   = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        pos_pred  = pos[0].clone()
        x_pos_had = torch.cat([x_had, pos_had], dim=-1)

        curriculum_steps = self.get_curriculum_steps()
        yy = yy[..., :curriculum_steps]
        pred = self.model(x_pos_had, pos_pred, curriculum_steps)
        pred.reshape(yy.shape)
        loss = self.criterion(
            torch.masked_select(pred, mask[..., :curriculum_steps].bool()).view(B, -1), 
            torch.masked_select(yy,   mask[..., :curriculum_steps].bool()).view(B, -1)
        )
        full_loss = loss
        return full_loss, loss
    
    def rollout(self, batch: Any):
        mask, pos, xx, yy, _, task = batch
        B, HW, Ti = xx.shape
        _, _,  To = yy.shape
        x_had     = xx [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos[0].clone()
        x_pos_had = torch.cat([x_had, pos_had], dim=-1)
        pred = self.model(x_pos_had, pos_pred, To)
        pred.reshape(yy.shape)
        full_loss = self.criterion(
            pred.view(B, -1), 
              yy.view(B, -1)
        )
        return full_loss
    

class FNOModule(ModuleTemplate):
    def step(self, batch: Any):
        _, _, a, u = batch
        B, HW, Ti = a.shape
        _, _,  To = u.shape
        a = rearrange(a, 'B (H W) T -> B H W T', H=64, W=64)
        u = rearrange(u, 'B (H W) T -> B H W T', H=64, W=64)
        pred_trajectory = []
        loss = 0.
        for t in range(0, To):
            y = u[..., t:t+1]
            pred = self.model(a)
            loss += self.criterion(pred.reshape(B, -1), y.reshape(B, -1))
            pred_trajectory.append(pred)
            a = torch.cat([a[..., 1:], pred], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return full_loss, loss
    
    def rollout(self, batch: Any):
        _, _, a, u = batch
        B, HW, Ti = a.shape
        _,  _, To = u.shape
        a = rearrange(a, 'B (H W) T -> B H W T', H=64, W=64)
        u = rearrange(u, 'B (H W) T -> B H W T', H=64, W=64)
        pred_trajectory = []
        for t in range(0, To):
            y = u[..., t:t+1]
            pred = self.model(a)
            pred_trajectory.append(pred)
            a = torch.cat([a[..., 1:], pred], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return full_loss


class MIONetModule(ModuleTemplate):
    def step(self, batch: Any):
        mask, pos, xx, yy, _, task = batch
        B, HW, Ti =   xx.shape
        _,  _,  D = pos.shape
        _,  _, To = yy.shape
        a_had    = xx [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        inputs = []
        for i in range(0, Ti):
            inputs.append(a_had[..., i])
        for i in range(0, D):
            inputs.append(pos_had[..., i])
        inputs.append(pos[0].unsqueeze(dim=1).repeat(1, To, 1).reshape(HW*To, D))
        inputs.append(torch.linspace(1, To, steps=To, device=xx.device).reshape(1, To, 1).repeat(HW, 1, 1).reshape(HW*To, 1))
        pred = self.model(inputs, To)
        pred = pred.reshape(yy.shape)
        loss = self.criterion(
            torch.masked_select(pred, mask[..., :To].bool()).view(B, -1), 
            torch.masked_select(yy,   mask[..., :To].bool()).view(B, -1)
        )
        full_loss = loss
        return full_loss, loss
    
    def rollout(self, batch: Any):
        mask, pos, xx, yy, _, task = batch
        B, HW, Ti =   xx.shape
        _,  _,  D = pos.shape
        _,  _, To = yy.shape
        a_had    = xx [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        inputs = []
        for i in range(0, Ti):
            inputs.append(a_had[..., i])
        for i in range(0, D):
            inputs.append(pos_had[..., i])
        inputs.append(pos[0].unsqueeze(dim=1).repeat(1, To, 1).reshape(HW*To, D))
        inputs.append(torch.linspace(1, To, steps=To, device=xx.device).reshape(1, To, 1).repeat(HW, 1, 1).reshape(HW*To, 1))
        pred = self.model(inputs, To)
        pred = pred.reshape(yy.shape)
        full_loss = self.criterion(pred.view(B, -1), yy.view(B, -1))
        return full_loss
    

class OFormerModule(ModuleTemplate):
    def step(self, batch: Any):
        mask, pos, a, u, _, task = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        if self.use_grad:
            pos_pred  = pos
        else:
            pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_      = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_false_shared(mask_.clone(), task, patch_size=2, patch_num=[18, 36])
        #agent_mask = random_false_shared(mask_.clone(), task)
        agent_a    = a  [agent_mask.repeat(1, 1, Ti).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        agent_aPos = torch.cat([agent_a, agent_pos], dim=-1)

        curriculum_steps = self.get_curriculum_steps()
        u         = u[..., :curriculum_steps]
        pred      = self.model(agent_aPos, agent_pos, pos_pred, curriculum_steps)
        if self.use_grad:
            loss = self.criterion(
                torch.masked_select(pred, mask_.bool()).view(B, -1),
                torch.masked_select(u,    mask_.bool()).view(B, -1)
            )
        else:
            loss = self.criterion(
                pred.                                   view(B, -1),
                torch.masked_select(u,    mask_.bool()).view(B, -1)
            )
        full_loss = loss

        # missing_rate = 0 available
        if self.use_grad:
            u_grad_x,    u_grad_y    = central_diff(u)
            pred_grad_x, pred_grad_y = central_diff(pred)
            grad_loss = rel_l2norm_loss(pred_grad_x, u_grad_x) + \
                        rel_l2norm_loss(pred_grad_y, u_grad_y)
            loss += 5e-2 * grad_loss
        
        return full_loss, loss

    def rollout(self, batch: Any):
        mask, pos, a, u, _, task = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        a_had     = a  [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos
        aPos_had  = torch.concat([a_had, pos_had], dim=-1)
        pred      = self.model(aPos_had, pos_had, pos_pred, To)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return full_loss


class OFormerFGModule(ModuleTemplate):
    def step(self, batch: Any):
        mask, pos, a, u, _, task = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_      = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_false_shared(mask_.clone(), task, patch_size=4, patch_num=[16, 16])
        agent_a    = a  [agent_mask.repeat(1, 1, Ti).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        agent_aPos = torch.cat([agent_a, agent_pos], dim=-1)
        pos_interp = pos.reshape(B, 64, 64, 2)[:, ::8, ::8, :].reshape(B, -1, 2)
        pred       = self.model(agent_aPos, agent_pos, pos_pred, pos_interp, To)
        loss  = self.criterion(
            pred                                   .view(B, -1),
            torch.masked_select(u,    mask_.bool()).view(B, -1)
        )
        full_loss = loss
        return full_loss, loss

    def rollout(self, batch: Any):
        mask, pos, a, u, _, task = batch
        B, _, Ti = a.shape
        _, _, To = u.shape
        a_had    = a  [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        aPos_had = torch.concat([a_had, pos_had], dim=-1)
        pos_pred = pos
        pos_interp = pos.reshape(B, 64, 64, 2)[:, ::8, ::8, :].reshape(B, -1, 2)
        pred     = self.model(aPos_had, pos_had, pos_pred, pos_interp, To)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return full_loss


class OursModule(ModuleTemplate):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__(params_model, params_optim, params_scheduler)
        self.alpha = params_model.alpha
        self.t     = params_model.t
        self.surrogate_ratio = params_model.surrogate_ratio

    def loss_surrogate(self, psi1, psi2):
        psi1 = rearrange(psi1, 'b h f c -> (b h f) c')
        psi2 = rearrange(psi2, 'b h f c -> (b h f) c')
        psi1 = psi1.div(psi1.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.t)
        psi2 = psi2.div(psi2.norm(dim=0).clamp(min=1e-6)) * math.sqrt(self.t)
        psi_K_psi_diag = (psi1 * psi2).sum(0)
        psi2_d_K_psi1 = torch.einsum('bi, bj -> ij', psi2, psi1)
        psi1_d_K_psi2 = torch.einsum('bi, bj -> ij', psi1, psi2)
        loss = - psi_K_psi_diag.sum() * 2
        reg  = (psi2_d_K_psi1 ** 2).triu(1).sum() + \
               (psi1_d_K_psi2 ** 2).triu(1).sum()
        loss /= psi_K_psi_diag.numel()
        reg  /= psi_K_psi_diag.numel()
        loss = loss + self.alpha*reg
        return loss
    
    def step(self, batch: Any):
        mask, pos, xx, yy, pos_ref, task = batch
        B, HW,Ti = xx.shape
        _, _, To = yy.shape

        # agent mission
        mask_      = mask[..., 0].unsqueeze(dim=-1)
        #agent_mask = random_false_shared(mask_.clone(), task, patch_size=3, patch_num=[30, 60]) # ERA5 patch-wise
        #agent_mask = random_false_shared(mask_.clone(), task, patch_size=2, patch_num=[18, 36]) # ERA5 patch-wise
        agent_mask = random_false_shared(mask_.clone(), task, patch_size=4, patch_num=[16, 16]) # NS patch-wise
        #agent_mask = random_false_shared(mask_.clone(), task) 
        #agent_mask = mask_.clone()  # no agent mission
        pred_trajectory = []
        loss = 0.

        curriculum_steps = self.get_curriculum_steps()
        yy = yy[..., :curriculum_steps]
        for t in range(0, curriculum_steps):
            y     = yy[..., t:t+1]
            pred  = self.model(pos_ref, xx, agent_mask)
            loss += self.criterion(
                torch.masked_select(pred, mask_.bool()).view(B, -1), 
                torch.masked_select(y,    mask_.bool()).view(B, -1)
            )
            psi1, psi2 = self.model.get_psi(pos_ref, xx, agent_mask, mask_)
            loss += self.surrogate_ratio * self.loss_surrogate(psi1, psi2)
            pred_trajectory.append(pred)
            xx = torch.cat([xx[..., 1:], y], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(
            torch.masked_select(pred, mask_.bool()).view(B, -1), 
            torch.masked_select(yy,   mask_.bool()).view(B, -1)
        )
        #check_model_parameters_isnan(self.model)
        return full_loss, loss
    
    def rollout(self, batch: Any):
        mask, pos, xx, yy, pos_ref, task = batch
        B, HW, Ti = xx.shape
        _,  _, To = yy.shape

        pred_trajectory = []
        for t in range(0, To):
            y    = yy[..., t:t+1]
            pred = self.model(pos_ref, xx, mask[..., :1])
            pred_trajectory.append(pred)
            xx = torch.cat([xx[..., 1:], pred], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), yy.reshape(B, -1))
        return full_loss

