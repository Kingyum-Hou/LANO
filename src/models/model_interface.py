import torch
import pytorch_lightning as pl
import numpy as np

from typing import Any
from omegaconf import DictConfig
from einops import rearrange

from models.IPOT import EncoderProcessorDecoder as IPOT, IPOTBasicPreprocessor, IPOTEncoder, IPOTProcessor, IPOTDecoder
from models.FNO import FNO2d
from models.Ours import OursModel
from models.MIONet import MIONet_periodic as MIONet
from models.OFORMER import OFormer
from models.OFORMER_FILLGAP import OFormerFillGap
import matplotlib.pyplot as plt
from tools import LpLoss
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingLR, MultiStepLR
import torch.nn.functional as F


def random_half_false_shared(mask):
    B, N, T = mask.shape
    for b in range(B):
        true_indices = torch.nonzero(mask[b, :, 0], as_tuple=False).squeeze(1)  # 只取 t=0 的索引
        num_to_flip  = len(true_indices) // 2  # 抽取一半数量
        indices_to_flip = true_indices[torch.randperm(len(true_indices))[:num_to_flip]]  # 随机抽取索引
        mask[b, indices_to_flip, :] = False
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
    elif cfg.name == "MIONET":
        sensors = int(torch.prod(torch.tensor(cfg.space_dim)) * \
                      (1-cfg.missing_rate))
        size = [sensors, 256, 256, 256, 256, 256, 256, 256]  # T slices as input functions
        sizes = []
        # for T history
        for i in range(cfg.input_channel):
            sizes.append(size)
        # for 2D positions
        for i in range(2): 
            sizes.append(size)
        # x,y
        sizes.append(['p', 256, 256, 256, 256])
        # t
        sizes.append([1, 256, 256, 256, 256])
        model = MIONet(sizes, 
                       cfg.activation, 
                       cfg.initializer
                       )
    elif cfg.name == "OFORMER":
        model = OFormer(
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
        )
    elif cfg.name == "OFORMER_FILLGAP":
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
        )
    elif cfg.name == "OURS":
        model = OursModel(
                    T_in              = cfg.input_channel, 
                    is_fillGap        = cfg.is_fillGap, 
                    is_OrthoAttention = cfg.is_OrthoAttention, 
                    outputs_timeStep  = cfg.output_channel,
                )
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
    if cfg.name == "StepLR":
        cfg       = cfg.StepLR
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.name == "OneCycleLR":
        cfg              = cfg.OneCycleLR
        train_loader_len = int(cfg.num_train/cfg.batch_size)
        scheduler        = OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=train_loader_len)
    elif cfg.name == 'CosineAnnealingLR':
        cfg = cfg.CosineAnnealingLR
        train_loader_len = int(cfg.num_train/cfg.batch_size)
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


class IPOTModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model     = get_model(self.cfg_model)
        self.criterion = LpLoss(size_average=False)
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

    def step(self, batch: Any):
        mask, pos, a, u = batch
        B, HW, Ti = a.shape
        a_had    = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred = pos[0].clone()
        a = torch.cat([a_had, pos_had], dim=-1)
        # input = [x, mesh]; x:[a+mesh], mesh.shape=(N, 2)
        pred = self.model(a, pos_pred, 40)
        pred.reshape(u.shape)
        loss = self.criterion(pred.view(B, -1), u.view(B, -1))
        return loss, pred, u, B
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("train/loss",    loss/B,  sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/l2_loss", l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("validation/loss",    loss/B,  sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/l2_loss", l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("test/loss",    loss/B,  sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/l2_loss", l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}


class FNOModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model     = get_model(self.cfg_model)
        self.criterion = LpLoss(size_average=False)
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

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
        return loss, full_loss, pred, u, B, To
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("train/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("validation/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        self.log("test/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}


class OURSModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        self.criterion  = LpLoss(size_average=False)

        self.ntrain     = params_model.ntrain
        self.ntest      = params_model.ntest
        self.output_dim = params_model.output_channel
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

    def step_1_(self, batch: Any):
        # origin&Plus, rollout, 分别为一次4步和一次一步，Plus时without OrthoAttention
        mask, pos, a, u = batch
        B, HW, Ti = a.shape
        aPos    = torch.concat([a, pos], dim=-1)
        pos_pred = pos[0]
        # a=(NO, 12); pos=(N, 2)
        pred = self.model(aPos, mask[..., 0], pos_pred, 40)
        pred.reshape(u.shape)
        loss = self.criterion(pred.view(B, -1), u.view(B, -1))
        return loss, pred, u, B
    
    def step_2_(self, batch: Any):
        # pro, 一次4步，without OrthoAttention
        mask, pos, a, u = batch
        B, HW,Ti = a.shape
        _, _, To = u.shape
        pos_pred = pos[0]
        pred_trajectory = []
        loss = 0.
        To = int(To/self.output_dim)
        for t in range(0, To):
            aPos    = torch.concat([a, pos], dim=-1)
            y = u[..., t*self.output_dim:(t+1)*self.output_dim]
            pred = self.model(aPos, mask[..., 0], pos_pred)
            loss += self.criterion(pred.reshape(B, -1), y.reshape(B, -1))
            pred_trajectory.append(pred)
            a = torch.cat([a[..., self.output_dim:], y], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return loss, full_loss, pred, u, B, To
    
    def step_3_(self, batch: Any):
        # task3实验代码，实验结果见ns_task3_Ours，表现不佳
        mask, pos, a, u = batch
        B, HW,Ti = a.shape
        _, _, To = u.shape
        pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_ = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_half_false_shared(mask_.clone())
        agent_a    = a  [agent_mask.repeat(1, 1, 10).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        pred_trajectory = []
        loss = 0.
        To = int(To/self.output_dim)
        for t in range(0, To):
            agent_aPos = torch.concat([agent_a, agent_pos], dim=-1)
            y          = u[..., t*self.output_dim:(t+1)*self.output_dim]
            pred  = self.model(agent_aPos, agent_mask.squeeze(dim=-1), agent_pos, pos_pred, pos)
            loss += self.criterion(
                pred.view(B, -1), 
                torch.masked_select(y, mask_.bool()).view(B, -1)
            )
            pred_trajectory.append(pred)
            a = torch.cat([a[..., self.output_dim:], y], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(
            pred.view(B, -1), 
            torch.masked_select(u, mask_.bool()).view(B, -1)
        )
        return loss, full_loss, pred, u, B, To
    
    def step(self, batch: Any):
        # plus, task3, 
        mask, pos, a, u = batch
        B, HW,Ti = a.shape
        _, _, To = u.shape
        pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_ = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_half_false_shared(mask_.clone())
        agent_a    = a  [agent_mask.repeat(1, 1, 10).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        agent_aPos = torch.concat([agent_a, agent_pos], dim=-1)
        
        pred = self.model(agent_aPos, agent_mask.squeeze(dim=-1), mask_.squeeze(dim=-1),
                          agent_pos, pos_pred, pos, forward_steps=To)
        loss = self.criterion(
            pred.                                view(B, -1), 
            torch.masked_select(u, mask_.bool()).view(B, -1)
        )
        return loss, pred, u, B

    def rollout_(self, batch: Any):
        mask, pos, a, u = batch
        B, HW, Ti = a.shape
        _,  _, To = u.shape
        a_had     = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos

        pred_trajectory = []
        loss = 0.
        To = int(To/self.output_dim)
        for t in range(0, To):
            aPos_had = torch.concat([a_had, pos_had], dim=-1)
            y = u[..., t*self.output_dim:(t+1)*self.output_dim]
            pred = self.model(aPos_had, mask[..., 0],
                              pos_had, pos_pred, pos)
            loss += self.criterion(pred.reshape(B, -1), y.reshape(B, -1))
            pred_trajectory.append(pred)
            a_had = torch.cat([a_had[..., self.output_dim:], 
                               pred[mask[..., :self.output_dim].bool()
                                    ].reshape(B, -1, self.output_dim)
                               ], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return loss, full_loss, pred, u, B, To
    
    def rollout(self, batch: Any):
        mask, pos, a, u = batch
        B, HW, Ti = a.shape
        _,  _, To = u.shape
        a_had     = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos
        aPos_had  = torch.concat([a_had, pos_had], dim=-1)
        
        pred = self.model(aPos_had, mask[..., 0], torch.ones_like(mask[..., 0]),
                          pos_had, pos_pred, pos, forward_steps=To)
        loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return loss, pred, u, B

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, B = self.step(batch)
        self.log(
            "train/loss", loss/self.ntrain,
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "validation/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "validation/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "test/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "test/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}


class MIONetModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model     = get_model(self.cfg_model)
        self.criterion = LpLoss(size_average=False)
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

    def step_(self, batch: Any):
        mask, pos, a, u = batch
        B, HW, Ti =   a.shape
        To = 40
        _,  _,  D = pos.shape
        a_had    = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pred_trajectory = []
        loss = 0.
        for t in range(0, To):
            y = u[..., t:t+1]
            inputs = []
            for i in range(0, Ti):
                inputs.append(  a_had[..., i])
            for i in range(0, D):
                inputs.append(pos_had[..., i])
            inputs.append(pos[0].reshape(HW, D))
            inputs.append(torch.tensor(1., device=a.device).repeat(1, HW, 1).reshape(HW, 1))
            pred = self.model(inputs)
            loss += self.criterion(pred.reshape(B, -1), y.reshape(B, -1))
            pred_trajectory.append(pred)
            a_had = torch.cat([a_had[..., 1:], pred[mask[..., 0:1].bool()].reshape(B, -1, 1)], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        full_loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return loss, full_loss, pred, u, B, To
    
    def step(self, batch: Any):
        mask, pos, a, u = batch
        B, HW, Ti =   a.shape
        _,  _,  D = pos.shape
        To        = 40
        a_had    = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        inputs = []
        for i in range(0, Ti):
            inputs.append(  a_had[..., i])
        for i in range(0, D):
            inputs.append(pos_had[..., i])
        inputs.append(pos[0].unsqueeze(dim=1).repeat(1, To, 1).reshape(HW*To, D))
        inputs.append(torch.linspace(1, To, steps=To, device=a.device).reshape(1, To, 1).repeat(HW, 1, 1).reshape(HW*To, 1))
        pred = self.model(inputs)
        loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        full_loss = loss
        return loss, full_loss, pred, u, B, To
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        #self.log("train/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/loss",           loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("train/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        #self.log("validation/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/loss",           loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("validation/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, full_loss, yhat, yref, B, To = self.step(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))
        #self.log("test/loss",        loss/B/To, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/loss",           loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/full_loss", full_loss/B, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        self.log("test/l2_loss",       l2_loss, sync_dist=self.is_sync_dist, on_step=False, on_epoch=True)
        return {"loss": loss}


class OFormerModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model     = get_model(self.cfg_model)
        self.criterion = LpLoss(size_average=False)

        self.ntrain     = params_model.ntrain
        self.ntest      = params_model.ntest
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

    def step(self, batch: Any):
        mask, pos, a, u = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_ = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_half_false_shared(mask_.clone())
        agent_a    = a  [agent_mask.repeat(1, 1, 10).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        agent_aPos = torch.cat([agent_a, agent_pos], dim=-1)

        pred = self.model(agent_aPos, agent_pos, pos_pred, To)
        loss = self.criterion(
            pred.                                   view(B, -1),
            torch.masked_select(u,    mask_.bool()).view(B, -1)
        )
        return loss, pred, u, B

    def rollout(self, batch: Any):
        mask, pos, a, u = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        a_had     = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos
        aPos_had  = torch.concat([a_had, pos_had], dim=-1)
        pred = self.model(aPos_had, pos_had, pos_pred, To)
        loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "validation/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "validation/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "test/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "test/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}


class OFormerFillGapModule(pl.LightningModule):
    def __init__(self, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model     = get_model(self.cfg_model)
        self.criterion = LpLoss(size_average=False)

        self.ntrain     = params_model.ntrain
        self.ntest      = params_model.ntest
        
        self.is_sync_dist = torch.cuda.device_count() > 1
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg_optim)
        scheduler = get_scheduler(optimizer,               self.cfg_scheduler)
        return [optimizer], [scheduler]

    def step(self, batch: Any):
        mask, pos, a, u = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        pos_pred  = pos[mask[..., 2].bool()].reshape(B, -1, 2)

        # agent mission
        mask_ = mask[..., 0].unsqueeze(dim=-1)
        agent_mask = random_half_false_shared(mask_.clone())
        agent_a    = a  [agent_mask.repeat(1, 1, 10).bool()].reshape(B, -1, Ti)
        agent_pos  = pos[agent_mask.repeat(1, 1,  2).bool()].reshape(B, -1,  2)
        agent_aPos = torch.cat([agent_a, agent_pos], dim=-1)

        pred = self.model(agent_aPos, agent_pos, pos_pred, pos, To)
        loss = self.criterion(
            pred.                                   view(B, -1),
            torch.masked_select(u,    mask_.bool()).view(B, -1)
        )
        return loss, pred, u, B

    def rollout(self, batch: Any):
        mask, pos, a, u = batch
        B,  _, Ti = a.shape
        _,  _, To = u.shape
        a_had     = a  [mask[..., :10].bool()].reshape(B, -1, Ti)
        pos_had   = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred  = pos
        aPos_had  = torch.concat([a_had, pos_had], dim=-1)
        
        pred = self.model(aPos_had, pos_had, pos_pred, pos, To)
        loss = self.criterion(pred.reshape(B, -1), u.reshape(B, -1))
        return loss, pred, u, B

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, B = self.step(batch)
        self.log(
            "train/loss", loss/self.ntrain, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "validation/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "validation/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref, B = self.rollout(batch)
        l2_loss = F.mse_loss(yhat.view(B, -1), yref.view(B, -1))*B
        self.log(
            "test/loss", loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        self.log(
            "test/l2_loss", l2_loss/self.ntest, 
            sync_dist=self.is_sync_dist, on_step=False, 
            on_epoch=True, reduce_fx=torch.sum
        )
        return {"loss": loss}
