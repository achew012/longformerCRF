import os
from torch import nn
import torch
from typing import List, Any, Dict

# from data import NERDataset
from common.utils import *
from metric.eval import eval_ceaf
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    set_seed,
    get_linear_schedule_with_warmup,
)
import torch
import pytorch_lightning as pl
from clearml import Dataset as ClearML_Dataset
from sklearn.metrics import classification_report
import ipdb


class WeightedFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, alpha=None, gamma=2, reduction="none"):
        super(WeightedFocalLoss, self).__init__(alpha, reduction=reduction)
        self.gamma = gamma
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = alpha

    def forward(self, input, target):
        ce_loss = torch.nn.functional.cross_entropy(
            input,
            target.type(torch.cuda.LongTensor),
            reduction=self.reduction,
            weight=self.weight,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class Transformer_CRF(nn.Module):
    def __init__(self, num_labels, start_label_id):
        super().__init__()
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.transitions = nn.Parameter(torch.randn(
            self.num_labels, self.num_labels), requires_grad=True)
        self.log_alpha = nn.Parameter(
            torch.zeros(1, 1, 1), requires_grad=False)
        self.score = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.log_delta = nn.Parameter(
            torch.zeros(1, 1, 1), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.path = nn.Parameter(torch.zeros(
            1, 1, dtype=torch.long), requires_grad=False)

    @staticmethod
    def log_sum_exp_batch(log_Tensor, axis=-1):
        # shape (batch_size,n,m)
        sum_score = torch.exp(log_Tensor - torch.max(log_Tensor, axis)
                              [0].view(log_Tensor.shape[0], -1, 1)).sum(axis)
        return torch.max(log_Tensor, axis)[0] + torch.log(sum_score)

    def reset_layers(self):
        self.log_alpha = self.log_alpha.fill_(0.)
        self.score = self.score.fill_(0.)
        self.log_delta = self.log_delta.fill_(0.)
        self.psi = self.psi.fill_(0.)
        self.path = self.path.fill_(0)

    def forward(self, feats, label_ids):
        forward_score = self._forward_alg(feats)
        max_logLL_allz_allx, path, gold_score = self._crf_decode(
            feats, label_ids)
        loss = torch.mean(forward_score - gold_score)
        self.reset_layers()
        return path, max_logLL_allz_allx, loss

    def _forward_alg(self, feats):
        """alpha-recursion or forward recursion; to compute the partition function"""
        # feats -> (batch size, num_labels)
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = self.log_alpha.expand(
            batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, seq_size):
            log_alpha = (self.log_sum_exp_batch(self.transitions +
                         log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        return self.log_sum_exp_batch(log_alpha)

    def _crf_decode(self, feats, label_ids):
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(
            batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)
        score = self.score.expand(batch_size, 1)

        log_delta = self.log_delta.expand(
            batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_delta[:, 0, self.start_label_id] = 0
        psi = self.psi.expand(batch_size, seq_size, self.num_labels).clone()

        for t in range(1, seq_size):
            batch_trans_score = batch_transitions.gather(
                -1, (label_ids[:, t] * self.num_labels + label_ids[:, t-1]).view(-1, 1))
            temp_score = feats[:, t].gather(-1,
                                            label_ids[:, t].view(-1, 1)).view(-1, 1)
            score = score + batch_trans_score + temp_score

            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = self.path.expand(batch_size, seq_size).clone()
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)
        for t in range(seq_size-2, -1, -1):
            path[:, t] = psi[:, t +
                             1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path, score


class NERLongformer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()

        clearml_data_object = ClearML_Dataset.get(
            dataset_name=self.cfg.clearml_dataset_name,
            dataset_project=self.cfg.clearml_dataset_project_name,
            dataset_tags=list(self.cfg.clearml_dataset_tags),
            only_published=False,
        )
        self.dataset_path = clearml_data_object.get_local_copy()

        print("CUDA available: ", torch.cuda.is_available())

        self.config = AutoConfig.from_pretrained(
            self.cfg.model_name)
        self.config.num_labels = len(self.cfg.role_map)
        self.config.attention_window = self.cfg.attention_window
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.cfg.model_name, config=self.config)

        # Load tokenizer
        self.softmax = nn.Softmax(dim=-1)
        self.crf_layer = Transformer_CRF(
            num_labels=self.config.num_labels, start_label_id=self.tokenizer.cls_token_id)

        if cfg.grad_ckpt:
            self.model.longformer.gradient_checkpointing_enable()

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def forward(self, **batch):
        # input_ids, attention_mask = batch["input_ids"], batch["attention_mask"],

        # , global_attention_mask=self._set_global_attention_mask(batch["input_ids"]))
        outputs = self.model(**batch)
        logits = outputs.logits
        logits, active_logits, loss = self.crf_layer(logits, batch["labels"])

        return (loss, logits)

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(**batch)
        return {'loss': outputs[0]}

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        outputs = self.forward(**batch)
        return {'loss': outputs[0], "logits": outputs[1]}

    def validation_step(self, batch, batch_nb):
        outputs = self._evaluation_step('val', batch, batch_nb)
        preds = outputs["logits"]
        labels = batch["labels"]
        return {'val_loss': outputs["loss"], "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        val_loss = []
        val_labels = []
        val_logits = []

        for batch in outputs:
            batch_loss = batch["val_loss"]
            batch_preds = batch["preds"]
            batch_labels = batch["labels"]
            val_loss.append(batch_loss)
            val_labels.append(batch_labels)
            val_logits.append(batch_preds)

        val_loss = torch.stack(val_loss, 0).mean()
        val_logits = torch.cat(val_logits, 0)
        val_labels = torch.cat(val_labels, 0)

        print(classification_report(val_labels.view(-1).cpu().detach(),
              val_logits.view(-1).cpu().detach()))
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_nb):
        outputs = self._evaluation_step('val', batch, batch_nb)
        preds = outputs["logits"]
        labels = batch["labels"]
        return {'test_loss': outputs["loss"], "preds": preds, "labels": labels, "tokens": batch["input_ids"]}

    def test_epoch_end(self, outputs):
        test_loss = []
        test_labels = []
        test_logits = []
        test_tokens = []

        for batch in outputs:
            batch_loss = batch["test_loss"]
            batch_preds = batch["preds"]
            batch_labels = batch["labels"]
            batch_tokens = batch["tokens"]
            test_loss.append(batch_loss)
            test_labels.append(batch_labels)
            test_logits.append(batch_preds)
            test_tokens.append(batch_tokens)

        test_loss = torch.stack(test_loss, 0).mean()
        test_logits = torch.cat(test_logits, 0)
        test_labels = torch.cat(test_labels, 0)
        test_tokens = torch.cat(test_tokens, 0)

        tokens = [self.tokenizer.convert_ids_to_tokens(
            token) for token in test_tokens]

        print(classification_report(test_labels.view(-1).cpu().detach(),
              test_logits.view(-1).cpu().detach(), target_names=[key for key in self.cfg.role_map.keys()]))

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True),
                "monitor": "val_loss",
                "frequency": 1
            }
        }
