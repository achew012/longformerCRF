import os
from torch import nn
import torch
from typing import List, Any, Dict

# from data import NERDataset
from common.utils import *
from metric.eval import eval_ceaf
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoConfig,
    set_seed,
    get_linear_schedule_with_warmup,
)
import torch
import pytorch_lightning as pl
from clearml import Dataset as ClearML_Dataset
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
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels), requires_grad=True)
        self.log_alpha = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.score = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.log_delta = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.path = nn.Parameter(torch.zeros(1, 1, dtype=torch.long), requires_grad=False)

    @staticmethod
    def log_sum_exp_batch(log_Tensor, axis=-1):
        # shape (batch_size,n,m)
        sum_score = torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis)
        return torch.max(log_Tensor, axis)[0] + torch.log(sum_score)

    def reset_layers(self):
        self.log_alpha = self.log_alpha.fill_(0.)
        self.score = self.score.fill_(0.)
        self.log_delta = self.log_delta.fill_(0.)
        self.psi = self.psi.fill_(0.)
        self.path = self.path.fill_(0)

    def forward(self, feats, label_ids):
        forward_score = self._forward_alg(feats)
        max_logLL_allz_allx, path, gold_score = self._crf_decode(feats, label_ids)
        loss = torch.mean(forward_score - gold_score)
        self.reset_layers()
        return path, max_logLL_allz_allx, loss

    def _forward_alg(self, feats):
        """alpha-recursion or forward recursion; to compute the partition function"""
        # feats -> (batch size, num_labels)
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = self.log_alpha.expand(batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, seq_size):
            log_alpha = (self.log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        return self.log_sum_exp_batch(log_alpha)

    def _crf_decode(self, feats, label_ids):
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)
        score = self.score.expand(batch_size, 1)

        log_delta = self.log_delta.expand(batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_delta[:, 0, self.start_label_id] = 0
        psi = self.psi.expand(batch_size, seq_size, self.num_labels).clone()

        for t in range(1, seq_size):
            batch_trans_score = batch_transitions.gather(
                -1, (label_ids[:, t] * self.num_labels + label_ids[:, t-1]).view(-1, 1))
            temp_score = feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
            score = score + batch_trans_score + temp_score

            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = self.path.expand(batch_size, seq_size).clone()
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)
        for t in range(seq_size-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path, score

class NERLongformer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained(self.args.model)
        self.config.gradient_checkpointing = True
        self.config.num_labels = len(self.labels2idx.keys())
        self.model = AutoModelForTokenClassification.from_pretrained(self.args.model, config=self.config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        self.tokenizer.model_max_length = self.args.max_input_len
        self.softmax = nn.Softmax(dim=-1)
        self.crf_layer = Transformer_CRF(num_labels=self.config.num_labels, start_label_id=self.tokenizer.cls_token_id)

        # Get loss weights
        dataset_split = self.dataset["train"]
        dataset = NERDataset(dataset=dataset_split, tokenizer=self.tokenizer, labels2idx=self.labels2idx, args=self.args)
        y_train = torch.stack(dataset.processed_dataset["labels"]).view(-1).cpu().numpy()
        self.loss_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))

        # list of doc ids
        self.test_doc_ids = [doc["docid"] for doc in self.dataset["test"]]


        if self.args.use_entity_embeddings:

            pretrained_lm_path = bucket_ops.get_file(
                remote_path=self.args.embedding_path
            )

            lm_args={
                "lr": 5e-4,
                "num_epochs":5,
                "train_batch_size":12,
                "eval_batch_size":1,
                "max_length": 2048, # be mindful underlength will cause device cuda side error
                "max_span_len": 15,
                "max_spans": 25,
                "mlm_task": False,
                "bio_task": True,
            }
            lm_args = argparse.Namespace(**lm_args)

            # WARNING different longformer models have different calls
            self.model.longformer = PretrainedModel.load_from_checkpoint(pretrained_lm_path, args = lm_args).longformer


    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

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

        outputs = self.model(**batch) #, global_attention_mask=self._set_global_attention_mask(batch["input_ids"]))
        logits = outputs.logits
        logits, active_logits, loss = self.crf_layer(logits, batch["labels"])

        loss=None       
        if "labels" in batch.keys():
            loss_fct = nn.CrossEntropyLoss(self.loss_weights, ignore_index=-100)
            # Only keep active parts of the loss
            if batch["attention_mask"] is not None:
                active_loss = batch["attention_mask"].view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = batch["labels"].view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

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
        if self.args.use_crf:
            preds = outputs["logits"]
        else:    
            preds = torch.argmax(outputs["logits"], dim=-1)
        labels = batch["labels"]
        return {'val_loss': outputs["loss"], "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()       

        val_logits = torch.stack([x["preds"] for x in outputs], dim=0).view(-1, self.tokenizer.model_max_length)
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1, self.tokenizer.model_max_length)
        print(classification_report(val_labels.view(-1).cpu().detach(), val_logits.view(-1).cpu().detach()))
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_nb):
        outputs = self._evaluation_step('val', batch, batch_nb)
        if self.args.use_crf:
            preds = outputs["logits"]
        else:    
            preds = torch.argmax(self.softmax(outputs["logits"]), dim=-1)
        labels = batch["labels"]
        return {'test_loss': outputs["loss"], "preds": preds, "labels": labels, "tokens": batch["input_ids"]}

    def test_epoch_end(self, outputs):

        logs={}
        doctexts_tokens, golds = read_golds_from_test_file(os.path.join(dataset_folder, "data/muc4-grit/processed/"), self.tokenizer)

        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_preds = torch.stack([x["preds"] for x in outputs], dim=0).view(-1, self.tokenizer.model_max_length)
        test_labels = torch.stack([x["labels"] for x in outputs]).view(-1, self.tokenizer.model_max_length)

        tokens = torch.stack([x["tokens"] for x in outputs]).view(-1, self.tokenizer.model_max_length).cpu().detach().tolist()
        tokens = [self.tokenizer.convert_ids_to_tokens(token) for token in tokens]
        print(classification_report(test_labels.view(-1).cpu().detach(), test_preds.view(-1).cpu().detach(), target_names=[key for key in self.labels2idx.keys()]))       

        self.idx2labels = {key: value for value, key in self.labels2idx.items()}
        predictions = [[(idx, self.idx2labels[tag]) for idx, tag in enumerate(doc)] for doc in test_preds.cpu().detach().tolist()]        

        # Filter out "O" tags       
        focused = [[(idx, self.idx2labels[tag]) for idx, tag in enumerate(doc) if self.idx2labels[tag]!="O"] for doc in test_preds.cpu().detach().tolist()]        

        doc_span_list = []

        # Convert valid BIO tags to spans
        for doc in focused:
            span_list = []
            spans = []
            for tag in doc:
                idx = tag[0]
                prefix = tag[1][:2]
                classname = tag[1][2:]
                if prefix=="B-":
                    if len(spans)>0:
                        span_list.append(spans)
                        spans=[]
                    spans.append(tag)
                elif prefix=="I-" and len(spans)>0 and classname==spans[0][1][2:] and spans[-1][0]==idx-1:
                    spans.append(tag)
                else:                    
                    pass

            if len(spans)>0:
                span_list.append(spans)

            # Append (start_idx, end_idx, classname) into span_list
            span_list = [[span[0][0], span[-1][0], span[0][1][2:]] for span in span_list]
            doc_span_list.append(span_list)

        # Convert spans to MUC-4 template
        muc4_pred_format = [{key: [] for key in role_map.keys()} for i in range(len(doc_span_list))]
        for idx, (span_list, doc) in enumerate(zip(doc_span_list, tokens)):
            for span in span_list:
                #muc4_pred_format[idx][span[-1]].append(' '.join(remove_led_prefix_from_tokens(doc[span[0]:span[1]+1])))
                muc4_pred_format[idx][span[-1]].append(self.tokenizer.convert_tokens_to_string(doc[span[0]+1:span[1]+2]))

        preds = OrderedDict()
        for key, doc in zip(self.test_doc_ids, muc4_pred_format):
            if key not in preds:
                preds[key] = OrderedDict()
                for idx, role in enumerate(role_map.keys()):
                    preds[key][role] = []
                    if idx+1 > len(doc): 
                        continue
                    elif doc[role]:
                        for mention in doc[role]:
                            if mention=="</s>" or mention=="<s>":
                                continue
                            else:    
                                preds[key][role].append([mention])                

        results = eval_ceaf(preds, golds)
        print("================= CEAF score =================")
        print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
        print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
        print("==============================================")

        logs["test_micro_avg_f1_phi_strict"] = results["strict"]["micro_avg"]["f1"]
        logs["test_micro_avg_precision_phi_strict"] = results["strict"]["micro_avg"]["p"]
        logs["test_micro_avg_recall_phi_strict"] = results["strict"]["micro_avg"]["r"]

        clearlogger.report_scalar(title='f1', series = 'test', value=logs["test_micro_avg_f1_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='precision', series = 'test', value=logs["test_micro_avg_precision_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='recall', series = 'test', value=logs["test_micro_avg_recall_phi_strict"], iteration=1) 

        preds_list = [{key: value} for key, value in preds.items()]
        to_jsonl("./predictions.jsonl", preds_list)
        task.upload_artifact(name='predictions', artifact_object="./predictions.jsonl")


    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return [optimizer]