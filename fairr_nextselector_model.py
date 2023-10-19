'''
This script is the joint selector component of FaiRR_Next. It selects a rule from a given theory and statement and a set of facts from all the facts.
'''

from helper import *
from basemodel import BaseModel


class FaiRRNextSelector(BaseModel):
    def __init__(self, arch='roberta_large', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,\
                    optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, freeze_epochs=-1, gpus=1,\
                    hf_name='roberta-large'):
        super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)
        self.save_hyperparameters()
        assert arch == 'roberta_large'

        self.p                         = types.SimpleNamespace()
        self.p.arch                    = arch
        self.p.train_batch_size        = train_batch_size
        self.p.eval_batch_size         = eval_batch_size
        self.p.accumulate_grad_batches = accumulate_grad_batches
        self.p.learning_rate           = learning_rate
        self.p.max_epochs              = max_epochs
        self.p.optimizer               = optimizer
        self.p.adam_epsilon            = adam_epsilon
        self.p.weight_decay            = weight_decay
        self.p.lr_scheduler            = lr_scheduler
        self.p.warmup_updates          = warmup_updates
        self.p.freeze_epochs           = freeze_epochs
        self.p.gpus                    = gpus

        self.text_encoder    = AutoModel.from_pretrained(hf_name)
        self.tokenizer       = AutoTokenizer.from_pretrained(hf_name)
        out_dim              = self.text_encoder.config.hidden_size
        self.out_dim         = out_dim
        self.rule_classifier = nn.Linear(out_dim, 1)
        self.fact_classifier = nn.Linear(out_dim, 1)
        self.dropout         = torch.nn.Dropout(self.text_encoder.config.hidden_dropout_prob)

        xavier_normal_(self.fact_classifier.weight)
        self.fact_classifier.bias.data.zero_()
        xavier_normal_(self.rule_classifier.weight)
        self.rule_classifier.bias.data.zero_()

    def forward(self, input_ids, attn_mask):
        last_hidden_state = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)['last_hidden_state']  # shape (batchsize, seqlen, hiddensize)
        last_hidden_state = self.dropout(last_hidden_state)
        fact_logits = self.fact_classifier(last_hidden_state).squeeze()
        rule_logits = self.rule_classifier(last_hidden_state).squeeze()

        return fact_logits, rule_logits

    def predict(self, input_ids, token_mask, attn_mask):
        device  = input_ids.device
        outputs = self(input_ids, attn_mask)
        fact_logits, rule_logits  = outputs

        token_mask_copy=token_mask.detach().clone()
                
        # for rules
        token_mask=torch.where(token_mask_copy==1,1,0)

        # First filter out the logits corresponding to the valid tokens
        mask_len          = token_mask.sum(1) # (batchsize) eg [8,3,2,1]
        mask_nonzero      = torch.nonzero(token_mask) # (z, 2) size tensor, having x, y coordinates of non zero elements. z = no. of non zero elements
        y_indices         = torch.cat([torch.arange(x) for x in mask_len]).to(device)
        x_indices         = mask_nonzero[:, 0]
        filtered_logits   = torch.full((input_ids.shape[0], mask_len.max()), -1000.0).to(device)
        filtered_logits[x_indices, y_indices] = torch.masked_select(rule_logits, token_mask.bool())

        # Then compute the predictions for each of the logit
        argmax_filtered_logits	= torch.argmax(filtered_logits, dim=1)
        preds 					= (F.one_hot(argmax_filtered_logits, num_classes=filtered_logits.shape[1])).int()

        # truncating preds to remove the cls token predictions (prioritize rule selection)
        preds = preds[:, 1:]

        # Finally, save a padded rule matrix with indices of the rules and the corresponding mask
        pred_mask_lengths = preds.sum(1)
        pred_mask_nonzero = torch.nonzero(preds)
        y_indices         = torch.cat([torch.arange(x) for x in pred_mask_lengths]).to(device)
        x_indices         = pred_mask_nonzero[:, 0]
        filtered_rule_ids = torch.full((input_ids.shape[0], pred_mask_lengths.max()), -1).to(device)
        filtered_rule_ids[x_indices, y_indices] = pred_mask_nonzero[:, 1]
        filtered_rule_mask     = (filtered_rule_ids != -1)

        # Make the -1's -> 0 so that we can select some rule. Given the mask we can always prune this later
        # This step is non-intuitive here. To understand this, we need to consider the for loop in the main decoding logic where the rule_ids are used.
        filtered_rule_ids[~filtered_rule_mask] = 0

        # # filtered_rule_ids of size (b*maxrule_ids)
        # return filtered_rule_ids, filtered_mask

        # for facts
        token_mask=torch.where(token_mask_copy==2,1,0)

        # First filter out the logits corresponding to the [SEP] tokens
        mask_len          = token_mask.sum(1)
        mask_nonzero      = torch.nonzero(token_mask)
        y_indices         = torch.cat([torch.arange(x) for x in mask_len]).to(device)
        x_indices         = mask_nonzero[:, 0]
        filtered_logits   = torch.full((input_ids.shape[0], mask_len.max()), -1000.0).to(device)
        filtered_logits[x_indices, y_indices] = torch.masked_select(fact_logits, token_mask.bool())

        # Then compute the predictions for each of the logit
        preds             = (filtered_logits > 0.0)

        # Finally, save a padded fact matrix with indices of the facts and the corresponding mask
        pred_mask_lengths = preds.sum(1)
        pred_mask_nonzero = torch.nonzero(preds)
        y_indices         = torch.cat([torch.arange(x) for x in pred_mask_lengths]).to(device)
        x_indices         = pred_mask_nonzero[:, 0]
        filtered_fact_ids = torch.full((input_ids.shape[0], pred_mask_lengths.max()), -1).to(device)
        filtered_fact_ids[x_indices, y_indices] = pred_mask_nonzero[:, 1]

        # create mask for instances that don't have any fact selected so that they are pruned later in the inference loop
        filtered_fact_mask     = ~(filtered_fact_ids.shape[1] == (filtered_fact_ids == -1).sum(1))

        return filtered_rule_ids, filtered_rule_mask, filtered_fact_ids, filtered_fact_mask

    def calc_loss(self, rule_outputs, fact_outputs, targets, token_mask):
        token_mask_copy=token_mask.detach().clone()
        
        # for rules
        token_mask=torch.where(token_mask_copy==1,1,0)

        # all rows of target are one hot, i.e., there is only 1 rule that needs to be selected
        assert torch.all(torch.sum(targets * token_mask, dim=1) == torch.ones(targets.shape[0]).to(targets.device))

        exp_logits = torch.exp(rule_outputs)
        assert exp_logits.shape == token_mask.shape

        masked_exp_logits = exp_logits * token_mask
        norm_masked_exp_logits = masked_exp_logits/torch.sum(masked_exp_logits, dim=1).unsqueeze(-1)

        # convert the 0's to 1 in norm_masked_exp_logits so that log makes it 0
        # can be done by setting those indexes in norm_masked_exp_logits to 1., where token_mask = 0
        zeros_mask = (1 - token_mask).bool() # zeros_mask is 0 for SEP/CLS token and 1 everywhere else
        norm_masked_exp_logits[zeros_mask] = 1. # setting those indices of norm_mask = 1. where zeros_mask = 1

        # handling log(0) --> log(small_value) for places where token_mask is 1 and norm_masked_exp_logits is 0
        zeros_mask_ = (norm_masked_exp_logits == 0)
        norm_masked_exp_logits[zeros_mask_] = 1e-8

        logvals = torch.log(norm_masked_exp_logits)
        rule_targets=targets*token_mask
        rule_loss_reduced = F.nll_loss(logvals, torch.nonzero(rule_targets)[:, 1], reduction='mean')

        # for facts
        token_mask=torch.where(token_mask_copy==2,1,0)
        fact_targets=targets*token_mask
        loss_not_reduced =  F.binary_cross_entropy_with_logits(fact_outputs, fact_targets, reduction = 'none')
        assert loss_not_reduced.shape == token_mask.shape
        loss_masked = loss_not_reduced * token_mask
        fact_loss_reduced = loss_masked.sum()/token_mask.sum()

        return rule_loss_reduced, fact_loss_reduced
    
    def calc_acc_util(self, preds, targets, token_mask):
        acc_not_reduced = (preds == targets).float()
        acc_masked      = torch.mul(acc_not_reduced, token_mask)
        acc_reduced     = acc_masked.sum()/token_mask.sum()
        acc             = 100 * acc_reduced

        return acc
    
    def calc_acc(self, preds, targets, token_mask):
        return self.calc_acc_util(preds, targets, token_mask)
    
    def calc_F1_util(self, preds, targets, token_mask):
        '''calculates the binary F1 score between preds and targets, with positive class being 1'''
        assert preds.shape == targets.shape
        assert preds.shape == token_mask.shape

        # get only the relevant indices of preds and targets, ie those which are non zero in token_mask
        mask           = (token_mask == 1)
        preds_masked   = torch.masked_select(preds, mask).cpu()
        targets_masked = torch.masked_select(targets, mask).cpu()

        binary_f1_class1 = f1_score(y_true=targets_masked, y_pred=preds_masked, pos_label=1, average='binary')
        binary_f1_class0 = f1_score(y_true=targets_masked, y_pred=preds_masked, pos_label=0, average='binary')
        macro_f1         = f1_score(y_true=targets_masked, y_pred=preds_masked, average='macro')
        micro_f1         = f1_score(y_true=targets_masked, y_pred=preds_masked, average='micro')

        return {'f1_class1':binary_f1_class1, 'f1_class0':binary_f1_class0, 'macro_f1':macro_f1, 'micro_f1':micro_f1}
    
    def calc_F1(self, preds, targets, token_mask):
        return self.calc_F1_util(preds, targets, token_mask)
    

    def calc_perf_metrics_util(self, preds, targets, token_mask):
        acc       = self.calc_acc(preds, targets, token_mask)
        F1_scores = self.calc_F1(preds, targets, token_mask)

        return {'acc':acc, 'f1_class1':F1_scores['f1_class1'], 'f1_class0':F1_scores['f1_class0'], 'macro_f1':F1_scores['macro_f1'], 'micro_f1':F1_scores['micro_f1']}
    
    def calc_perf_metrics(self, preds, targets, token_mask):
        return self.calc_perf_metrics_util(preds, targets, token_mask)
    
    def run_step(self, batch, split):
        rule_outputs, fact_outputs = self(batch['all_sents'], batch['attn_mask'])
        token_mask_copy = batch['all_token_mask']
        targets    = batch['all_token_labels']

        rule_loss, fact_loss = self.calc_loss(rule_outputs.squeeze(), fact_outputs.squeeze(), targets.squeeze(), token_mask_copy.squeeze())

        # for rules
        token_mask=torch.where(token_mask_copy==1,1,0)
        relevant_outputs        = rule_outputs * token_mask
        argmax_relevant_outputs = torch.argmax(relevant_outputs, dim=1)
        rule_preds              = (F.one_hot(argmax_relevant_outputs, num_classes=rule_outputs.shape[1])).int()
        perf_metrics            = self.calc_perf_metrics(rule_preds.squeeze(), targets.squeeze(), token_mask.squeeze())

        if split == 'train':
            self.log(f'rules train_loss_step', rule_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            for metric in perf_metrics.keys():
                self.log(f'rules train_{metric}_step', perf_metrics[metric], on_step=True, on_epoch=True)
        else:
            self.log(f'rules {split}_loss_step', rule_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            for metric in perf_metrics.keys():
                self.log(f'rules {split}_{metric}_step', perf_metrics[metric], on_step=True, on_epoch=True)

        # for facts
        token_mask=torch.where(token_mask_copy==2,1,0)
        fact_preds        = (fact_outputs > 0.0).float().squeeze()
        perf_metrics      = self.calc_perf_metrics(fact_preds.squeeze(), targets.squeeze(), token_mask.squeeze())

        if split == 'train':
            self.log(f'facts train_loss_step', fact_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            for metric in perf_metrics.keys():
                self.log(f'facts train_{metric}_step', perf_metrics[metric], on_step=True, on_epoch=True)
        else:
            self.log(f'facts {split}_loss_step', fact_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            for metric in perf_metrics.keys():
                self.log(f'facts {split}_{metric}_step', perf_metrics[metric], on_step=True, on_epoch=True)

        lambd = 1
        loss = rule_loss+lambd*fact_loss
        self.log('loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    def aggregate_epoch(self, outputs, split):
        # rule_preds        = torch.cat([x['rule_preds'].reshape(-1) for x in outputs])
        # targets      = torch.cat([x['targets'].reshape(-1) for x in outputs])
        # token_mask   = torch.cat([x['token_mask'].reshape(-1) for x in outputs])
        # rule_loss         = torch.stack([x['loss'] for x in outputs]).mean()
        # rule_perf_metrics = self.calc_perf_metrics(rule_preds.squeeze(), targets.squeeze(), token_mask.squeeze())
        #
        # if split == 'train':
        #     self.log(f'rules train_loss_epoch', rule_loss.item())
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'rules train_{metric}_epoch', rule_perf_metrics[metric], prog_bar=True)
        # elif split == 'valid':
        #     self.log(f'rules valid_loss_epoch', rule_loss.item(), sync_dist=True)
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'rules valid_{metric}_epoch', rule_perf_metrics[metric], prog_bar=True)
        # elif split == 'test':
        #     self.log(f'rules test_loss_epoch', rule_loss.item(), sync_dist=True)
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'rules test_{metric}_epoch', rule_perf_metrics[metric], prog_bar=True)
        #     self.predictions = torch.stack((rule_preds, targets), dim=1)
        #     print('predictions tensor in ruletaker class, shape = {}'.format(self.predictions.shape))
        #
        # fact_preds        = torch.cat([x['preds'].reshape(-1) for x in outputs])
        # targets      = torch.cat([x['targets'].reshape(-1) for x in outputs])
        # token_mask   = torch.cat([x['token_mask'].reshape(-1) for x in outputs])
        # fact_loss         = torch.stack([x['loss'] for x in outputs]).mean()
        # fact_perf_metrics = self.calc_perf_metrics(fact_preds.squeeze(), targets.squeeze(), token_mask.squeeze())
        #
        # if split == 'train':
        #     self.log(f'facts train_loss_epoch', fact_loss.item())
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'facts train_{metric}_epoch', fact_perf_metrics[metric], prog_bar=True)
        # elif split == 'valid':
        #     self.log(f'facts valid_loss_epoch', fact_loss.item(), sync_dist=True)
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'facts valid_{metric}_epoch', fact_perf_metrics[metric], prog_bar=True)
        # elif split == 'test':
        #     self.log(f'facts test_loss_epoch', fact_loss.item(), sync_dist=True)
        #     for metric in rule_perf_metrics.keys():
        #         self.log(f'facts test_{metric}_epoch', fact_perf_metrics[metric], prog_bar=True)
        #     self.predictions = torch.stack((fact_preds, targets), dim=1)
        #     print('predictions tensor in ruletaker class, shape = {}'.format(self.predictions.shape))
        return

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.p.weight_decay,
            },
            {
                'params': [p for n, p in self.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in self.rule_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.p.weight_decay,
            },
            {
                'params': [p for n, p in self.rule_classifier.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in self.fact_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.p.weight_decay,
            },
            {
                'params': [p for n, p in self.fact_classifier.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        if self.p.optimizer == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate, eps=self.p.adam_epsilon,
                              betas=[0.9, 0.98])
        else:
            raise NotImplementedError

        if self.p.lr_scheduler == 'linear_with_warmup':
            if self.p.warmup_updates > 1.0:
                warmup_steps = int(self.p.warmup_updates)
            else:
                warmup_steps = int(self.total_steps * self.p.warmup_updates)
            print(f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps,
                                      num_training_steps=self.total_steps)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        elif self.p.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
