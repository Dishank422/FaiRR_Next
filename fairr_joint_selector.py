'''
This script is the joint selector component of FaiRR_Next. It selects a rule from a given theory and statement and a set of facts from all the facts.
'''

from helper import *
from basemodel import BaseModel


class FaiRRJointSelector(BaseModel):
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
        self.dropout    = torch.nn.Dropout(self.text_encoder.config.hidden_dropout_prob)

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


