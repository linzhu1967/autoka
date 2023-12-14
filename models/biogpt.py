import torch
import numpy as np

from transformers import BioGptForSequenceClassification
from transformers.optimization import AdamW
import pytorch_lightning as pl

from sklearn.metrics import f1_score

from utils.utils import get_loss






class BiogptTextClassifier(pl.LightningModule):

    def __init__(self, config_dict, model_dict):
        super().__init__()
        self.model = BioGptForSequenceClassification.from_pretrained(model_dict["model_type_or_dir"], #"microsoft/biogpt"
                                                                      num_labels=model_dict["num_labels"], 
                                                                      problem_type="multi_label_classification")
        for p in self.model.biogpt.embed_tokens.parameters():
            p.requires_grad=False

        for p in self.model.biogpt.embed_positions.parameters():
            p.requires_grad=False

        for p in self.model.biogpt.layers[:23].parameters():
            p.requires_grad=False

        self.config = config_dict
        self.learning_rate = config_dict["lr"]
        self.validation_step_accuracy = []
        self.validation_step_loss = []
        self.val_labels = []
        self.val_preds = []


    def training_step(self, batch, batch_idx):
        # print("training_step")
        input_dict = {
                "attention_mask":batch['attention_mask'], 
                "input_ids":batch['input_ids'], 
            }
        predicted_logits = self.model(**input_dict).logits
        # print("predicted_logits")
        # print(predicted_logits.shape)
        # print(predicted_logits)
        # print("labels")
        # print(batch["labels"].shape)
        # print(batch["labels"])
        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(predicted_logits, batch["labels"])
        
        # print("loss", loss)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def forward(self, batch):
        # print("forward")
        input_dict = {
                "attention_mask":batch['attention_mask'], 
                "input_ids":batch['input_ids'], 
            }
        logits = self.model(**input_dict).logits
        return logits
    
    def validation_step(self, batch, batch_idx):
        # print("validation_step")
        predicted_logits = self.forward(batch)
        predicted_results = torch.max(predicted_logits, 1)[1]
        test_dict = self.acc_and_f1(predicted_results.cpu().numpy().astype(int), batch['labels'].cpu().numpy().astype(int))
        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(predicted_logits, batch["labels"]) 
        
        self.log('val_accuracy', test_dict["accuracy"])
        self.log('val_loss', loss)
        
        self.validation_step_accuracy.append(test_dict["accuracy"])
        self.validation_step_loss.append(loss.item()) 

        self.val_labels.append(batch['labels'].cpu())
        self.val_preds.append(predicted_results.cpu())

    
    def on_validation_epoch_end(self):
        val_accuracy_epoch = np.average(np.array(self.validation_step_accuracy))
        val_loss_epoch = np.average(np.array(self.validation_step_loss))

        # Concatenate predictions and labels
        val_preds = np.concatenate(self.val_preds, axis=0)
        val_labels = np.concatenate(self.val_labels, axis=0)
        # Calculate F1-macro score
        f1_macro = f1_score(val_labels, val_preds, average='macro')
        
        self.log('val_accuracy_epoch', val_accuracy_epoch)
        self.log('val_f1_macro_epoch', f1_macro)
        self.log('val_loss_epoch', val_loss_epoch)
        print("val_accuracy_epoch", val_accuracy_epoch)
        print("val_f1_macro_epoch", f1_macro)
        print("val_loss_epoch", val_loss_epoch)

    
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.config["weight_decay"])
        # optimizer = torch.optim.AdamW([
        #     {'params': self.model.biogpt.layers[23].parameters(), 'lr': self.learning_rate, 'weight_decay':self.config["weight_decay"]},
        #     {'params': self.model.biogpt.encoder.layer[11:].parameters(), 'lr': self.learning_rate, 'weight_decay':self.config["weight_decay"]}  
        # ])
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, weight_decay=self.config["weight_decay"])
        return optimizer
    

    def acc_and_f1(self, preds, labels):
        # print("preds: ", preds)
        # print(type(preds))
        # print("labels: ", labels)
        # print(type(labels))
        acc = float((preds == labels).mean())
        # f1 = float(f1_score(y_true=labels, y_pred=preds, average="macro")) 
        return {
            "accuracy": acc,
            # "f1": f1,
        }

   
