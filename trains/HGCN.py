import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.functions import dict_to_str, js_divergence, kl_divergence, mmd_loss
from utils.metricsTop import MetricsTop
logger = logging.getLogger('MSA')

class HGCN():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        if args.loss_name == 'L2':
            self.pixelwise_loss = nn.MSELoss()
        elif args.loss_name == 'MMD':
            self.pixelwise_loss = mmd_loss
        elif args.loss_name == 'kl':
            self.pixelwise_loss = kl_divergence
        elif args.loss_name == 'js':
            self.pixelwise_loss = js_divergence

        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        if self.args.use_bert_finetune:
            # OPTIMIZER: finetune Bert Parameters.
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.Model.TGCN.text_model.named_parameters())

            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
                 'lr': self.args.learning_rate_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
                {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
                 'lr': self.args.learning_rate_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.decay)

        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            # train_loss = 0.0
            avg_trloss = []
            avg_mrloss = []
            avg_frloss = []
            avg_closs = []

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    # 目前提出的模型只能使用含有数据增强的数据。
                    vision_lm = batch_data['vision_lm'].to(self.args.device)
                    audio_lm = batch_data['audio_lm'].to(self.args.device)
                    text_lm = batch_data['text_lm'].to(self.args.device)

                    # encoder-decoder
                    optimizer.zero_grad()
                    proj_l, proj_a, proj_v = model.Model.TGCN(text, audio, vision)
                    proj_l_all, proj_a_all, proj_v_all = model.Model.MGCN(proj_l, proj_a, proj_v)

                    proj_l_lm, proj_a_lm, proj_v_lm = model.Model.TGCN(text_lm, audio_lm, vision_lm)
                    proj_l_all_lm, proj_a_all_lm, proj_v_all_lm = model.Model.MGCN(proj_l_lm, proj_a_lm, proj_v_lm)

                    fusion_feature_x = model.Model.fusion(proj_l_all, proj_a_all, proj_v_all)
                    fusion_feature_lm = model.Model.fusion(proj_l_all_lm, proj_a_all_lm, proj_v_all_lm)

                    output_x = model.Model.classifier(fusion_feature_x)
                    y_pred.append(output_x.cpu())
                    y_true.append(labels.cpu())
                    output_lm = model.Model.classifier(fusion_feature_lm)
                    y_pred.append(output_lm.cpu())
                    y_true.append(labels.cpu())
                    # 分类loss
                    c_loss = self.criterion(output_x, labels) + self.criterion(output_lm, labels)
                    avg_closs.append(c_loss.item())

                    # TGCN loss
                    trl = self.pixelwise_loss(proj_l.detach(), proj_l_lm) + self.pixelwise_loss(proj_a.detach(), proj_a_lm) \
                         + self.pixelwise_loss(proj_v.detach(), proj_v_lm)
                    avg_trloss.append(trl.item())

                    # MGCN loss
                    mrl = self.pixelwise_loss(proj_l_all.detach(), proj_l_all_lm) + self.pixelwise_loss(proj_a_all.detach(), proj_a_all_lm) \
                          + self.pixelwise_loss(proj_v_all.detach(), proj_v_all_lm)
                    avg_mrloss.append(mrl.item())

                    # Fusion loss
                    frl = self.pixelwise_loss(fusion_feature_x.detach(), fusion_feature_lm)
                    avg_frloss.append(frl.item())
                    tot_loss = self.args.weight * (self.args.alpha * frl + (1-self.args.alpha) * (trl + mrl)) + c_loss
                    tot_loss.backward()

                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                        self.args.grad_clip)
                    optimizer.step()

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> closs: %.4f trloss: %.4f mrloss: %.4f frloss: %.4f %s"
                        % (self.args.modelName, epochs - best_epoch, epochs, self.args.cur_time,
                           np.mean(avg_closs), np.mean(avg_trloss), np.mean(avg_mrloss), np.mean(avg_frloss), dict_to_str(train_results)))
            # validation
            val_results = self.do_valid(model, dataloader['valid'])
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_valid(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # 目前提出的模型只能使用含有数据增强的数据。
                    vision_lm = batch_data['vision_lm'].to(self.args.device)
                    audio_lm = batch_data['audio_lm'].to(self.args.device)
                    text_lm = batch_data['text_lm'].to(self.args.device)

                    proj_l, proj_a, proj_v = model.Model.TGCN(text, audio, vision)
                    proj_l_all, proj_a_all, proj_v_all = model.Model.MGCN(proj_l, proj_a, proj_v)
                    fusion_feature_x = model.Model.fusion(proj_l_all, proj_a_all, proj_v_all)

                    proj_l_lm, proj_a_lm, proj_v_lm = model.Model.TGCN(text_lm, audio_lm, vision_lm)
                    proj_l_all_lm, proj_a_all_lm, proj_v_all_lm = model.Model.MGCN(proj_l_lm, proj_a_lm, proj_v_lm)
                    fusion_feature_lm = model.Model.fusion(proj_l_all_lm, proj_a_all_lm, proj_v_all_lm)

                    output_x = model.Model.classifier(fusion_feature_x)
                    loss = self.criterion(output_x, labels)
                    y_pred.append(output_x.cpu())
                    y_true.append(labels.cpu())
                    output_lm = model.Model.classifier(fusion_feature_lm)
                    loss += self.criterion(output_lm, labels)
                    y_pred.append(output_lm.cpu())
                    y_true.append(labels.cpu())

                    eval_loss += loss.item()

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName + '-' + self.args.augment, dict_to_str(eval_results)))
        return eval_results
