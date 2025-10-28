import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision import transforms


class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, train_data_loader, \
                valid_data_loader=None, metric=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.metric = metric
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug

        self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.logger = logger
        self.writer = writer

        # 增加 history 字典，用于记录每个 epoch 的 train/val loss 和 train/val accuracy
        # 方便后续在外部调用或绘图。
        self.history = {
                    'train_loss': [],  # 存放每个 epoch 的训练集平均 loss
                    'train_acc': [],  # 存放每个 epoch 的训练集 top1 accuracy
                    'val_loss': [],  # 存放每个 epoch 的验证集平均 loss
                    'val_acc': []  # 存放每个 epoch 的验证集 top1 accuracy
        }

        self.early_stop_patience = 50
        self.no_improve_count = 0
        self.best_model_path = './checkpoint/' + model_type + '/best_model.ckpt'

        assert self.logger is not None, "Logger was not passed into Trainer"

    def fit(self):
        for epoch in range(self.start_epoch):
            self.lr_schedule.step()

        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.logger.append(f'Epoch {epoch}/{self.num_epochs - 1}')
                self.logger.append('-' * 60)
                self.cur_epoch = epoch
                self.lr_schedule.step()

                if self.is_debug:
                    self._dump_infos()

                self._train()
                self._valid()
        except StopIteration:
            self.logger.append("Training stopped early due to no improvement.")

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('best loss: %f' % (self.best_loss))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()  # Set model to training mode
        losses = []
        if self.metric is not None:
            self.metric[0].reset()

        for i, (inputs, labels) in enumerate(self.train_data_loader):              # Notice
            if self.is_use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.squeeze()
            else:
                labels = labels.squeeze()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)            # Notice 
            loss = self.loss_fn[0](outputs, labels)
            if self.metric is not None:
                prob     = F.softmax(outputs, dim=1).data.cpu()
                self.metric[0].add(prob, labels.data.cpu())
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())       # Notice
            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                # batch_mean_loss  = np.mean(losses)
                # print_str = '[%s]\tTraining Batch[%d/%d]\t Class Loss: %.4f\t'           \
                #             % (local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss)
                # if i == len(self.train_data_loader) - 1 and self.metric is not None:
                #     top1_acc_score = self.metric[0].value()[0]
                #     # top5_acc_score = self.metric[0].value()[1]
                #     print_str += '@Top-1 Score: %.4f\t' % (top1_acc_score)
                #     # print_str += '@Top-5 Score: %.4f\t' % (top5_acc_score)
                # self.logger.append(print_str)
            # 当前 epoch 截止到第 i 个 batch 的平均 loss
            batch_mean_loss = np.mean(losses)
            # 将 loss 打印为 8 位小数
            print_str = '[%s]\tTraining Batch[%d/%d]\t Class Loss: %.8f\t' \
                                     %(local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss)

            if i == len(self.train_data_loader) - 1 and self.metric is not None:
                top1_acc_score = self.metric[0].value()[0]
                # top5_acc_score = self.metric[0].value()[1]
                # 同样把 accuracy 格式化为 8 位小数
                print_str += '@Top-1 Score: %.8f\t' % (top1_acc_score)
                # print_str += '@Top-5 Score: %.8f\t' % (top5_acc_score)
                self.logger.append(print_str)
        # trainer.py 中
        if self.writer is not None:
            self.writer.add_scalar('loss/loss_c', batch_mean_loss, self.cur_epoch)
          # ========== 这一轮 (_train) 结束后，记录 epoch-level 的 train loss & train acc ==========
          # 1. 本 epoch 的平均 loss（取所有 batch loss 的平均值）
        epoch_train_loss = np.mean(losses)
          # 2. 本 epoch 的 train top-1 accuracy，依赖 metric[0]

        if self.metric is not None:
            epoch_train_acc = self.metric[0].value()[0]
        else:
            epoch_train_acc = 0.0
        # 存入 history
        self.history['train_loss'].append(epoch_train_loss)
        self.history['train_acc'].append(epoch_train_acc)

    def _valid(self):
        self.model.eval()
        losses = []
        all_preds, all_labels = [], []
        if self.metric is not None:
            self.metric[0].reset()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.valid_data_loader):
                if self.is_use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.squeeze()
                else:
                    labels = labels.squeeze()

                outputs = self.model(inputs)
                loss = self.loss_fn[0](outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if self.metric is not None:
                    self.metric[0].add(probs.data.cpu(), labels.data.cpu())
                losses.append(loss.item())

        batch_mean_loss = np.mean(losses)
        top1_acc_score = self.metric[0].value()[0] if self.metric is not None else 0.0
        # top5_acc_score = self.metric[0].value()[1]

        print_str = '[Validation] Loss: %.8f, @Top-1: %.8f' % (
            batch_mean_loss, top1_acc_score)
        self.logger.append(print_str)

        # 保存 best 模型
        if top1_acc_score >= self.best_acc:
            self.best_acc = top1_acc_score
            self.best_loss = batch_mean_loss
            self.no_improve_count = 0
            self._save_best_model()
        else:
            self.no_improve_count += 1

        # 可视化混淆矩阵
        self._log_confusion_matrix(all_labels, all_preds)

        # ========== 这一轮 (_valid) 结束后，记录 epoch-level 的 val loss & val acc ==========
        self.history['val_loss'].append(batch_mean_loss)
        self.history['val_acc'].append(top1_acc_score)

        # Early Stopping
        if self.no_improve_count >= self.early_stop_patience:
            self.logger.append("Early stopping triggered.")
            raise StopIteration

    def _log_confusion_matrix(self, targets, preds):
        if self.writer is None:
            return

        cm = confusion_matrix(targets, preds)
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['NP', 'P'], yticklabels=['NP', 'P'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # 转成 image 并写入 tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = transforms.ToTensor()(image)
        self.writer.add_image('Confusion_Matrix', image, self.cur_epoch)
        plt.close()

    # def _save_best_model(self):
    #     # Save Model
    #     self.logger.append('Saving Model...')
    #     state = {
    #         'state_dict': self.model.state_dict(),
    #         'best_acc': self.best_acc,
    #         'cur_epoch': self.cur_epoch,
    #         'num_epochs': self.num_epochs
    #     }
    #     if not os.path.isdir('./checkpoint/' + self.model_type):
    #         os.makedirs('./checkpoint/' + self.model_type)
    #     torch.save(state, './checkpoint/' + self.model_type + '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')   # Notice
    def _save_best_model(self):
        self.logger.append('Saving Best Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch,
            'num_epochs': self.num_epochs
        }
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        torch.save(state, self.best_model_path)
