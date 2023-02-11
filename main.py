import os
import sys
import time
import datetime
import shutil
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from utils import utils, parser
from data import loader
import matplotlib.pyplot as plt
from models import MMDLNetV0

class Run(object):
    def __init__(self, yaml_dir):
        self.yaml_dir = yaml_dir
        self.args = EasyDict(parser.get_parser(self.yaml_dir))
        self.current_step = 0
        self.current_test_step = 0
        self.last_map = 0
        if torch.cuda.is_available() and not self.args.gpu is None:
            self.args.device = torch.device('cuda:%s'%(self.args.gpu))
            cudnn.benchmark = True
            cudnn.enabled = True
        else:
            self.args.device = torch.device('cpu')

        utils.record_config(self.args)
        now = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        if self.args.test_only:
            self.logger = utils.get_logger(os.path.join(self.args.job_dir, 'logger'+now+'_Test.log'))
        else:
            self.logger = utils.get_logger(os.path.join(self.args.job_dir, 'logger'+now+'_Train.log'))
        self.logger.info('args = %s', self.args)
        self.model = eval(self.args.model)(self.args).to(self.args.device)
        # torch.cuda.manual_seed(42)
        self.logger.info(self.model)

        self.optimizer = parser.get_optimizer(self.args, self.model)
        self.scheduler = parser.get_scheduler(self.args, self.optimizer)

        self.train_data = loader(self.args, split='train')
        self.valid_data = loader(self.args, split='test')
        self.train_loader = DataLoader(self.train_data, batch_size=int(self.args.batch_size), num_workers=int(self.args.num_workers), shuffle=True, pin_memory=True, drop_last=True)
        self.valid_loader = DataLoader(self.valid_data, batch_size=int(self.args.batch_size), num_workers=int(self.args.num_workers), shuffle=False, pin_memory=True, drop_last=True)
        # print('over')
        self.writer = SummaryWriter('./vlog')
        self.x_axis_data = [i for i in range(self.args.epochs)]
        self.train_loss = []
        self.test_loss = []


    def train(self, epoch):
        self.model.train()
        for param_group in self.optimizer.param_groups:
            cur_lr = param_group['lr']
        self.logger.info('learning_rate: '+str(cur_lr))
        start = time.time()
        num_iter = len(self.train_loader)
        total_loss, total_diff, total_simi, total_cls,total_mse, total_map = 0., 0., 0., 0., 0., 0.
        
        for i, (scene, optical, audio, label) in enumerate(self.train_loader):
            scene = Variable(scene.to(self.args.device))
            optical = Variable(optical.to(self.args.device))
            audio = Variable(audio.to(self.args.device))
            label = Variable(label.to(self.args.device))
            
            output, loss_cls = self.model(scene, optical, audio, label)
            
            loss = loss_cls
            total_cls += loss_cls
            total_loss += loss
            map = utils.cal_ap(output, label).mean()
            total_map += map
            self.optimizer.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            self.optimizer.step()

            self.current_step += 1
            if (i+1) % int(self.args.print_freq) == 0:
                self.logger.info('Train Epoch[{0}]({1}/{2}): '
                                 'Loss_Cls: {loss_cls:.4f}, '
                                 'Total_Loss: {total_loss:.4f}, '
                                 'mAP: {total_map:.4f}, '
                                 'Time: {cost_time:.4f}. '.format(epoch, int(i+1), num_iter, loss_cls=total_cls/(i+1), \
                                                                  total_loss=total_loss/(i+1),\
                                                                   total_map=total_map/(i+1), cost_time=(time.time()-start)/(i+1)))

        average_loss, average_map = total_loss/(i+1), total_map/(i+1)
        average_diff, average_cls = total_diff/(i+1), total_cls/(i+1)
        average_simi, average_mse = total_simi/(i+1), total_mse/(i+1)

        self.train_loss.append(average_loss.data.cpu().numpy())
        self.writer.add_scalar('train_loss', average_loss, epoch)
        self.writer.add_scalar('train_mAP', average_map, epoch)
        self.logger.info('Train Epoch[{0}] : '
                         'Loss_Cls: {loss_cls:.4f}, '
                         'Loss_diff: {loss_diff:.4f}, '
                         'Loss_simi: {loss_simi:.4f}, '
                         'Loss_mse: {loss_mse:.4f}, '
                         'Total_Loss: {total_loss:.4f}, '
                         'mAP: {total_map:.4f}, '
                         'Time: {cost_time:.4f}. '.format(epoch, loss_cls=average_cls, loss_diff=average_diff, loss_simi=average_simi, loss_mse=average_mse, total_loss=average_loss, \
                                                           total_map=average_map, cost_time=time.time() - start))

    def valid(self, epoch):
        self.model.eval()
        start = time.time()
        total_loss, total_diff, total_simi, total_cls,total_mse, total_map = 0., 0., 0., 0., 0., 0.
        with torch.no_grad():
            num_iter = len(self.valid_loader)
            for i, (scene, optical, audio, label) in enumerate(self.valid_loader):
                scene = scene.to(self.args.device)
                optical = optical.to(self.args.device)
                audio = audio.to(self.args.device)
                label = label.to(self.args.device)
                output, loss_cls = self.model(scene, optical, audio, label)
                loss = loss_cls
                total_cls += loss_cls
                total_loss += loss
                map = utils.cal_ap(output, label).mean()
                total_map += map

                self.current_test_step += 1

        average_loss = total_loss / (i+1)
        average_diff = total_diff / (i + 1)
        average_simi = total_simi / (i + 1)
        average_mse = total_mse / (i + 1)
        average_cls = total_cls / (i+1)
        average_map = total_map / (i+1)
        self.test_loss.append(average_loss.data.cpu().numpy())

        self.writer.add_scalar('valid_loss', average_loss, epoch)
        self.writer.add_scalar('valid_mAP', average_map, epoch)

        self.logger.info('Valid Epoch[{0}] : '
                         'Loss_Cls: {loss_cls:.4f}, '
                         'Loss_diff: {loss_diff:.4f}, '
                         'Loss_simi: {loss_simi:.4f}, '
                         'Loss_mse: {loss_mse:.4f}, '
                         'Total_Loss: {total_loss:.4f}, '
                         'mAP: {total_map:.4f}, '
                         'Time: {cost_time:.4f}. '.format(epoch, loss_cls=average_cls, loss_diff=average_diff, loss_simi=average_simi, loss_mse=average_mse, total_loss=average_loss, \
                                                           total_map=average_map, cost_time=time.time() - start))
        return average_loss, average_map

    def test(self):
        self.valid_data = loader(self.args, split='test')
        self.valid_loader = DataLoader(self.valid_data, batch_size=int(self.args.batch_size),
                                       num_workers=int(self.args.num_workers), shuffle=False, pin_memory=True,
                                       drop_last=False)

        self.logger.info('Test from the best model ...')
        checkpoint = torch.load(os.path.join(self.args.resume_dir, self.args.checkpoint))
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        total_map, total_one_error, total_coverage, total_ranking, total_hamming = 0., 0., 0., 0., 0.
        start = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (scene, optical, audio, label) in enumerate(self.valid_loader):
                scene = scene.to(self.args.device)
                optical = optical.to(self.args.device)
                audio = audio.to(self.args.device)
                label = label.to(self.args.device)
                logit, loss_cls = self.model(scene, optical, audio, label)
                total_map += utils.cal_ap(logit, label).mean()
                total_one_error += utils.cal_one_error(logit, label)
                total_coverage += utils.cal_coverage(logit, label)
                total_ranking += utils.cal_RankingLoss(logit, label)
                total_hamming += utils.cal_HammingLoss(logit, label)
        average_map = total_map / (i+1)
        average_one_error = total_one_error / (i+1)
        average_coverage = total_coverage / (i+1)
        average_ranking = total_ranking / (i+1)
        average_hamming = total_hamming / (i+1)

        self.logger.info('=> Test all indexes: '
                         'mAP: {map: .4f}, '
                         'one_error: {one_error:.4f}, '
                         'coverage: {coverage:.4f}, '
                         'ranking_loss: {ranking:.4f}, '
                         'hamming_loss: {hamming:.4f}, '
                         'Time: {cost_time:.4f} '.format(map=average_map, one_error=average_one_error, \
                                                               coverage=average_coverage, ranking=average_ranking, \
                                                               hamming=average_hamming, cost_time=time.time()-start))

    def run(self):
        if self.args.test_only:
            self.test()
        else:
            start_epoch = 0
            best_map = 0.
            last_map = 0.
            if self.args.use_resume:
                self.logger.info('Resume from pretrained model: %s ...' % (self.args.checkpoint) )
                checkpoint = torch.load(os.path.join(self.args.resume_dir, self.args.checkpoint))
                resume_dict = checkpoint['state_dict']
                optim_dict = checkpoint['optimizer']
                start_epoch = int(checkpoint['epoch']+1)
                best_map = float(checkpoint['best_map'])
                state_dict = utils.load_state_dict(self.model, resume_dict)
                self.model.load_state_dict(state_dict)
                print('Load model parameter successfully ......')
                self.optimizer.load_state_dict(optim_dict)
                print('Load optimizer parameter successfully ......')

            epoch = start_epoch
            while epoch < self.args.epochs:
                epoch_time = time.time()
                self.train(epoch=epoch)
                valid_loss, valid_map = self.valid(epoch=epoch)
                is_best = False
                if valid_map > best_map:
                    best_map = valid_map
                    is_best = True

                state_dict = {
                    'epoch' : epoch,
                    'state_dict' : self.model.state_dict(),
                    'best_map': best_map,
                    'optimizer': self.optimizer.state_dict()
                }

                utils.save_checkpoint(epoch, state_dict, is_best, self.args.job_dir)
                if self.args.scheduler[0] == 'ReduceLROnPlateau':
                    self.scheduler.step(best_map)
                else:
                    self.scheduler.step()

                if self.args.epochs - epoch <= 10:
                    self.last_map += valid_map
                    if valid_map > last_map: # save best model among
                        last_map = valid_map


                epoch += 1
                self.logger.info('=> Best index: '
                                 'mAP: {best_map:.4f}. '
                                 'Time: {cost_time: .4f} '.format(best_map=best_map, cost_time=time.time()-epoch_time))

            self.logger.info('=> Last mAP_avg: '
                             'last_mAP: {last_mAP:.4f}. '.format(last_mAP=self.last_map/10))
            filename = os.path.join(self.args.job_dir, 'best_model_%.2f.pth.tar'%(best_map*100))
            shutil.copyfile(os.path.join(self.args.job_dir, 'best_model.pth.tar'), filename)

            # plt.plot(self.x_axis_data, np.array(self.train_loss), 'r-', label='Training Set')
            # plt.plot(self.x_axis_data, np.array(self.test_loss), 'b-', label='Test Set')
            # plt.show()

        self.writer.close()


if __name__ == '__main__':
    app = Run('./libs/MMDLNetV0/mmdlnetv0_base.yaml')
    # app = Run(sys.argv[1])
    app.run()
    print('train over')