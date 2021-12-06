#encoding:utf-8
from pathlib import Path
from visdom import Visdom
import tensorboardX
import time
class Logger(object):
    def __init__(self,vis=True):
        self.dir='../exp/log'
        self.train_file=f'{self.dir}/train.txt'
        self.test_file=f'{self.dir}/test.txt'
        self.time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.save_dir=f'{self.dir}/{self.time_str}'
        self.check_dir(self.save_dir)
        self.writer = tensorboardX.SummaryWriter(log_dir=self.save_dir)
        self.vis=vis
        if self.vis:
            self.viz=Visdom(server='10.5.133.209',port=8097,env='track')
        # self.viz.line([0.], [0], win='train/loss', opts=dict(title='train/loss'))
        # self.viz.line([0.], [0], win='train/hm_loss', opts=dict(title='train/hm_loss'))
        # self.viz.line([0.], [0], win='train/id_loss', opts=dict(title='train/id_loss'))
        # self.viz.line([0.], [0], win='train/off_loss', opts=dict(title='train/off_loss'))
        # self.viz.line([0.], [0], win='train/wh_loss', opts=dict(title='train/wh_loss'))
        # self.viz.line([0.], [0], win='test/loss', opts=dict(title='test/loss'))
        # self.viz.line([0.], [0], win='test/hm_loss', opts=dict(title='test/hm_loss'))
        # self.viz.line([0.], [0], win='test/id_loss', opts=dict(title='test/id_loss'))
        # self.viz.line([0.], [0], win='test/off_loss', opts=dict(title='test/off_loss'))
        # self.viz.line([0.], [0], win='test/wh_loss', opts=dict(title='test/wh_loss'))
    def check_dir(self,path):
        if not Path(path).exists():
            Path(path).mkdir()
    def write(self,phase,loss_stats,epochs,num_iters,batch_i,flag=True):
        niters=(epochs-1)*num_iters+batch_i+1
        if flag:
            for k, v in loss_stats.items():
                self.writer.add_scalar(f'{phase}/{k}', v.item(), niters)
                if self.vis:
                    self.viz.line([v.item()], [niters], win=f'{phase}/{k}', update='append',opts=dict(title=f'{phase}/{k}'))
        else:
            for k, v in loss_stats.items():
                self.writer.add_scalar(f'{phase}/{k}', v, niters)
                if self.vis:
                    self.viz.line([v], [niters], win=f'{phase}/{k}', update='append',opts=dict(title=f'{phase}/{k}'))
        # self.writer.add_scalars(phase,loss_stats,niters)