import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import data_iterator,load_mnist_2d
from models import Network
from basic_layer import Linear,Tanh,Relu,Sigmoid
from argparse import ArgumentParser
from utils import cal_classification_acc,onehot_encoding,EuclideanLoss
import yaml,argparse,os

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='mnist_log/3_layer_tanh_sigmoid')
parser.add_argument('--config',type = str, default='config.yml')
parser.add_argument('--data',type=str,default='MNIST')
args = parser.parse_args()

if __name__ == '__main__':
    # init dataset
    train_data, test_data, train_label, test_label = load_mnist_2d(args.data)

    # init config
    config = yaml.load(open('config.yml','r'),Loader=yaml.SafeLoader)
    train_cfg = config['train']
    test_cfg = config['test']
    net_cfg = config['network']

    # init network
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Tanh('tanh1'))
    model.add(Linear('fc2',256 ,256,0.001))
    model.add(Tanh('tanh2'))
    model.add(Linear('fc3',256,10,0.001))
    model.add(Sigmoid('sigmoid'))
    model.train()

    model.train()

    # init loss
    loss_func = EuclideanLoss('EuclideanLoss')
    total_iteration = 0

    # init log dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_log_file_path = os.path.join(args.log_dir,train_cfg['log'])
    test_log_file_path = os.path.join(args.log_dir,test_cfg['log'])
    
    for epoch in range(train_cfg['max_epoch']):
        model.train()
        batch_acc = []
        batch_loss = []
        for iteration ,(train_input, label) in enumerate(data_iterator(train_data, train_label, train_cfg['batch_size'])):
            batch_size = train_cfg['batch_size']
            target = onehot_encoding(label,10)

            # forward
            train_output = model.forward(train_input)

            # cal loss
            loss_val = loss_func.forward(train_output, target)

            # cal accuracy
            acc_val = cal_classification_acc(train_output, label)

            # backward
            grad = loss_func.backward(train_output, target)
            model.backward(grad)
            model.update(net_cfg)

            # log record
            batch_acc.append(acc_val)
            batch_loss.append(loss_val)

            if iteration % train_cfg['record_freq'] == 0: # show evey 50 iterations
                batch_loss_mean = np.mean(batch_loss)
                batch_acc_mean = np.mean(batch_acc)
                context = '[Train] Epoch:%d , iter:%d , batch loss:%.6f , batch acc:%.6f , bz:%d'%(epoch,iteration,batch_loss_mean,batch_acc_mean,batch_size)
                f = open(train_log_file_path,'a')
                f.writelines(context+'\n')
                f.close()
                print(context)
                batch_acc = []
                batch_loss = []
            total_iteration += 1

        # test model
        test_acc_list = []
        if epoch % train_cfg['test_epoch'] == 0:
            # feeze the model
            model.eval()
            for test_input, label in data_iterator(test_data,test_label,test_cfg['batch_size']):
                batch_size = test_cfg['batch_size']

                target = onehot_encoding(label,10)
                test_output = model.forward(test_input)

                # cal accuracy
                test_acc = cal_classification_acc(test_output,label)

                # record log
                test_acc_list.append(test_acc)

            test_acc_mean = np.mean(test_acc_list)
            context = '[Test] Epoch:%d , epoch accay:%.6f , bz:%d'%(epoch,test_acc_mean,test_cfg['batch_size'])
            print(context)
            f = open(test_log_file_path,'a')
            f.writelines(context+'\n')
            f.close()


