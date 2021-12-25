import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='train',help='test or train,ex: --mode train')
parser.add_argument('--log',type = str, default='mnist_log/4_layer_tanh_sigmoid/train.log')
args = parser.parse_args()

def read_trainlog(logpath):

    epoch_list =  []
    iter_list = []
    loss_list = []
    acc_list = []
    batch_size = None
    for line in open(logpath,'r').readlines():
        line = line.strip()
        spt = line.split(' , ')
        epoch_list.append(int(spt[0].split(':')[-1]))
        iter_list.append(int(spt[1].split(':')[-1]))
        loss_list.append(float(spt[2].split(':')[-1]))
        acc_list.append(float(spt[3].split(':')[-1]))
        batch_size = int(spt[4].split(':')[-1])
    return epoch_list,iter_list,loss_list,acc_list,batch_size


def read_testlog(logpath):
    epoch_list =  []
    acc_list = []
    batch_size = None
    for line in open(logpath,'r').readlines():
        line = line.strip()
        spt = line.split(' , ')
        epoch_list.append(int(spt[0].split(':')[-1]))
        acc_list.append(float(spt[1].split(':')[-1]))
        batch_size = int(spt[2].split(':')[-1])

    return epoch_list,acc_list,batch_size

if __name__ == '__main__':
    data = None
    if args.mode == 'train':
        epoch_list,iter_list,loss_list,acc_list,batch_size = read_trainlog(args.log)
        max_epoch = np.max(epoch_list)+1
        acc_epoch_array = np.zeros((max_epoch,1))
        loss_epoch_array = np.zeros((max_epoch,1))
        for i,epoch in enumerate(epoch_list):
            acc_epoch_array[epoch] += acc_list[i]
            loss_epoch_array[epoch] += loss_list[i]
        iter_num = np.max(iter_list)
        acc_epoch_array /= iter_num
        loss_epoch_array /= iter_num

        color_set = 'r' if 'relu' in args.log else 'b'

        # plot
        epoch_array = np.linspace(start=0,stop= max_epoch-1,num = max_epoch ).reshape(-1,1)
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(2,1,1)
        plt.plot(epoch_array,acc_epoch_array,c=color_set)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Train accuracy')
        plt.subplot(2,1,2)
        plt.plot(epoch_array,loss_epoch_array,c=color_set)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train loss')
        plt.show()

    elif args.mode == 'test':
        epoch_list,acc_list,batch_size =read_testlog(args.log)

        color_set = 'r' if 'relu' in args.log else 'b'
        
        # plot
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(1,1,1)
        plt.plot(epoch_list,acc_list,c=color_set)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Test accuracy')
        plt.show()

    print('ggpd')

    


    