import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataset import data_iterator,read_data
from models import Network
from basic_layer import Linear,Tanh,Relu,Sigmoid
from argparse import ArgumentParser
from utils import MSE_loss,cal_acc
import yaml,argparse,os

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='2_layer_log_tanh')
parser.add_argument('--data',type=str,default='a1b2c2d2.csv')
args = parser.parse_args()

if __name__ == '__main__':
    data = read_data(args.data)
    data =np.array(data)
    x,y_label = data[:,0],data[:,1]
    x_train,x_test,y_train,y_test = train_test_split(x,y_label,test_size = 0.1)

    config = yaml.load(open('config.yml','r'),Loader=yaml.SafeLoader)
    train_cfg = config['train']
    test_cfg = config['test']
    net_cfg = config['network']

    model = Network()
    model.add(Linear('fc1', 1, 128, 0.001))
    model.add(Tanh('tanh1'))
    model.add(Linear('fc2',128,1,0.001))
    model.train()

    loss = MSE_loss(name='MSE_loss')
    loss_list = []
    total_iteration = 0

    train_log_file_path = os.path.join(args.log_dir,train_cfg['log'])
    test_log_file_path = os.path.join(args.log_dir,test_cfg['log'])
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    for epoch in range(train_cfg['max_epoch']):
        # activate the model
        model.train()

        batch_acc = []
        batch_loss = []
        iteration = 0
        train_input_array = np.array([])
        train_res_array = np.array([])

        for train_input, label in data_iterator(x_train,y_train, train_cfg['batch_size']):
            batch_size = train_cfg['batch_size']
            train_input = train_input.reshape(train_input.shape[0],1)
            label = label.reshape(label.shape[0],1)

            # forward
            output = model.forward(train_input)

            # cal loss
            loss_val = loss.forward(output, label)

            # backward
            grad = loss.backward(output,label)
            model.backward(grad)
            model.update(net_cfg)

            # cal accuracy
            acc_val = cal_acc(output, label)

            # record the log
            loss_list.append(x)
            batch_loss.append(loss_val)
            batch_acc.append(acc_val)
            if train_input_array.size == 0:
                train_input_array = train_input.copy()
            else:
                np.concatenate((train_input_array, train_input),axis = 0)
            if train_res_array.size == 0:
                train_res_array = output.copy()
            else:
                np.concatenate((train_res_array, output),axis = 0)

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
            iteration += 1
            total_iteration += 1

        if epoch % train_cfg['test_epoch'] == 0:
            # plot on train set
            plt.subplots_adjust(hspace = 0.5)
            plt.subplot(2,1,1)
            plt.scatter(x_train, y_train,s=1,c='b')
            plt.scatter(train_input_array,train_res_array,s=1,c='r')
            plt.title('Loss:%.3f'%(batch_loss_mean))

            # feeze the model
            model.eval()

            # test the model
            test_input_array = np.array([])
            test_res_array = np.array([])
            test_acc_list = []
            for test_input, label in data_iterator(x_test,y_test, test_cfg['batch_size']):
                batch_size = test_cfg['batch_size']
                test_input = test_input.reshape(test_input.shape[0],1)
                label = label.reshape(label.shape[0],1)

                # forward
                output = model.forward(test_input)

                # cal accuracy
                test_acc = cal_acc(output, label)
                test_acc_list.append(test_acc)

                if test_input_array.size == 0:
                    test_input_array = test_input.copy()
                else:
                    test_input_array = np.concatenate((test_input_array,test_input),axis = 0)
                if test_res_array.size == 0:
                    test_res_array =  output.copy()
                else:
                    test_res_array = np.concatenate((test_res_array,output),axis=0)
            test_accuracy = np.mean(test_acc_list)
            context = '[Test] Epoch:%d , epoch accay:%.6f , bz:%d'%(epoch,test_accuracy,test_cfg['batch_size'])
            print(context)
            f = open(test_log_file_path,'a')
            f.writelines(context+'\n')
            f.close()

            # plot on test set
            test_plot_folder = os.path.join(args.log_dir,'test_plot')
            if not os.path.exists(test_plot_folder):
                os.makedirs(test_plot_folder)
            plot_save_path = os.path.join(test_plot_folder,'epoch%diter%d.jpg'%(epoch,iteration))
            plt.subplot(2,1,2)
            plt.scatter(test_input_array,test_res_array,s=1,c='r')
            plt.scatter(x_test,y_test,s=1,c='b')
            plt.title('Test accuracy:%.3f'%(test_accuracy))
            plt.savefig(plot_save_path)

            plt.close()


            








