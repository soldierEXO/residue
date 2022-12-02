import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT import GATModel
from utils.data_processing import load_data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

def train(args):
    threshold = args.d

    # loading and spliting data
    if args.val == "": 
        
        fasta_path = args.train #l包含label
        feature_dir = args.ft_dir
        data_list, labels_list = load_data(fasta_path, feature_dir, threshold)

        data_train, data_val, _, _ = train_test_split(data_list, labels_list, test_size=0.2, shuffle=True, random_state=41)
    else:
        fasta_path_train = args.train
        fasta_path_val = args.val
        feature_dir = args.ft_dir
        data_train, labels_list = load_data(fasta_path_train, feature_dir, threshold)
        data_val, labels_list = load_data(fasta_path_val, feature_dir, threshold)

        data_train = shuffle(data_train, random_state = 1024)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    node_feature_dim = data_train[0].x.shape[1]
    n_class = 2

    # tensorboard, record the change of auc, acc and loss
    writer = SummaryWriter()


    if args.pretrained_model == "":
        model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device)
    else:
        model = torch.load(args.pretrained_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(data_train, batch_size=args.b)
    val_dataloader = DataLoader(data_val, batch_size=args.b)

    best_acc = 0
    best_auc = 0
    min_loss = 1000
    save_acc = '/'.join(args.save.split('/')[:-1]) + '/acc_' + args.save.split('/')[-1]
    save_auc = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1]
    save_loss = '/'.join(args.save.split('/')[:-1]) + '/loss_' + args.save.split('/')[-1]
    for epoch in range(args.e):
        print('Epoch ', epoch)
        model.train()
        arr_loss = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            out = output[0]
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        
        avgl = np.mean(arr_loss)
        print("Training Average loss :", avgl)

        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            preds = []
            y_true = []
            arr_loss = []
            for data in val_dataloader:
                data = data.to(device)

                output = model(data.x, data.edge_index, data.batch)
                out = output[0]

                loss = criterion(out, data.y)
                arr_loss.append(loss.item())

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]
                correct = (pred == data.y).sum().float()
                total_correct += correct
                total_num += data.num_graphs
                preds.extend(score.cpu().detach().data.numpy())
                y_true.extend(data.y.cpu().detach().data.numpy())

            acc = (total_correct / total_num).cpu().detach().data.numpy()
            auc = roc_auc_score(y_true, preds)
            val_loss = np.mean(arr_loss)
            print("Validation accuracy: ", acc)
            print("Validation auc:", auc)
            print("Validation loss:", val_loss)

            writer.add_scalar('Loss', avgl, global_step=epoch)
            writer.add_scalar('acc', acc, global_step=epoch)
            writer.add_scalar('auc', auc, global_step=epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model, save_acc)

            if auc > best_auc:
                best_auc = auc
                torch.save(model, save_auc)

            if np.mean(val_loss) < min_loss:
                min_loss = val_loss
                torch.save(model, save_loss)

            print('-' * 50)

        scheduler.step()

    print('best acc:', best_acc)
    print('best auc:', best_auc)
    if args.o is not None:
        with open(args.o, 'a') as f:
            localtime = time.asctime(time.localtime(time.time()))
            f.write(str(localtime) + '\n')
            f.write('args: ' + str(args) + '\n')
            f.write('auc result: ' + str(best_auc) + '\n\n')

class args_func():
    pos_t = 'sAMPpred-GAT-main\\example\\positive\\example_pos.fasta'
    pos_v = 'sAMPpred-GAT-main\\example\\positive\\example_pos.fasta'
    pos_npz = 'sAMPpred-GAT-main\\example\\positive\\npz\\'
    neg_t = 'sAMPpred-GAT-main\\example\\negative\\example_neg.fasta'
    neg_v = 'sAMPpred-GAT-main\\example\\negative\\example_neg.fasta'
    neg_npz = 'sAMPpred-GAT-main\\example\\negative\\npz\\'

    lr = 0.0001
    drop = 0.3
    e = 10
    b = 1
    hd = 64

    pretrained_model = ""
    save = 'sAMPpred-GAT-main\\saved_models\\example.model'
    heads = 8

    d = 37
    o = 'log.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input file
    parser.add_argument('-train', type=str, default='/mnt/home/wanging/sAMPpred-GAT-main/datasets/datasets/seq/graphbind/DRNA-962_Train.text',
                        help='Path of the training dataset')
    parser.add_argument('-val', type=str, default='',
                        help='Path of the validation dataset')
    parser.add_argument('-feature_dir', type=str, default='/mnt/home/wanging/sAMPpred-GAT-main/datasets/datasets/seq/DNA_Train/',
                        help='Path of the npz folder, which saves the predicted structure')

    # 0.001 for pretrain， 0.0001 or train
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate') 
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-e', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('-b', type=int, default=1, help='Batch size')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')

    parser.add_argument('-pretrained_model', type=str, default="",
                        help='The path of pretraining model, if "", the model will be trained from scratch')
    parser.add_argument('-save', type=str, default='saved_models/samp.model',
                        help='The path saving the trained models')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')

    parser.add_argument('-d', type=int, default=37, help='Distance threshold to construct a graph, 0-37, 37 means 20A')
    parser.add_argument('-o', type=str, default='log.txt', help='File saving the raw prediction results')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    train(args)

    end_time = datetime.datetime.now()
    print('End time(min):', (end_time - start_time).seconds / 60)
