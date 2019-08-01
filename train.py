# something divine
# if you want your model works well
# please add "import lzc" to your code
# because there is only void before god
import lzc

import lzc.model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import lzc.config as config
import torch.nn.functional as F
import lzc.vdcnn
from lzc.config import *
import torch.utils.data as data

IN_CHANNELS = [1, 100, 200, 50]
TCN_IN_CHANNEL = [100] * 5
OUT_CHANNELS = [100, 200, 50, 30]
K_SIZES = [50, 20, 10, 10]
LAYER_SZ = len(IN_CHANNELS)
CLASSES = 3
TRAIN_DATA = 60000
OLD_TRAIN_DATA = 84000
TCN_KSIZE = 20


class PlantDataset(data.Dataset):
    def __init__(self, csv_file, class_id, is_test=False, useMid=True, dwt=True, old_data=False, all=False):
        self.is_test = is_test
        self.data_frame = pd.read_csv(csv_file)
        if (dwt):
            IN_CHANNELS[0] = len(self.data_frame.columns) - 2
        else:
            IN_CHANNELS[0] = 1
        # self.root_dir = root_dir
        self.class_id = class_id
        self.all = all
        if self.all:
            self.all_len = OLD_DATA_LEN if old_data else MAX_LENGTH
        else:
            if (old_data):
                self.all_len = OLD_DATA_LEN - OLD_TRAIN_DATA if is_test else OLD_TRAIN_DATA
            else:
                self.all_len = MAX_LENGTH - TRAIN_DATA if is_test else TRAIN_DATA

        self.useMid = useMid
        self.col_from = 2 if dwt else 0
        self.old_data = old_data
        self.train_len = OLD_TRAIN_DATA if old_data else TRAIN_DATA

    def __len__(self):
        return math.floor(self.all_len / config.SPLIT_SIZE)

    def __getitem__(self, idx):
        idx *= config.SPLIT_SIZE
        if not self.all:
            if self.useMid:
                if self.is_test:
                    idx += int(self.train_len / 2)
                else:
                    if idx > int(self.train_len / 2):
                        idx += self.all_len - self.train_len
            else:
                if self.is_test:
                    idx += self.train_len

        arr = self.data_frame.iloc[idx:idx + config.SPLIT_SIZE, self.col_from:].values

        ret = {'data': torch.FloatTensor(arr), 'target': torch.tensor(self.class_id)}
        return ret


def output_acc(pred, target, out=False):
    output = torch.argmax(pred, dim=1).squeeze()
    sz = output.size(0)
    target_sq = target.squeeze()
    eqx = output.eq(target_sq)
    if (out):
        print(eqx)
    eq = eqx.sum().item()

    return eq / sz, sz, eq


def get_model(whichmodel):
    if whichmodel == 'baseline':
        model = lzc.model.MyModel(IN_CHANNELS, OUT_CHANNELS, K_SIZES, LAYER_SZ, CLASSES)
    elif whichmodel == 'tcn':
        model = lzc.model.TCNModel(IN_CHANNELS[0], TCN_IN_CHANNEL, TCN_KSIZE, CLASSES)
    elif whichmodel=='vdcnn':
        # epoch = 100
        model = lzc.vdcnn.vdcnn(IN_CHANNELS[0], CLASSES)
    else:
        model= lzc.vdcnn.nb_vdcnn(IN_CHANNELS[0], CLASSES)
    return model


def train_model(whichmodel, useMidTest=True, withDwt=True, oldData=False):
    max_acc = 0
    cls = [0, 1, 2]
    add_c = "_mid_" if useMidTest else "_end_"
    add_d = "_dwt_" if withDwt else "_1ch_"
    add_o = "_old_" if oldData else "_new_"
    add_c += add_d + add_o +"_190728_"
    save_pos = whichmodel + add_c + "_save.pt"
    best_save_pos = whichmodel + add_c + "_best.pt"
    log_pos = whichmodel + add_c + '_logdata.log'
    log = open(log_pos, 'w')
    if oldData:
        filenames = ["lzc/olddata/" + ("m" if withDwt else "big") + str(cls[i]) + ('_dwt300.csv' if withDwt else ".txt")
                     for
                     i in range(len(cls))]
    else:
        filenames = ["lzc/" + str(cls[i]) + ('m_dwt300.csv' if withDwt else "m.csv") for i in range(len(cls))]
    print(filenames)
    dataset = torch.utils.data.ConcatDataset(
        [PlantDataset(filenames[i], cls[i], useMid=useMidTest, dwt=withDwt, old_data=oldData) for i in range(len(cls))])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(
            [PlantDataset(filenames[i], cls[i], is_test=True, useMid=useMidTest, dwt=withDwt, old_data=oldData)
             for i in range(len(cls))]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    epoch = EPOCH
    model = get_model(whichmodel)
    optimizer = optim.Adam(model.parameters(), 0.0001)
    print(model)
    model = model.cuda()

    loss_func = nn.CrossEntropyLoss()
    # loss_func =nn.NLLLoss()
    for i in range(epoch):
        total_loss = torch.tensor(0.).cuda()
        total_test_loss = torch.tensor(0.).cuda()
        n_x = 0
        n_y = 0
        for i_batch, sample_batched in enumerate(dataloader):
            data = sample_batched['data'].cuda()
            target = sample_batched['target'].cuda()
            optimizer.zero_grad()
            data = data.transpose(1, 2)
            pred = model(data, data.size(0))
            loss = loss_func(pred, target)
            acc, sz, eq = output_acc(pred, target)
            n_x += 1
            total_loss += loss
            loss.backward()
            print(i_batch, loss, acc, eq, sz)
            optimizer.step()
        sz_now = 0
        eq_now = 0
        for i_batch, now in enumerate(test_dataloader):
            with torch.no_grad():
                data = now['data'].cuda()
                data = data.transpose(1, 2)
                target = now['target'].cuda()
                pred = model(data, data.size(0))
                acc, sz, eq = output_acc(pred, target, True)
                print(whichmodel, add_c, "Epoch ", i, acc, sz, eq, "max=", max_acc)
                sz_now += sz
                eq_now += eq
                n_y += 1
                total_test_loss += F.cross_entropy(pred, target)
        if max_acc < (eq_now / sz_now):
            max_acc = eq_now / sz_now
            torch.save(model.state_dict(), best_save_pos)
        print(i, total_loss.item() / n_x, total_test_loss.item() / n_y, (eq_now / sz_now), max_acc,
              file=log)
        log.flush()
    torch.save(model.state_dict(), save_pos)
    log.close()


if __name__ == '__main__':
    print(BATCH_SIZE)

    all_models = ['baseline', 'tcn', 'vdcnn','nb_vdcnn']
    #
    for i in all_models:
        train_model('nb_vdcnn', False, False)
        train_model('nb_vdcnn', False, True)
        train_model('nb_vdcnn', True, False)
        train_model('nb_vdcnn', True, True)
        # train_model('nb_vdcnn', False, False)

        # train_model(i, withDwt=True, useMidTest=False)
        train_model(i, withDwt=True, useMidTest=True)
        train_model(i, withDwt=True, oldData=True)
        train_model(i, withDwt=True, oldData=True, useMidTest=False)
        train_model(i, withDwt=False, oldData=True)
        train_model(i, withDwt=False, oldData=True, useMidTest=False)
    # train_model('tcn', withDwt=False, oldData=True, useMidTest=False)
    # train_model('vdcnn', withDwt=False, oldData=True, useMidTest=False)
