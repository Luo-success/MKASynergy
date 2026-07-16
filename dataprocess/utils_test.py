import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
import random
import copy

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed dataprocess, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset




        print('Start processing dataprocess...')

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed dataprocess found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed dataprocess {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])


        # 新增：按位置取 smiles

    def get_smi(self, positions):
        return [self.smi_list[i] for i in positions]

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
        #pass

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False


    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of dataprocess', data_len)

        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            cell = self.get_cell_feature(target, xt_featrue)
            if cell == False:
                print('cell', cell)
                sys.exit()
            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))

            # c_size, features, edge_index = smile_graph[smiles]

            ### CHANGE ### 解析现在返回 4 个值
            c_size, features, edge_index = smile_graph[smiles]



            YData = DATA.Data(x=torch.Tensor(features),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                              y=torch.Tensor([labels]))
            YData.cell = torch.FloatTensor([new_cell])


            YData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to dataprocess list
            data_list.append(YData)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')