import torch
import scipy.io
import os
import random
import numpy as np
import pandas as pd
from utils import misc
from PIL import Image

class SevenPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform

        #classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        classes_name = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

        self.score_range = args.score_range
        # file path
        self.data_root = args.data_root
        self.split_path = os.path.join(self.data_root, 'Split_4', 'split_4_train_list.mat')
        self.split = scipy.io.loadmat(self.split_path)['consolidated_train_list']
        
        self.args_class_idx = args.class_idx

        if args.class_idx == 0:
            self.data_path = self.data_root
            self.split = self.split.tolist()
            if self.subset == 'test':
                self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
                self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
                self.split_test = self.split_test.tolist()
        else:
            self.sport_class = classes_name[args.class_idx - 1]
            self.class_idx = args.class_idx # sport class index(from 1 begin)
            #self.data_path = os.path.join(self.data_root, '{}-out'.format(self.sport_class))
            self.data_path = os.path.join(self.data_root, self.sport_class)
            self.split = self.split[self.split[:,0] == self.class_idx].tolist()
            if self.subset == 'test':
                self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
                self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
                self.split_test = self.split_test[self.split_test[:,0] == self.class_idx].tolist()
            # setting
        self.length = args.frame_length
        self.voter_number = args.voter_number

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else:
            self.dataset = self.split.copy()

    def load_video(self, class_idx, idx):
        video_path = os.path.join(self.data_root, 'A%02d'%class_idx, '%03d'%idx)
        #video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % ( i + 1 ))) for i in range(self.length)]  
        video = [Image.open(os.path.join(video_path, 'View-1', 'img_%05d.JPG' % ( i + 1 ))) for i in range(self.length)]
        return self.transforms(video)
    
    def load_video2(self, class_idx, idx):
        video_path = os.path.join(self.data_root, 'A%02d'%class_idx, '%03d'%idx)
        #video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % ( i + 1 ))) for i in range(self.length)]  
        video = [Image.open(os.path.join(video_path, 'View-2', 'img_%05d.JPG' % ( i + 1 ))) for i in range(self.length)]
        return self.transforms(video)
    
    def load_Skeleton(self, class_idx, idx):
        skeleton_path = os.path.join(self.data_root, 'Skeleton', 'A%02d'%class_idx, '%03d'%idx)
        raw_data = pd.read_csv(os.path.join(skeleton_path, 'skeleton_points.csv'),header=0)
        # transfer
        raw_data = raw_data.values
        raw_21 = raw_data.reshape(-1, 21, 3)  # (103, 21, 3)
        raw_25 = misc.remap21_to_25(raw_21)  # (103, 25, 3)
        skeleton_data = torch.from_numpy(raw_25).float()
        skeleton_data = skeleton_data.unsqueeze(0)  # → (1, 103, 25, 3)
        skeleton_data = misc.normalize_ntu_skeleton(skeleton_data.numpy(), 0, 1)  # (1, 103, 25, 3)
        skeleton_data = torch.from_numpy(skeleton_data).float() # → (1, 103, 25, 3)
        skeleton_data = skeleton_data.permute(3, 1, 2, 0) # → (3, 103, 25, 1)
        return skeleton_data
    
    def load_EMG(self, class_idx, idx):
        #print(class_idx, idx)
        emg_path = os.path.join(self.data_root, 'EMG','A%02d'%class_idx, '%03d'%idx)
        raw_data = pd.read_csv(os.path.join(emg_path, 'EMG.csv'),header=None)
        
        if class_idx in [1,2,4,5,14,16]: #1
            L_muscle = max(abs(raw_data.iloc[:, 0]))
            R_muscle = max(abs(raw_data.iloc[:, 2]))
        elif class_idx in [3,6,15,20]: #2
            L_muscle = max(abs(raw_data.iloc[:, 1]))
            R_muscle = max(abs(raw_data.iloc[:, 3]))
        elif class_idx in [7,17,18,19]:# Single
            L_muscle = max(abs(raw_data.iloc[:, 0]))
            R_muscle = max(abs(raw_data.iloc[:, 1]))
        elif class_idx in [8,9,10,11,12,13]: # Aveage
            L_muscle = max((abs(raw_data.iloc[:, 0])+abs(raw_data.iloc[:, 1]))*0.5)
            R_muscle = max((abs(raw_data.iloc[:, 2])+abs(raw_data.iloc[:, 3]))*0.5)
        L_contribution = L_muscle / (L_muscle + R_muscle)
        R_contribution = R_muscle / (L_muscle + R_muscle)
        emg_data = torch.tensor([[L_contribution, R_contribution]])
        return emg_data


    def delta(self):
        delta = []
        dataset = self.split.copy()
        if self.args_class_idx == 0:
            for k in range(20):
                dataset2 = [row for row in dataset if row[0] == k + 1]
                for i in range(len(dataset2)):
                    for j in range(i+1,len(dataset2)):
                        delta.append(abs(misc.normalize(dataset2[i][2], int(dataset2[i][0]), self.score_range) - misc.normalize(dataset2[j][2], int(dataset2[j][0]), self.score_range)))
        else:
            for i in range(len(dataset)):
                for j in range(i+1,len(dataset)):
                    delta.append(abs(misc.normalize(dataset[i][2], int(dataset[i][0]), self.score_range) - misc.normalize(dataset[j][2], int(dataset[j][0]), self.score_range)))
        return delta

    def __getitem__(self,index):
        sample_1  = self.dataset[index]
        #assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])
        class_idx = int(sample_1[0])

        data = {}
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(class_idx, idx)
            data['video2'] = self.load_video2(class_idx, idx)
            data['Skeleton'] = self.load_Skeleton(class_idx, idx)
            data['EMG'] = self.load_EMG(class_idx, idx)
            data['final_score'] = misc.normalize(sample_1[2], class_idx, self.score_range)
            # choose a list of sample in training_set
            train_file_list = self.split.copy()
            if self.args_class_idx == 0:
                train_file_list = [row for row in self.split if row[0] == class_idx]
            else:
                train_file_list = self.split.copy()
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            #print(len(choosen_sample_list))
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp_idx = int(item[1])
                tmp['video'] = self.load_video(class_idx, tmp_idx)
                tmp['video2'] = self.load_video2(class_idx, tmp_idx)
                tmp['Skeleton'] = self.load_Skeleton(class_idx, tmp_idx)
                tmp['EMG'] = self.load_EMG(class_idx, tmp_idx)
                tmp['final_score'] = misc.normalize(item[2], class_idx, self.score_range)
                target_list.append(tmp)
            return data , target_list
        else:
            # train phase
            data['video'] = self.load_video(class_idx, idx)
            data['video2'] = self.load_video2(class_idx, idx)
            data['Skeleton'] = self.load_Skeleton(class_idx, idx)
            data['EMG'] = self.load_EMG(class_idx, idx)
            data['final_score'] = misc.normalize(sample_1[2], class_idx, self.score_range)
         
            # choose a sample
            # did not using a pytorch sampler, using diff_dict to pick a video sample
            if self.args_class_idx == 0:
                file_list = [row for row in self.split if row[0] == class_idx]
            else:
                file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            tmp_idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[tmp_idx]
            target = {}
            # sample 2
            target['video'] = self.load_video(class_idx, int(sample_2[1]))
            target['video2'] = self.load_video2(class_idx, int(sample_2[1]))
            target['Skeleton'] = self.load_Skeleton(class_idx, int(sample_2[1]))
            target['EMG'] = self.load_EMG(class_idx, int(sample_2[1]))
            target['final_score'] = misc.normalize(sample_2[2], class_idx, self.score_range)
            return data , target
    def __len__(self):
        return len(self.dataset)