import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
# optimizer
import torch.optim as optim
import traceback
# model
from models import I3D_backbone
from models import RegressTree
from models import STGCNFeatureExtractor
from models import EMG_Spectrogram_ResNet18_Encoder
from models import FeatureCompressor
# utils
from utils.misc import import_class
from utils.Group_helper import Group_helper
from torchvideotransforms import video_transforms, volume_transforms


def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    try:
        train_trans, test_trans = get_video_trans()
        Dataset = import_class("datasets." + args.benchmark)
        train_dataset = Dataset(args, transform=train_trans, subset='train')
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return train_dataset, test_dataset
    except Exception as e:
        traceback.print_exc()
        exit()

def model_builder(args):
    base_model = I3D_backbone(I3D_class = 400)
    base_model.load_pretrain(args.pretrained_i3d_weight)
    st_gcn = STGCNFeatureExtractor()
    emg_decoder = EMG_Spectrogram_ResNet18_Encoder()
    feature_fuse = FeatureCompressor(input_dim=1536, output_dim=1024)
    Regressor = RegressTree(
                        #in_channel = 2 * base_model.get_feature_dim() + 1, 
                        in_channel = 2 * 2306 + 1, 
                        hidden_channel = 256, 
                        depth = args.RT_depth)  
    return base_model, Regressor, st_gcn, emg_decoder, feature_fuse

def build_group(dataset_train, args):
    delta_list = dataset_train.delta()
    group = Group_helper(delta_list, args.RT_depth, Symmetrical = True, Max = args.score_range, Min = 0)
    return group

def build_opti_sche(base_model, regressor, st_gcn, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': regressor.parameters()},
            {'params': st_gcn.parameters(), 'lr': args.base_lr * args.lr_factor},
            #{'params': emg_decoder.parameters()}
            #{'params': feature_fuse.parameters()}
        ], lr = args.base_lr , weight_decay = args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler
'''
def build_opti_sche(regressor,emg_decoder, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            #{'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': regressor.parameters()},
            {'params': emg_decoder.parameters()}
        ], lr = args.base_lr , weight_decay = args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler
'''
def resume_train(base_model, regressor, st_gcn, emg_decoder, optimizer, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    gcn_ckpt = {k.replace("module.", ""): v for k, v in state_dict['st_gcn'].items()}
    st_gcn.load_state_dict(gcn_ckpt)

    emg_ckpt = {k.replace("module.", ""): v for k, v in state_dict['emg_decoder'].items()}
    emg_decoder.load_state_dict(emg_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best = state_dict['epoch_best']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best, rho_best, L2_min, RL2_min



def load_model(base_model, regressor, args):
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)
    
    epoch_best = state_dict['epoch_best']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch_best - 1, rho_best,  L2_min, RL2_min))
    return 