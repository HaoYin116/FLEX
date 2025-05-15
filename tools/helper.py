import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import time
import numpy as np
import torch.nn as nn


def  network_forward_train(base_model, regressor, st_gcn, emg_decoder, pred_scores, video_1, video_12, skeleton_1, emg_1, label_1, video_2, video_22, skeleton_2, emg_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, batch_idx, batch_num, args):
    loss = 0.0
    start = time.time()
    #combined_feature_1 , combined_feature_2 = base_model(video_1,video_2,label = [label_1, label_2], is_train = True, theta = args.score_range)
    _, _, video_11, video_21 = base_model(video_1,video_2,label = [label_1, label_2], is_train = True, theta = args.score_range)
    #combined_feature_12 , combined_feature_22 = base_model(video_12,video_22,label = [label_1, label_2], is_train = True, theta = args.score_range)
    _, _, video_12, video_22 = base_model(video_12,video_22,label = [label_1, label_2], is_train = True, theta = args.score_range)

    #combined_feature_1 = (combined_feature_1 + combined_feature_12) / 2
    #combined_feature_2 = (combined_feature_2 + combined_feature_22) / 2
    #visual_feature1 = (video_11 + video_12) / 2
    #visual_feature2 = (video_21 + video_22) / 2
    visual_feature1 = torch.cat([video_11, video_12], 1)
    visual_feature2 = torch.cat([video_21, video_22], 1)
    #print('Vis', visual_feature1.mean(), visual_feature1.min(), visual_feature1.max())
    #print('Vis', visual_feature2.mean(), visual_feature2.min(), visual_feature2.max(), '\n')

    skeleton_feature1 = st_gcn(skeleton_1)
    skeleton_feature2 = st_gcn(skeleton_2)
    #print('Ske', skeleton_feature1.mean(), skeleton_feature1.min(), skeleton_feature1.max())
    #print('Ske', skeleton_feature2.mean(), skeleton_feature2.min(), skeleton_feature2.max(), '\n')
    
    #emg_feature1 = emg_decoder(emg_1.squeeze(0))
    #emg_feature2 = emg_decoder(emg_2.squeeze(0))
    emg_feature1 = emg_1.squeeze(0)
    emg_feature2 = emg_2.squeeze(0)
    #emg_feature1 = nn.LayerNorm(512, elementwise_affine=False)(emg_feature1)
    #emg_feature2 = nn.LayerNorm(512, elementwise_affine=False)(emg_feature2)
    #print('EMG', emg_feature1.mean(), emg_feature1.min(), emg_feature1.max())
    #print('EMG', emg_feature2.mean(), emg_feature2.min(), emg_feature2.max(), '\n')

    combined_feature1 = torch.cat([visual_feature1, skeleton_feature1, emg_feature1], 1)
    combined_feature2 = torch.cat([visual_feature2, skeleton_feature2, emg_feature2], 1)

    combined_feature_1 = torch.cat((combined_feature1, combined_feature2, label_1 / args.score_range),1)
    combined_feature_2 = torch.cat((combined_feature2, combined_feature1, label_2 / args.score_range),1)

    combined_feature = torch.cat((combined_feature_1,combined_feature_2),0)
    
    out_prob , delta = regressor(combined_feature)
    # tree-level label
    glabel_1, rlabel_1 = group.produce_label(label_2 - label_1)
    glabel_2, rlabel_2 = group.produce_label(label_1 - label_2)
    # predictions
    leaf_probs = out_prob[-1].reshape(combined_feature.shape[0],-1)
    leaf_probs_1 = leaf_probs[:leaf_probs.shape[0]//2]
    leaf_probs_2 = leaf_probs[leaf_probs.shape[0]//2:]
    delta_1 = delta[:delta.shape[0]//2]
    delta_2 = delta[delta.shape[0]//2:]
    # loss
    loss += nll(leaf_probs_1,glabel_1.argmax(0))
    loss += nll(leaf_probs_2,glabel_2.argmax(0))
    for i in range(group.number_leaf()):
        mask = rlabel_1[i] >= 0
        if mask.sum() != 0:
            loss += mse(delta_1[:,i][mask].reshape(-1,1).float(), rlabel_1[i][mask].reshape(-1,1).float())
        mask = rlabel_2[i] >= 0
        if mask.sum() != 0:
            loss += mse(delta_2[:,i][mask].reshape(-1,1).float(), rlabel_2[i][mask].reshape(-1,1).float())
    #loss = loss / args.step_per_update
    loss.backward()

    if opti_flag:
        optimizer.step()
        optimizer.zero_grad()

    end = time.time()
    batch_time = end - start
    '''
    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
                % (epoch, args.max_epoch, batch_idx, batch_num,
                batch_time, loss.item(), optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))
    '''
    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
                % (epoch, args.max_epoch, batch_idx, batch_num,
                batch_time, loss.item(), 0, optimizer.param_groups[0]['lr']))

    # evaluate result of training phase
    relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(),delta_2.detach().cpu().numpy())
    if args.benchmark == 'MTL':
        if args.usingDD:
            score = (relative_scores.cuda() + label_2)  * diff
        else:
            score = relative_scores.cuda() + label_2
    elif args.benchmark == 'Seven':
        score = relative_scores.cuda() + label_2
    else:
        raise NotImplementedError()
    pred_scores.extend([i.item() for i in score])

def network_forward_test(base_model, regressor, st_gcn, emg_decoder, pred_scores, video_1, video_12, skeleton_1, emg_1, video_2_list, video_22_list, skeleton_2_list, emg_2_list, label_2_list, diff, group, args):
    score = 0
    for video_2,video_22,skeleton_2,emg_2,label_2 in zip(video_2_list,video_22_list,skeleton_2_list,emg_2_list,label_2_list):
        #combined_feature1 = base_model(video_1,video_2, label = [label_2], is_train = False , theta = args.score_range)
        _, video_11f, video_21f = base_model(video_1,video_2, label = [label_2], is_train = False , theta = args.score_range)
        #combined_feature2 = base_model(video_12,video_22, label = [label_2], is_train = False , theta = args.score_range)
        _, video_12f, video_22f = base_model(video_12,video_22, label = [label_2], is_train = False , theta = args.score_range)
        #combined_feature = (combined_feature1 + combined_feature2) / 2
        #visual_feature1 = (video_11f + video_12f) / 2
        #visual_feature2 = (video_21f + video_22f) / 2

        visual_feature1 = torch.cat([video_11f, video_12f], 1)
        visual_feature2 = torch.cat([video_21f, video_22f], 1)
        
        skeleton_feature1 = st_gcn(skeleton_1)
        skeleton_feature2 = st_gcn(skeleton_2)

        emg_feature1 = emg_1.squeeze(0)
        emg_feature2 = emg_2.squeeze(0)

        combined_feature1 = torch.cat([visual_feature1, skeleton_feature1, emg_feature1], 1)
        combined_feature2 = torch.cat([visual_feature2, skeleton_feature2, emg_feature2], 1)

        combined_feature = torch.cat((combined_feature2, combined_feature1, label_2 / args.score_range), 1)
        #combined_feature = torch.cat((emg_feature1, emg_feature2, label_2 / args.score_range), 1)

        out_prob , delta = regressor(combined_feature)
        # evaluate result of training phase
        leaf_probs = out_prob[-1].reshape(combined_feature.shape[0],-1)
        relative_scores = group.inference(leaf_probs.detach().cpu().numpy(),delta.detach().cpu().numpy())
        if args.benchmark == 'MTL':
            if args.usingDD:
                score += (relative_scores.cuda() + label_2)  * diff
            else:
                score += relative_scores.cuda() + label_2
        elif args.benchmark == 'Seven':
            score += relative_scores.cuda() + label_2
            # score += misc.denormalize(relative_scores.cuda() + label_2, args.class_idx, args.score_range)
        else:
            raise NotImplementedError()
    pred_scores.extend([i.item() / len(video_2_list) for i in score])

def save_checkpoint(base_model, regressor,  st_gcn, emg_decoder, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, exp_name, args):
    torch.save({
                'base_model' : base_model.state_dict(),
                'regressor' : regressor.state_dict(),
                'st_gcn' : st_gcn.state_dict(),
                'emg_decoder' : emg_decoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'epoch_best': epoch_best,
                'rho_best' : rho_best,
                'L2_min' : L2_min,
                'RL2_min' : RL2_min,
                }, os.path.join(args.experiment_path, exp_name + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred ,pred_scores)
    np.save(save_path_true ,true_scores)
