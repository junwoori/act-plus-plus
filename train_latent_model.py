import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args["chunk_size"]

    print("[DEBUG] args.keys():", args.keys())
    required = ["ckpt_dir", "policy_class", "task_name", "seed", "num_steps"]
    for r in required:
        print(f"[DEBUG] args['{r}']:", args.get(r, None))


    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = ['left_wrist', 'right_wrist', 'top']
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': True,
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }
    config['num_steps'] = args.get('num_steps', 10000)
    # if is_eval:
    #     ckpt_names = [f'policy_best.ckpt']
    #     results = []
    #     for ckpt_name in ckpt_names:
    #         success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
    #         results.append([ckpt_name, success_rate, avg_return])

    #     for ckpt_name, success_rate, avg_return in results:
    #         print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    #     print()
    #     exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names,batch_size_train, batch_size_val, chunk_size)

    # save dataset stats
    # if not os.path.isdir(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    # with open(stats_path, 'wb') as f:
    #     pickle.dump(stats, f)

    ckpt_name = f'policy_last.ckpt'
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, ckpt_name)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'latent_model_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

    


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise NotImplementedError
#     return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


# def eval_bc(config, ckpt_name, save_episode=True):
#     set_seed(1000)
#     ckpt_dir = config['ckpt_dir']
#     state_dim = config['state_dim']
#     real_robot = config['real_robot']
#     policy_class = config['policy_class']
#     onscreen_render = config['onscreen_render']
#     policy_config = config['policy_config']
#     camera_names = config['camera_names']
#     max_timesteps = config['episode_len']
#     task_name = config['task_name']
#     temporal_agg = config['temporal_agg']
#     onscreen_cam = 'angle'

#     # load policy and stats
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     policy = make_policy(policy_class, policy_config)
#     loading_status = policy.load_state_dict(torch.load(ckpt_path))
#     print(loading_status)
#     policy.to(device)
#     policy.eval()
#     print(f'Loaded: {ckpt_path}')
#     stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
#     with open(stats_path, 'rb') as f:
#         stats = pickle.load(f)

#     pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
#     post_process = lambda a: a * stats['action_std'] + stats['action_mean']

#     # load environment
#     if real_robot:
#         from aloha_scripts.robot_utils import move_grippers # requires aloha
#         from aloha_scripts.real_env import make_real_env # requires aloha
#         env = make_real_env(init_node=True)
#         env_max_reward = 0
#     else:
#         from sim_env import make_sim_env
#         env = make_sim_env(task_name)
#         env_max_reward = env.task.max_reward

#     query_frequency = policy_config['num_queries']
#     if temporal_agg:
#         query_frequency = 1
#         num_queries = policy_config['num_queries']

#     max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

#     num_rollouts = 50
#     episode_returns = []
#     highest_rewards = []
#     for rollout_id in range(num_rollouts):
#         rollout_id += 0
#         ### set task
#         if 'sim_transfer_cube' in task_name:
#             BOX_POSE[0] = sample_box_pose() # used in sim reset
#         elif 'sim_insertion' in task_name:
#             BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

#         ts = env.reset()

#         ### onscreen render
#         if onscreen_render:
#             ax = plt.subplot()
#             plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
#             plt.ion()

#         ### evaluation loop
#         if temporal_agg:
#             all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device)

#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).to(device)
#         image_list = [] # for visualization
#         qpos_list = []
#         target_qpos_list = []
#         rewards = []
#         with torch.inference_mode():
#             for t in range(max_timesteps):
#                 ### update onscreen render and wait for DT
#                 if onscreen_render:
#                     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
#                     plt_img.set_data(image)
#                     plt.pause(DT)

#                 ### process previous timestep to get qpos and image_list
#                 obs = ts.observation
#                 if 'images' in obs:
#                     image_list.append(obs['images'])
#                 else:
#                     image_list.append({'main': obs['image']})
#                 qpos_numpy = np.array(obs['qpos'])
#                 qpos = pre_process(qpos_numpy)
#                 qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
#                 qpos_history[:, t] = qpos
#                 curr_image = get_image(ts, camera_names)

#                 ### query policy
#                 if config['policy_class'] == "ACT":
#                     if t % query_frequency == 0:
#                         all_actions = policy(qpos, curr_image)
#                     if temporal_agg:
#                         all_time_actions[[t], t:t+num_queries] = all_actions
#                         actions_for_curr_step = all_time_actions[:, t]
#                         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
#                         actions_for_curr_step = actions_for_curr_step[actions_populated]
#                         k = 0.01
#                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
#                         exp_weights = exp_weights / exp_weights.sum()
#                         exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
#                         raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
#                     else:
#                         raw_action = all_actions[:, t % query_frequency]
#                 elif config['policy_class'] == "CNNMLP":
#                     raw_action = policy(qpos, curr_image)
#                 else:
#                     raise NotImplementedError

#                 ### post-process actions
#                 raw_action = raw_action.squeeze(0).cpu().numpy()
#                 action = post_process(raw_action)
#                 target_qpos = action

#                 ### step the environment
#                 ts = env.step(target_qpos)

#                 ### for visualization
#                 qpos_list.append(qpos_numpy)
#                 target_qpos_list.append(target_qpos)
#                 rewards.append(ts.reward)

#             plt.close()
#         if real_robot:
#             move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
#             pass

#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards!=None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

#         if save_episode:
#             save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
#     for r in range(env_max_reward+1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

#     print(summary_str)

#     # save success rate to txt
#     result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
#     with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write('\n\n')
#         f.write(repr(highest_rewards))

#     return success_rate, avg_return


def forward_pass(data, policy, latent_model, device):
    image_data, qpos_data, action_data, is_pad = data

    #CUDA 전송
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)

    forward_dict = {}
    gt_labels = policy.vq_encode(qpos_data, action_data, is_pad)
    inputs = torch.cat([torch.zeros_like(gt_labels)[:, [0]], gt_labels[:, :-1]], dim=1)
    output_logits = latent_model(inputs)
    ce_loss = F.cross_entropy(output_logits, gt_labels)

    with torch.no_grad():
        output_labels = F.one_hot(torch.argmax(output_logits, dim=-1), num_classes=gt_labels.shape[-1]).float()
        # output_latents = F.softmax(output_logits, dim=-1)
        l1_error = F.l1_loss(output_labels, gt_labels, reduction='mean')
        # l1_errors = []
        # for i in range(l1_errors.shape[1]):
        #     l1_errors.append(torch.mean(l1_errors[:, i]).item())
    
    forward_dict['loss'] = ce_loss
    forward_dict['l1_error'] = l1_error

    return forward_dict


def train_bc(train_dataloader, val_dataloader, config, ckpt_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    save_every = config.get('save_every', 10)
    eval_every = config.get('eval_every', 1)
    validate_every = config.get('validate_every', 1)

    set_seed(seed)

    if policy_class == 'CNNMLP':
        raise NotImplementedError("Latent model training is not supported for policy_class == 'CNNMLP'.")

    

    vq_dim = policy_config['vq_dim']
    vq_class = policy_config['vq_class']
    latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class).to(device)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy_config['action_dim'] = 16
    policy = make_policy(policy_class, policy_config)

    if os.path.exists(ckpt_path):
        _ = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    else:
        print(f"[INFO] No checkpoint found at {ckpt_path}. Training from scratch.")

    policy.eval()
    policy.to(device)

    optimizer = torch.optim.AdamW(latent_model.parameters(), lr=config['lr'])

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        # Validation
        if epoch % validate_every == 0:
            with torch.inference_mode():
                latent_model.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, latent_model, device)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(latent_model.state_dict()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            print(' '.join(f'{k}: {v.item():.3f}' for k, v in epoch_summary.items()))

        # Training
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, latent_model, device)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        # Train loss summary
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        print(' '.join(f'{k}: {v.item():.3f}' for k, v in epoch_summary.items()))

        # Save + plot
        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'latent_model_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(latent_model.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # 마지막 저장
    torch.save(latent_model.state_dict(), os.path.join(ckpt_dir, f'latent_model_last.ckpt'))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, f'latent_model_epoch_{best_epoch}_seed_{seed}.ckpt'))
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info





def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'latent_model_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--dataset_dir', type=str, default=None, help='Path to dataset directory')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument("--num_steps", type=int, default=10000, help="Total number of training steps for ACT model")
    parser.add_argument('--save_every', type=int, default=500, help='checkpoint save frequency')
    parser.add_argument('--eval_every', type=int, default=500, help='evaluation frequency')
    parser.add_argument('--validate_every', type=int, default=500, help='validation frequency')


    
    main(vars(parser.parse_args()))
