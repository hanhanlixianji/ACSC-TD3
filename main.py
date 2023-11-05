import numpy as np
import torch
import gym
import gym_carla
from td3 import TD3
import matplotlib.pyplot as plt
import Buffer
from torch.utils.tensorboard import SummaryWriter
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed):
    
    params = {
      'number_of_vehicles': 0,
      'number_of_walkers': 0,
      'display_size': 64,
      'max_past_step': 1,
      'dt': 0.1,
      'discrete': False,
      'discrete_acc': [-3.0, 0.0, 3.0],
      'discrete_steer': [-0.2, 0.0, 0.2],
      'continuous_accel_range': [-3.0, 3.0],
      'continuous_steer_range': [-0.3, 0.3],
      'ego_vehicle_filter': 'vehicle.lincoln*',
      'port': 2000,
      'town': 'Town02',
      'task_mode': 'roundabout',
      'max_time_episode': 2000,
      'max_waypt': 12,
      'obs_range': 32,
      'lidar_bin': 0.5,
      'd_behind': 12,
      'out_lane_thres': 2.0,
      'desired_speed': 8,
      'max_ego_spawn_times': 200,
      'display_route': False,
      'pixor_size': 64,
      'pixor': False,
    }


    env = gym.make('scenario1_safe-v0', params=params)
    
    state_dim = 4
    action_dim = 2
    max_action = 3
    expl_noise = 0.2
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:', env.action_space.low[0])

    test = False
    Loadmodel = False
    ModelIdex = 200
    random_seed = seed

    Max_episode = 5000
    save_interval = 100

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    writer = SummaryWriter(log_dir='ep_r')

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 3e-4,
        "q_lr": 3e-4,
        "safe_lr":1e-4,
        "Q_batchsize":256,
    }
    model = TD3(**kwargs)
    if Loadmodel: model.load(ModelIdex)
    replay_buffer = Buffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e5))

    all_ep_r = []
    smooth_ep_r = []
    step_avg = 0
    num_accident = 0
    num_success = 0

    for episode in range(Max_episode):
        obs = env.reset()
        s = obs['state']
        done = False
        ep_r = 0
        steps = 0
        # expl_noise *= 0.999
        
        temp_buffer = []

        '''Interact & trian'''
        while not done:
            steps+=1
            if test:
                a = model.select_action(s)
                obs, r, done, success, accident, info = env.step(a)
                s_prime = obs['state']
                # env.render()
                if accident:
                    num_accident += 1
                num_success += success
            else:
                a = ( model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                obs, r, done, success, accident, info = env.step(a)
                
                if accident:
                    num_accident += 1
               
                s_prime = obs['state']
                
                num_success += success
                
                
                
                temp_buffer.append((s, a, r, s_prime, accident, done))
                
                

                if replay_buffer.size > 2000: model.train(replay_buffer)

            s = s_prime
            ep_r += r
            
        if not test:
            if temp_buffer[-1][-1] == True:
                for i, t in enumerate (temp_buffer):
                    sum = 0
                    for j in range(len(temp_buffer)-i):
                        sum += j
                    # target_fi = 0.5 * math.exp(-(sum/len(temp_buffer)))
                    target_fi = 0.5 * (1 - math.exp(-0.2*(sum/len(temp_buffer))))
                    s, a, r, s_prime, _, done = t
                    replay_buffer.add(s, a, r, s_prime, target_fi, done)
            else:
                for i, t in enumerate (temp_buffer):
                    sum = 0
                    for j in range(len(temp_buffer)-i):
                        sum += j
                    target_fi = 0.5 * math.exp(-0.2*(sum/len(temp_buffer)))
                    # target_fi = 0.5 * ( - math.exp(-(sum/len(temp_buffer))))
                    s, a, r, s_prime, _, done = t
                    replay_buffer.add(s, a, r, s_prime, target_fi, done)
                    
            # if replay_buffer.size > 2000: model.train(replay_buffer)


            '''plot & save'''
            if (episode+1)%save_interval==0:
                model.save(episode + 1)
                # plt.plot(all_ep_r)
                # plt.savefig('seed{}-ep{}.png'.format(random_seed,episode+1))
                # plt.clf()


        '''record & log'''
        all_ep_r.append(ep_r)
        
        if episode == 0: smooth_ep_r.append(ep_r)
        else: smooth_ep_r.append(smooth_ep_r[-1]*0.9 + ep_r*0.1)
        avg_accident = num_accident / (episode + 1)
        avg_success = num_success / (episode + 1)
        
        # if episode == 0: all_ep_r.append(ep_r)
        # else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        step_avg = step_avg + (steps - step_avg)/(episode + 1)
        # writer.add_scalar('smooth_ep_r', smooth_ep_r[-1], global_step=episode)
        writer.add_scalar('ep_r', ep_r, global_step=episode)
        # writer.add_scalar('exploare', expl_noise, global_step=episode)
        # writer.add_scalar('step_avg', step_avg, global_step=episode)
        writer.add_scalar('step_num', steps, global_step=episode)
        writer.add_scalar('accident', num_accident, global_step=episode)
        writer.add_scalar('success', num_success, global_step=episode)
        writer.add_scalar('avg_accident', avg_accident, global_step=episode)
        writer.add_scalar('avg_success', avg_success, global_step=episode)
        print('seed:',random_seed,'episode:', episode+1,'score:', ep_r, 'step:',steps , 'max:', max(all_ep_r), 'accident', num_accident, 'success', num_success)

    # np.save('ep_r/scenario_1_safe_pure_v2.npy', all_ep_r)
    # np.save('ep_r/safe_town03_muti_smooth.npy', smooth_ep_r)

    env.close()



if __name__ == '__main__':
    main(seed=1)




































