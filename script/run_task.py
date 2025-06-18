import sys
sys.path.append('./')
import sapien.core as sapien
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance
    
def main():
    task_name = input()
    #task_name = 'empty_cup_place'
    task = class_decorator(task_name)
    task_config_path = f'./task_config/{task_name}.yml'

    assert os.path.isfile(task_config_path), "task config file is missing"

    with open(task_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    embodiment_type = args.get('embodiment')
    embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')

    with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]['file_path']
        if robot_file is None:
            raise "No embodiment files"
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, 'config.yml')
        with open(robot_config_file, 'r', encoding='utf-8') as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    if len(embodiment_type) == 1:
        args['left_robot_file'] = get_embodiment_file(embodiment_type[0])
        args['right_robot_file'] = get_embodiment_file(embodiment_type[0])
        # args['robot_file'] = get_embodiment_file(embodiment_type[0])
        # args['dual_arm_embodied'] = False
        args['dual_arm_embodied'] = True
    elif len(embodiment_type) == 3:
        args['left_robot_file'] = get_embodiment_file(embodiment_type[0])
        args['right_robot_file'] = get_embodiment_file(embodiment_type[1])
        args['embodiment_dis'] = embodiment_type[2]
        args['dual_arm_embodied'] = False
    else:
        raise "embodiment items should be 1 or 3"
    
    # args['embodiment_config'] = get_embodiment_config(args['robot_file'])
    args['left_embodiment_config'] = get_embodiment_config(args['left_robot_file'])
    args['right_embodiment_config'] = get_embodiment_config(args['right_robot_file'])
    
    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + '_' + str(embodiment_type[1])
        
    # output camera config
    print('============= Config =============\n')
    print('Messy Table: ' + str(args['messy_table']))
    print('Random Texture: ' + str(args['random_texture']))
    print('Head Camera Config: '+ str(args['head_camera_type']) + f', ' + str(args['collect_head_camera']))
    print('Wrist Camera Config: '+ str(args['wrist_camera_type']) + f', ' + str(args['collect_wrist_camera']))
    print('Embodiment Config:: '+ embodiment_name)
    print('\n=======================================')

    args['embodiment_name'] = embodiment_name
    args['save_path'] += '/' + str(args['task_name']) + '_' + str(args['head_camera_type'])
    run(task, args)


def run(TASK_ENV, args):
    epid = 0       
    seed_list=[]   
    suc_num = 0    
    fail_num = 0
    
    print(f"Task name: {args['task_name']}")

    if not args['use_seed']:
        while suc_num < args['episode_num']:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed = epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    suc_num+=1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})   ")
                    fail_num +=1
                
                TASK_ENV.close()
                if (args['render_freq']):
                    TASK_ENV.viewer.close()
                epid +=1
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print(f"simulate data episode {suc_num} fail! (seed = {epid})   ")
                print('Error: ', stack_trace)
                print(' -------------')
                fail_num +=1
                TASK_ENV.close()
                if (args['render_freq']):
                    TASK_ENV.viewer.close()
                epid +=1
        
        with open('./task_config/seeds/'+args['task_name']+f'_{args["embodiment_name"]}.txt', 'w') as file:
            for sed in seed_list:
                file.write("%s " % sed)
        print(f'\nComplete simulation, failed {fail_num} times')

    else:
        print(f'using saved seeds list')
        with open('./task_config/seeds/'+args['task_name']+f'_{args["embodiment_name"]}.txt', 'r') as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    if args['collect_data']:
        print('Start data collection')

        args['render_freq']=0
        args['is_save'] = True

        for id in range(args['st_episode'], args['episode_num']):
            TASK_ENV.setup_demo(now_ep_num=id, seed = seed_list[id],**args)
            info_file_path = args['save_path']+'/scene_info.json'
            os.makedirs(args['save_path'], exist_ok=True)

            if not os.path.exists(info_file_path):
                with open(info_file_path, 'w', encoding='utf-8') as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, 'r', encoding='utf-8') as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f'{id}'] = info
            with open(info_file_path, 'w', encoding='utf-8') as file:
                json.dump(info_db, file, ensure_ascii=False)

            TASK_ENV.close()
            print('\nsuccess!')

            
if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
