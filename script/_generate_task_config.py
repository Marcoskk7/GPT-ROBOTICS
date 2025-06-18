import os, yaml, sys
sys.path.append('./')
CONFIGS_PATH='./task_config' 
names = [
    'empty_cup_place',
    'blocks_stack_hard',
    'bowls_stack',
    'dual_shoes_place',
    'put_bottles_dustbin',
]
for task_name in names:
    task_config_path = f'{CONFIGS_PATH}/{task_name}.yml'
    data = {
        'task_name': task_name,
        'random_texture': False,
        'messy_table': False,
        'render_freq': 0,
        'eval_video_log': True,
        'collect_data': True,
        'episode_num': 100,
        'random_camera': False, # TODO
        'random_embodiment': False, # TODO
        'embodiment': ['aloha-agilex-1'],
        'head_camera_type': 'D435',
        'wrist_camera_type': 'D435', # L515, D435, others
        'collect_head_camera': True,
        'collect_wrist_camera': True,
        'data_type':{
            'rgb': True,
            'observer': False,
            'depth': True,
            'pointcloud': True,
            'endpose': False,
            'qpos': True,
            'mesh_segmentation': False,
            'actor_segmentation': False
        },
        'use_seed': False,
        'dual_arm': True,
        'save_path': './data',
        'pcd_down_sample_num': 1024,
        'pcd_crop': True,
        'save_freq': 15,
        'st_episode': 0
    }
    with open(task_config_path, 'w') as f:
        yaml.dump(data,f,default_flow_style = False,sort_keys=False)