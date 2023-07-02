path_sd15 = './models/v1-5-pruned.ckpt'
path_sd15_with_control = './models/control_sd15_canny.pth'
path_input = './models/solsticeAKoreanWebtoon_v10AreumSafetensors.safetensors'
path_output = './models/control_solsticeAKoreanWebtoon_canny.pth'



import os


assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
assert os.path.exists(path_input), 'Input path_input does not exists!'
assert os.path.exists(os.path.dirname(path_output)), 'Output folder not exists!'


import torch
from share import *
from cldm.model import load_state_dict

import platform
import gc
import psutil

print(platform.architecture())

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f'Current memory usage: {process.memory_info().rss / 1024 ** 3} GB')


sd15_state_dict = load_state_dict(path_sd15)
gc.collect()
print_memory_usage()
input_state_dict = load_state_dict(path_input)
gc.collect()
print_memory_usage()
sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
gc.collect()
print_memory_usage()


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


keys = sd15_with_control_state_dict.keys()

final_state_dict = {}
for key in keys:
    is_first_stage, _ = get_node_name(key, 'first_stage_model')
    is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
    if is_first_stage or is_cond_stage:
        final_state_dict[key] = input_state_dict[key]
        continue
    p = sd15_with_control_state_dict[key]
    is_control, node_name = get_node_name(key, 'control_')
    if is_control:
        sd15_key_name = 'model.diffusion_' + node_name
    else:
        sd15_key_name = key
    if sd15_key_name in input_state_dict:
        p_diff = input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
        p_new = p + p_diff
        del input_state_dict[sd15_key_name]
        del sd15_state_dict[sd15_key_name]
        gc.collect()
        print_memory_usage()
        # print(f'Offset clone from [{sd15_key_name}] to [{key}]')
    else:
        p_new = p
        # print(f'Direct clone to [{key}]')
    final_state_dict[key] = p_new

torch.save(final_state_dict, path_output)
print('Transferred model saved at ' + path_output)