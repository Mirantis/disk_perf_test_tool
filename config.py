import os
import yaml

from petname import Generate as pet_generate


cfg_dict = {}


def load_config(file_name):
    global cfg_dict
    first_load = len(cfg_dict) == 0
    cfg_dict.update(yaml.load(open(file_name).read()))

    if first_load:
        var_dir = cfg_dict.get('internal', {}).get('var_dir_root', '/tmp')

    while True:
        dr = os.path.join(var_dir, pet_generate(2, "_"))
        if not os.path.exists(dr):
            break

    cfg_dict['var_dir'] = dr
    os.makedirs(cfg_dict['var_dir'])

    cfg_dict['charts_img_path'] = os.path.join(cfg_dict['var_dir'], 'charts')
    os.makedirs(cfg_dict['charts_img_path'])

    cfg_dict['vm_ids_fname'] = os.path.join(cfg_dict['var_dir'], 'os_vm_ids')
    cfg_dict['report'] = os.path.join(cfg_dict['var_dir'], 'report.html')
