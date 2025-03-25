import flax.traverse_util
import jax.tree_util
import numpy as np


def get_pretrain_params():

    x=np.load('ckpts/mix_224.npz')
    params={}
    for k in x.keys():
        if 'llm' in k:
            if 'layers' in k:
                datas=x[k]
                for i,data in enumerate(datas):
                    params.update({k.replace('llm/layers', f'layer_{i}'): data})
            else:
                params.update({k.replace('llm/',''):x[k]})
    return flax.traverse_util.unflatten_dict(params,sep='/')


def get_pretrain_vit_params():

    x=np.load('ckpts/mix_224.npz')
    params={}
    for k in x.keys():
        if 'img' in k:
            if 'encoderblock' in k:
                datas=x[k]
                for i,data in enumerate(datas):
                    params.update({k.replace('img/Transformer/encoderblock', f'Transformer/encoderblock_{i}'): data})
            else:
                params.update({k.replace('img/',''):x[k]})
    return flax.traverse_util.unflatten_dict(params,sep='/')



def get_pretrain_pali_params(path='paligemma-3b-pt-224.f16.npz'):
    print(path)
    x=np.load(path,allow_pickle=True)
    params={}
    for k in x.keys():
            if 'encoderblock' in k:
                datas=x[k]
                for i,data in enumerate(datas):
                    params.update({k.replace('encoderblock', f'encoderblock_{i}'): data})
            elif 'layers' in k:
                datas=x[k]
                for i,data in enumerate(datas):
                    params.update({k.replace('layers', f'layer_{i}'): data})
            else:
                params.update({k:x[k]})
    return flax.traverse_util.unflatten_dict(params,sep='/')



