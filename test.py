import os
import torch
import hydra
from omegaconf import OmegaConf
from importlib import import_module

# ===============
# assign "config_name" and import testing module
# ===============


@hydra.main(config_path='conf', config_name='cebd_test', version_base="1.1")
def main(config):
    # GPU setting
    if not config.gpus or config.gpus == -1:
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = config.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    n_gpu = len(gpus)

    """
    assert n_gpu <= torch.cuda.device_count(
    ), 'Can\'t find %d GPU device on this machine.' % (n_gpu)
    """
    # -- Modified by Chu King on October 24, 2024
    # -- The machine doesn't necessarily have GPUs.
    if n_gpu == 0:
        print ("\n[WARNING] No GPU found. Running on CPU.\n")
    else:
        print ("\n[INFO] Using {} GPU(s).\n".format(n_gpu))

    # show config
    config_v = OmegaConf.to_yaml(config, resolve=True)
    print('='*40+'\n', config_v, '\n'+'='*40+'\n')

    # testing
    tester_name = 'srcs.tester.%s' % config.tester_name
    testing_module = import_module(tester_name)
    testing_module.testing(gpus, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
