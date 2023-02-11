from .loader import loader_1d, loader_2d, loader_3d


def loader(opt, split):
    modules = opt.modules.strip().split(',')
    if len(modules) == 1:
        data = loader_1d(opt.data_dir, modules[0], split)
    elif len(modules) == 2:
        data = loader_2d(opt.data_dir, modules[0], modules[1], split)
    elif len(modules) == 3:
        data = loader_3d(opt.data_dir, modules[0], modules[1], modules[2], split)
    else:
        print('Not Implementation ....')
    return data