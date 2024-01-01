import sys
import os
from core.config import Config
from core import Trainer
import time
from core.utils import parser
from train import train

sys.dont_write_bytecode = True


def main(rank, config):
    begin = time.time()
    trainer = Trainer(rank, config)
    print(config)
    for _ in trainer.train_loop(config): # `train` is a generator in order to be used with hyperfind.
        pass

    # for _ in train(config):
    #     pass
    # print(1)
    print("Time cost : ", time.time() - begin)


def get_default_config():
    """
        由于原来的模型需要的参数实在太多，全部加到yaml里太复杂，索性就用论文中的方式直接获取了，
        但其实等价于读取yaml(事实上还是会读取yaml并覆盖默认参数)

        returns:
            args : 所有默认配置参数
    """

    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range", args["seed"])

    return args


if __name__ == "__main__":
    config = parser.get_parser().parse_args()
    config = vars(config)  # Converting argparse Namespace to a dict.

    main(0, config)
    # default = get_default_config()
    # # config = Config("./config/finetune.yaml").get_config_dict()
    # # config = Config("./config/lwf.yaml").get_config_dict()
    # # config = Config("./config/lwf.yaml").get_config_dict()
    # config = Config("./config/podnet_cnn_cifar100.yaml").get_config_dict()
    # #
    # config = dict(default, **config) # 用yaml的参数覆盖默认参数
    # print(config)
    # if config["n_gpu"] > 1:
    #     pass
    #     os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    #     # torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    # else:
    #     main(0, config)
