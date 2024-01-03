import sys
import os
from core.config import Config
from core import Trainer
import time
from core.utils import parser

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

    return args


if __name__ == "__main__":
    default = get_default_config()

    # config = Config("./config/podnet_nme_cifar100_50steps.yaml").get_config_dict()
    # config = Config("./config/podnet_nme_cifar100_25steps.yaml").get_config_dict()
    # config = Config("./config/podnet_nme_cifar100_10steps.yaml").get_config_dict()
    # config = Config("./config/podnet_nme_cifar100_5steps.yaml").get_config_dict()
    #
    config = Config("./config/podnet_nme_cifar10_5steps.yaml").get_config_dict()
    # config = Config("./config/ucir_cifar100.yaml").get_config_dict()
    # config = Config("./config/icarl_cifar100.yaml").get_config_dict()
    # config = Config("./config/lwm_cifar100.yaml").get_config_dict()

    config = dict(default, **config)  # 用yaml的参数覆盖默认参数
    main(0, config)

    # # config = Config("./config/finetune.yaml").get_config_dict()
    # # config = Config("./config/lwf.yaml").get_config_dict()
    # # config = Config("./config/lwf.yaml").get_config_dict()
    # config = Config("./config/podnet_cnn_cifar100_5steps.yaml").get_config_dict()
    # #
    # config = dict(default, **config) # 用yaml的参数覆盖默认参数
    # print(config)
    # if config["n_gpu"] > 1:
    #     pass
    #     os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    #     # torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    # else:
    #     main(0, config)
