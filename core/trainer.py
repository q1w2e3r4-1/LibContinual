import json
import os
import pickle
import statistics

import time
import random
import torch
import yaml
import logging
from torch import nn
from time import time
from tqdm import tqdm
from core.data import get_dataloader
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
from core.model.buffer import *
import core.model as arch
from core.model.buffer import *
from torch.utils.data import DataLoader
import numpy as np
import sys
from core.utils import Logger, fmt_date_str

from core.lib import factory
from core.lib import logger as logger_lib
from core.lib import metrics, results_utils, utils


class Trainer(object):
    """
    The Trainer.

    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, args):
        print(1, args)
        logger_lib.set_logging_level(args["logging"])

        autolabel = _set_up_options(args)
        self.logger = self._init_logger(args)
        if args["autolabel"]:
            args["label"] = autolabel

        if args["label"]:
            print("Label: {}".format(args["label"]))
            try:
                os.system("echo '\ek{}\e\\'".format(args["label"]))
            except:
                pass
        if args["resume"] and not os.path.exists(args["resume"]):
            raise IOError(f"Saved model {args['resume']} doesn't exist.")

        if args["save_model"] != "never" and args["label"] is None:
            raise ValueError(f"Saving model every {args['save_model']} but no label was specified.")

        self.seed_list = copy.deepcopy(args["seed"])
        self.device = copy.deepcopy(args["device"])

        self.start_date = utils.get_date()

        self.orders = copy.deepcopy(args["order"])
        print("orders :", self.orders)
        del args["order"]
        if self.orders is not None:
            assert isinstance(self.orders, list) and len(self.orders)
            assert all(isinstance(o, list) for o in self.orders)
            assert all([isinstance(c, int) for o in self.orders for c in o])
        else:
            self.orders = [None for _ in range(len(self.seed_list))]
        # self.rank = rank
        # self.config = config
        # self.config['rank'] = rank
        # self.distribute = self.config['n_gpu'] > 1  # 暂时不考虑分布式训练
        # # (
        # #     self.result_path,
        # #     self.log_path,
        # #     self.checkpoints_path,
        # #     self.viz_path
        # # ) = self._init_files(config)                     # todo   add file manage
        # self.logger = self._init_logger(config)
        # self.device = self._init_device(config)
        # # self.writer = self._init_writer(self.viz_path)   # todo   add tensorboard
        #
        # print(self.config)
        #
        # self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)
        # self.model = self._init_model(config)  # todo add parameter select
        # (
        #     self.train_loader,
        #     self.test_loader,
        # ) = self._init_dataloader(config)
        #
        # self.buffer = self._init_buffer(config)
        # (
        #     self.init_epoch,
        #     self.inc_epoch,
        #     self.optimizer,
        #     self.scheduler,
        # ) = self._init_optim(config)
        #
        # self.train_meter, self.test_meter = self._init_meter()
        #
        # self.val_per_epoch = config['val_per_epoch']

    def _init_logger(self, config, mode='train'):
        '''
        Init logger.

        Args:
            config (dict): Parsed config file.

        Returns:
            logger (Logger)
        '''

        save_path = config['save_path']
        log_path = os.path.join(save_path, "log")
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_prefix = config['model'] + "-" + config['eval_type'] + \
                     "-" + f"increment{config['increment']}" + "-" + f"epoch{config['epochs']}"  # mode
        log_file = os.path.join(log_path, "{}-{}.log".format(log_prefix, fmt_date_str()))

        # if not os.path.isfile(log_file):
        #     os.mkdir(log_file)

        logger = Logger(log_file)

        # hack sys.stdout
        sys.stdout = logger

        return logger

    def _init_device(self, config):
        """"
        Init the devices from the config.
        
        Args:
            config(dict): Parsed config file.
            
        Returns:
            device: a device.
        """
        init_seed(config['seed'], config['deterministic'])
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_ids'])

        device = torch.device("cuda:{}".format(config['device_ids']))
        return device

    def _init_files(self, config):
        pass

    def _init_writer(self, config):
        pass

    def _init_meter(self, ):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
            # self.writer,
        )

        test_meter = [AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            # self.writer,
        ) for _ in range(self.task_num)]

        return train_meter, test_meter

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """
        params_dict_list = {"params": self.model.parameters()}

        optimizer = get_instance(
            torch.optim, "optimizer", config, params=self.model.parameters()
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details

        if 'init_epoch' in config.keys():
            init_epoch = config['init_epoch']
        else:
            init_epoch = config['epoch']

        return init_epoch, config['epoch'], optimizer, scheduler

    def _init_data(self, config):
        return config['init_cls_num'], config['inc_cls_num'], config['task_num']

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        backbone = get_instance(arch, "backbone", config)
        dic = {"backbone": backbone, "device": self.device}

        model = get_instance(arch, "classifier", config, **dic)
        print(model.eval())
        print("Trainable params in the model: {}".format(count_parameters(model)))

        model = model.to(self.device)
        return model

    def _init_dataloader(self, config):
        '''
        Init DataLoader

        Args:
            config (dict): Parsed config file.

        Returns:
            train_loaders (list): Each task's train dataloader.
            test_loaders (list): Each task's test dataloader.
        '''
        train_loaders = get_dataloader(config, "train")
        test_loaders = get_dataloader(config, "test", cls_map=train_loaders.cls_map)

        return train_loaders, test_loaders

    def _init_buffer(self, config):
        '''
        Init Buffer
        
        Args:
            config (dict): Parsed config file.

        Returns:
            buffer (Buffer): a buffer for old samples.
        '''
        buffer = get_instance(arch, "buffer", config)

        return buffer

    def train_loop(self, args):
        """
        The norm train loop:  before_task, train, test, after_task
        """

        avg_inc_accs, last_accs, forgettings = [], [], []
        for i, seed in enumerate(self.seed_list):
            print("Launching run {}/{}".format(i + 1, len(self.seed_list)))
            args["seed"] = seed
            args["device"] = self.device

            # start_time = time.time()

            for avg_inc_acc, last_acc, forgetting in self._start_train(args, self.start_date, self.orders[i], i):
                yield avg_inc_acc, last_acc, forgetting, False

            avg_inc_accs.append(avg_inc_acc)
            last_accs.append(last_acc)
            forgettings.append(forgetting)

            # print("Training finished in {}s.".format(int(time.time() - start_time)))
            yield avg_inc_acc, last_acc, forgetting, True

        print("Label was: {}".format(args["label"]))

        print(
            "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
                len(self.seed_list), _aggregate_results(avg_inc_accs), _aggregate_results(last_accs),
                _aggregate_results(forgettings)
            )
        )
        print("Individual results avg: {}".format([round(100 * acc, 2) for acc in avg_inc_accs]))
        print("Individual results last: {}".format([round(100 * acc, 2) for acc in last_accs]))
        print(
            "Individual results forget: {}".format([round(100 * acc, 2) for acc in forgettings])
        )

        print(f"Command was {' '.join(sys.argv)}")
        # experiment_begin = time()
        # for task_idx in range(self.task_num):
        #     print("================Task {} Start!================".format(task_idx))
        #     if hasattr(self.model, 'before_task'):
        #         self.model.before_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx),
        #                                self.test_loader.get_loader(task_idx))
        #
        #     (
        #         _, __,
        #         self.optimizer,
        #         self.scheduler,
        #     ) = self._init_optim(self.config)
        #
        #     self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
        #
        #     dataloader = self.train_loader.get_loader(task_idx)
        #
        #     if isinstance(self.buffer, LinearBuffer) and task_idx != 0:
        #         datasets = dataloader.dataset
        #         datasets.images.extend(self.buffer.images)
        #         datasets.labels.extend(self.buffer.labels)
        #         dataloader = DataLoader(
        #             datasets,
        #             shuffle=True,
        #             batch_size=self.config['batch_size'],
        #             drop_last=True
        #         )
        #
        #     print("================Task {} Training!================".format(task_idx))
        #     print("The training samples number: {}".format(len(dataloader.dataset)))
        #
        #     best_acc = 0.
        #     for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):
        #         print("learning rate: {}".format(self.scheduler.get_last_lr()))
        #         print("================ Train on the train set ================")
        #         train_meter = self._train(epoch_idx, dataloader)
        #         print("Epoch [{}/{}] |\tLoss: {:.3f} \tAverage Acc: {:.3f} ".format(epoch_idx,
        #                                                                             self.init_epoch if task_idx == 0 else self.inc_epoch,
        #                                                                             train_meter.avg('loss'),
        #                                                                             train_meter.avg("acc1")))
        #
        #         if (epoch_idx + 1) % self.val_per_epoch == 0 or (epoch_idx + 1) == self.inc_epoch:
        #             print("================ Test on the test set ================")
        #             test_acc = self._validate(task_idx)
        #             best_acc = max(test_acc["avg_acc"], best_acc)
        #             print(
        #                 " * Average Acc: {:.3f} Best acc {:.3f}".format(test_acc["avg_acc"], best_acc)
        #             )
        #             print(
        #                 " Per-Task Acc:{}".format(test_acc['per_task_acc'])
        #             )
        #
        #         self.scheduler.step()
        #
        #     if hasattr(self.model, 'after_task'):
        #         self.model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx),
        #                               self.test_loader.get_loader(task_idx))
        #
        #     if self.buffer.strategy == 'herding':
        #         hearding_update(self.train_loader.get_loader(task_idx).dataset, self.buffer, self.model.backbone,
        #                         self.device)
        #     elif self.buffer.strategy == 'random':
        #         random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)

    def _start_train(self, args, start_date, class_order, run_id):
        # set global params
        self._set_seed(args["seed"], args["threads"], args["no_benchmark"], args["detect_anomaly"])
        factory.set_device(args)

        inc_dataset, model = _set_data_model(args, class_order)
        results, results_folder = _set_results(args, start_date)

        memory, memory_val = None, None
        metric_logger = metrics.MetricLogger(
            inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
        )

        for task_id in range(inc_dataset.n_tasks):
            print("================Task {} Start!================".format(task_id))
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
            if task_info["task"] == args["max_task"]:
                break

            model.set_task_info(task_info)

            # ---------------
            # 1. Prepare Task
            # ---------------
            model.eval()
            model.before_task(train_loader, val_loader if val_loader else test_loader)

            print("================Task {} Training!================".format(task_id))
            print("The training samples number: {}".format(len(train_loader.dataset)))

            # -------------
            # 2. Train Task
            # -------------
            self._train_task(args, model, train_loader, val_loader, test_loader, run_id, task_id, task_info)

            # ----------------
            # 3. Conclude Task
            # ----------------
            model.eval()
            self._after_task(args, model, inc_dataset, run_id, task_id, results_folder)

            # ------------
            # 4. Eval Task
            # ------------
            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            ypreds, ytrue = model.eval_task(test_loader)
            metric_logger.log_task(
                ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
            )

            if args["dump_predictions"] and args["label"]:
                os.makedirs(
                    os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
                )
                with open(
                        os.path.join(
                            results_folder, "predictions_{}".format(run_id),
                            str(task_id).rjust(len(str(30)), "0") + ".pkl"
                        ), "wb+"
                ) as f:
                    pickle.dump((ypreds, ytrue), f)

            if args["label"]:
                print(args["label"])
            print("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
            print("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
            print(
                "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
            )
            print("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
            print("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
            print("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
            if task_id > 0:
                print(
                    "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["old_accuracy"],
                        metric_logger.last_results["avg_old_accuracy"]
                    )
                )
                print(
                    "New accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["new_accuracy"],
                        metric_logger.last_results["avg_new_accuracy"]
                    )
                )
            if args.get("all_test_classes"):
                print(
                    "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
                )
                print(
                    "unSeen classes: {:.2f}.".format(
                        metric_logger.last_results["unseen_classes_accuracy"]
                    )
                )

            results["results"].append(metric_logger.last_results)

            avg_inc_acc = results["results"][-1]["incremental_accuracy"]
            last_acc = results["results"][-1]["accuracy"]["total"]
            forgetting = results["results"][-1]["forgetting"]
            yield avg_inc_acc, last_acc, forgetting

            memory = model.get_memory()
            memory_val = model.get_val_memory()

        print(
            "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
        )
        if args["label"] is not None:
            results_utils.save_results(
                results, args["label"], args["model"], start_date, run_id, args["seed"]
            )

        del model
        del inc_dataset

    def _train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        self.model.train()
        meter = self.train_meter
        meter.reset()

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.observe(batch)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                pbar.update(1)

                meter.update("acc1", acc)

        return meter

    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        meter = self.test_meter

        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter[t].reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter[t].update("acc1", acc)

                per_task_acc.append(round(meter[t].avg("acc1"), 2))

        return {"avg_acc": np.mean(per_task_acc), "per_task_acc": per_task_acc}

    def _set_seed(self, seed, nb_threads, no_benchmark, detect_anomaly):
        print("Set seed {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if no_benchmark:
            print("CUDA algos are not determinists but faster!")
        else:
            print("CUDA algos are determinists but very slow!")
        torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
        torch.set_num_threads(nb_threads)
        if detect_anomaly:
            print("Will detect autograd anomaly.")
            torch.autograd.set_detect_anomaly(detect_anomaly)

    def _train_task(self, config, model, train_loader, val_loader, test_loader, run_id, task_id, task_info):
        if config["resume"] is not None and os.path.isdir(config["resume"]) \
                and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
            model.load_parameters(config["resume"], run_id)
            print(
                "Skipping training phase {} because reloading pretrained model.".format(task_id)
            )
        elif config["resume"] is not None and os.path.isfile(config["resume"]) and \
                os.path.exists(config["resume"]) and task_id == 0:
            # In case we resume from a single model file, it's assumed to be from the first task.
            model.network = config["resume"]
            print(
                "Skipping initial training phase {} because reloading pretrained model.".
                    format(task_id)
            )
        else:
            print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            model.train()
            model.train_task(train_loader, val_loader if val_loader else test_loader)

    def _after_task(self, config, model, inc_dataset, run_id, task_id, results_folder):
        if config["resume"] and os.path.isdir(config["resume"]) and not config["recompute_meta"] \
                and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
            model.load_metadata(config["resume"], run_id)
        else:
            model.after_task_intensive(inc_dataset)

        model.after_task(inc_dataset)

        if config["label"] and (
                config["save_model"] == "task" or
                (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
                (config["save_model"] == "first" and task_id == 0)
        ):
            model.save_parameters(results_folder, run_id)
            model.save_metadata(results_folder, run_id)


def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)


def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))


def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    if len(list_results) > 1:
        res = res + " +/- " + str(round(statistics.stdev(list_results) * 100, 2))
    return res


def _set_data_model(config, class_order):
    inc_dataset = factory.get_data(config, class_order)
    config["classes_order"] = inc_dataset.class_order

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(config["model"], start_date, config["label"])
    else:
        results_folder = None

    if config["save_model"]:
        print("Model will be save at this rythm: {}.".format(config["save_model"]))

    results = results_utils.get_template_results(config)

    return results, results_folder
