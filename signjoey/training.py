#!/usr/bin/env python
import warnings
from google.protobuf.reflection import ParseMessage
warnings.filterwarnings("ignore")

from torch.nn.parallel import DistributedDataParallel as DDP, distributed
import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os, sys
import shutil
import time
import queue
sys.path.append(os.getcwd())#slt dir
import signjoey
from signjoey.model import build_model
from signjoey.batch import Batch, Batch_from_examples
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
    is_main_process
)
from signjoey.model import SignModel, get_loss_for_batch
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.data import load_data, make_data_iter
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.prediction import test
from signjoey.metrics import wer_single
from signjoey.vocabulary import SIL_TOKEN
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict, distributed=False) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]
        self.train_config = config['training']
        self.distributed = distributed
        # files for logging and storing
        if is_main_process():
            self.model_dir = make_model_dir(
                train_config["model_dir"], overwrite=train_config.get("overwrite", False)
            )
        else:
            self.model_dir = train_config["model_dir"]
        
        if self.distributed:
            torch.distributed.barrier()

        self.logger = make_logger(model_dir=self.model_dir, log_file='train.rank{}.log'.format(os.environ['RANK']))
        self.logging_freq = train_config.get("logging_freq", 100)

        if is_main_process():
            self.valid_report_file = "{}/validations.txt".format(self.model_dir)
            self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")
        else:
            self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard_rank{}/".format(os.environ['LOCAL_RANK']))
        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        # Check if we are doing only recognition or only translation or both
        self.do_recognition = (
            config["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            config["training"].get("translation_loss_weight", 1.0) > 0.0
        )

        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.lr_decay_cnt = 0
        self.lr_decay_max_cnt = train_config.get("lr_decay_max_cnt", 3)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # data_augmentation parameters
        self.frame_subsampling_ratio = config["data"].get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = config["data"].get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = config["data"].get(
            "random_frame_masking_ratio", None
        )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        self.use_cuda = train_config["use_cuda"]

        if self.use_cuda:
            self.model.cuda()
        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )
        if distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)


        # initialize training statistics
        self.steps = 0
        self.noimprove_delta = (self.train_config.get(
            "patience", 10)+1)*self.validation_freq
        self.last_update_steps = -self.noimprove_delta+1 #prevent stop training at th very beginning
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )


    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.ctcloss_level = train_config.get('ctcloss_level', 'token')
        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True,
            reduction='mean' if self.ctcloss_level=='token' else 'sum'
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint  
        # since symlink_update is not supported in azure storage, we use copy instead
        symlink_update(
            "{}/{}.ckpt".format(self.model_dir,self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        if not self.use_cuda:
            map_location = 'cpu' 
        else:
            if 'LOCAL_RANK' in os.environ:
                map_location = 'cuda:{}'.format(os.environ['LOCAL_RANK'])
            else:
                map_location = 'cuda'
        model_checkpoint = load_checkpoint(path=path, map_location=map_location)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                model_checkpoint["scheduler_state"] is not None
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.total_gls_tokens = model_checkpoint["total_gls_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # # move parameters to cuda already in map_location
        # if self.use_cuda:
        #     self.model.cuda()

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset, distributed=False) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter, train_sampler = make_data_iter(
            train_data,
            collate_fn=lambda x: Batch_from_examples(
                    is_train=True,
                    example_list=x,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    dataset=train_data,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                ),
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            distributed=distributed,
            shuffle=self.shuffle,
        )
        epoch_no = None
        for epoch_no in range(self.epochs):
            if distributed:
                train_sampler.set_epoch(epoch_no)
                
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            for batch in iter(train_iter):
                # reactivate training
                # create a Batch object from torchtext batch
                # batch = Batch(
                #     is_train=True,
                #     torch_batch=batch,
                #     txt_pad_index=self.txt_pad_index,
                #     sgn_dim=self.feature_size,
                #     use_cuda=self.use_cuda,
                #     frame_subsampling_ratio=self.frame_subsampling_ratio,
                #     random_frame_subsampling=self.random_frame_subsampling,
                #     random_frame_masking_ratio=self.random_frame_masking_ratio,
                # )
                batch._make_cuda()
                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0

                recognition_loss, translation_loss = self._train_batch(
                    batch, update=update
                )

                if self.do_recognition:
                    if is_main_process():
                        self.tb_writer.add_scalar(
                            "train/train_recognition_loss", recognition_loss, self.steps
                        )
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    if is_main_process():
                        self.tb_writer.add_scalar(
                            "train/train_translation_loss", translation_loss, self.steps
                        )
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                        epoch_no + 1, self.steps,
                    )

                    if self.do_recognition:
                        elapsed_gls_tokens = (
                            self.total_gls_tokens - processed_gls_tokens
                        )
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(
                            recognition_loss
                        )
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(
                            elapsed_gls_tokens / elapsed
                        )
                    if self.do_translation:
                        elapsed_txt_tokens = (
                            self.total_txt_tokens - processed_txt_tokens
                        )
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(
                            translation_loss
                        )
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(
                            elapsed_txt_tokens / elapsed
                        )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    self.logger.info('Validate on data on process rank 0')
                    valid_start_time = time.time()
                    if is_main_process():
                        self.logger.info('validate_on_data!')
                        # TODO (Cihan): There must be a better way of passing
                        #   these recognition only and translation only parameters!
                        #   Maybe have a NamedTuple with optional fields?
                        #   Hmm... Future Cihan's problem.
                        val_res = validate_on_data(
                            model=self.model.module,
                            data=valid_data,
                            batch_size=self.eval_batch_size,
                            use_cuda=self.use_cuda,
                            batch_type=self.eval_batch_type,
                            dataset_version=self.dataset_version,
                            sgn_dim=self.feature_size,
                            txt_pad_index=self.txt_pad_index,
                            # Recognition Parameters
                            do_recognition=self.do_recognition,
                            recognition_loss_function=self.recognition_loss_function
                            if self.do_recognition
                            else None,
                            recognition_loss_weight=self.recognition_loss_weight
                            if self.do_recognition
                            else None,
                            recognition_beam_size=self.eval_recognition_beam_size
                            if self.do_recognition
                            else None,
                            # Translation Parameters
                            do_translation=self.do_translation,
                            translation_loss_function=self.translation_loss_function
                            if self.do_translation
                            else None,
                            translation_max_output_length=self.translation_max_output_length
                            if self.do_translation
                            else None,
                            level=self.level if self.do_translation else None,
                            translation_loss_weight=self.translation_loss_weight
                            if self.do_translation
                            else None,
                            translation_beam_size=self.eval_translation_beam_size
                            if self.do_translation
                            else None,
                            translation_beam_alpha=self.eval_translation_beam_alpha
                            if self.do_translation
                            else None,
                            frame_subsampling_ratio=self.frame_subsampling_ratio,
                        )
                        self.model.train()

                        if self.do_recognition:
                            # Log Losses and ppl
                            self.tb_writer.add_scalar(
                                "valid/valid_recognition_loss",
                                val_res["valid_recognition_loss"],
                                self.steps,
                            )
                            self.tb_writer.add_scalar(
                                "valid/wer", val_res["valid_scores"]["wer"], self.steps
                            )
                            self.tb_writer.add_scalars(
                                "valid/wer_scores",
                                val_res["valid_scores"]["wer_scores"],
                                self.steps,
                            )

                        if self.do_translation:
                            self.tb_writer.add_scalar(
                                "valid/valid_translation_loss",
                                val_res["valid_translation_loss"],
                                self.steps,
                            )
                            self.tb_writer.add_scalar(
                                "valid/valid_ppl", val_res["valid_ppl"], self.steps
                            )

                            # Log Scores
                            self.tb_writer.add_scalar(
                                "valid/chrf", val_res["valid_scores"]["chrf"], self.steps
                            )
                            self.tb_writer.add_scalar(
                                "valid/rouge", val_res["valid_scores"]["rouge"], self.steps
                            )
                            self.tb_writer.add_scalar(
                                "valid/bleu", val_res["valid_scores"]["bleu"], self.steps
                            )
                            self.tb_writer.add_scalars(
                                "valid/bleu_scores",
                                val_res["valid_scores"]["bleu_scores"],
                                self.steps,
                            )

                        if self.early_stopping_metric == "recognition_loss":
                            assert self.do_recognition
                            ckpt_score = val_res["valid_recognition_loss"]
                        elif self.early_stopping_metric == "translation_loss":
                            assert self.do_translation
                            ckpt_score = val_res["valid_translation_loss"]
                        elif self.early_stopping_metric in ["ppl", "perplexity"]:
                            assert self.do_translation
                            ckpt_score = val_res["valid_ppl"]
                        else:
                            ckpt_score = val_res["valid_scores"][self.eval_metric]
                        
                        if distributed:
                            #broadcast to other processes
                            self.logger.info(
                                'broadcast from 0 to others: ckpt_score={}'.format(float(ckpt_score)))
                            torch.distributed.broadcast(torch.tensor(ckpt_score).cuda(), src=0)                      

                    else:
                        if distributed:
                            ckpt_score = torch.empty(1,).cuda()
                            torch.distributed.broadcast(ckpt_score, src=0)
                            self.logger.info('broadcast from 0 to {}: ckpt_score {}'.format(os.environ['LOCAL_RANK'],float(ckpt_score)))

                    if distributed:
                        torch.distributed.barrier()

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"] if is_main_process() else None
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            "Hooray! New best validation result [%s]!",
                            self.early_stopping_metric,
                        )
                        if self.ckpt_queue.maxsize > 0:
                            new_best = True
                            if is_main_process():
                                self.logger.info("Saving new checkpoint.")
                                self._save_checkpoint()
                            if distributed:
                                torch.distributed.barrier()

                    if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        
                        if distributed and is_main_process():
                            torch.distributed.broadcast(torch.tensor(now_lr).cuda(), src=0)
                            self.logger.info('broadcast from 0 to others: now_lr={}'.format(float(now_lr)))
                        elif distributed:
                            now_lr0 = torch.empty(1,).cuda()
                            torch.distributed.broadcast(now_lr0, src=0) 
                            self.logger.info('broadcast from 0 to {}: now_lr0={}'.format(os.environ['LOCAL_RANK'], float(now_lr0)))
                            #assert now_lr0==now_lr  
                        if distributed:
                            torch.distributed.barrier() 

                        if prev_lr != now_lr:
                            if self.steps-self.last_update_steps <= self.noimprove_delta:
                                self.lr_decay_cnt += 1
                            else: # there is some improve during last patient phase.
                                self.lr_decay_cnt = 0
                            self.last_update_steps = self.steps
                        else:
                            if self.steps-self.last_update_steps <= self.noimprove_delta:
                                pass #be patient
                            else:
                                self.lr_decay_cnt = 0 # improve 

                        if self.lr_decay_cnt>=self.lr_decay_max_cnt:
                            self.stop = True
                        # if prev_lr != now_lr:
                        #     if self.last_best_lr != prev_lr:
                        #         self.stop = True

                    current_lr = -1
                    # ignores other param groups for now
                    for param_group in self.optimizer.param_groups:
                        current_lr = param_group["lr"]
                    if new_best:
                        self.last_best_lr = current_lr
                    if current_lr < self.learning_rate_min:
                        self.stop = True

                    self.tb_writer.add_scalar("train/learning_rate", current_lr,  self.steps)
                    
                    if distributed and is_main_process():
                        self.logger.info('broadcast from 0 to others: new_best={}'.format(new_best))
                        torch.distributed.broadcast(torch.tensor(int(new_best)).cuda(), src=0)
                    elif distributed:
                        new_best0 = torch.empty(1,).cuda()
                        torch.distributed.broadcast(new_best0, src=0)
                        self.logger.info('broadcast from 0 to {}: new_best0={}'.format(os.environ['LOCAL_RANK'], bool(int(new_best0)))) 
                        self.logger.info('                      : new_best={}'.format(new_best))
                        #assert new_best0==new_best, (new_best0, new_best)
                    if distributed:
                        torch.distributed.barrier()
                    

                    if distributed and is_main_process():
                        self.logger.info('broadcast from 0 to others: stop={}'.format(self.stop))
                        torch.distributed.broadcast(torch.tensor(int(self.stop)).cuda(), src=0)
                    elif distributed:
                        stop0 = torch.empty(1,).cuda()
                        torch.distributed.broadcast(stop0, src=0)  
                        self.logger.info('broadcast from 0 to {}: stop0={} '.format(os.environ['LOCAL_RANK'], int(stop0)))    
                        self.logger.info('                      : stop={} '.format(self.stop))
                        #assert stop0==self.stop, (stop0, self.stop)
                    if distributed:
                        torch.distributed.barrier()

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    # append to validation report
                    if is_main_process():
                        self._add_report(
                            current_lr=current_lr,
                            valid_scores=val_res["valid_scores"],
                            valid_recognition_loss=val_res["valid_recognition_loss"]
                            if self.do_recognition
                            else None,
                            valid_translation_loss=val_res["valid_translation_loss"]
                            if self.do_translation
                            else None,
                            valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                            eval_metric=self.eval_metric,
                            new_best=new_best,
                        )
                        self.logger.info(
                            "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                            "Recognition Beam Size: %d\t"
                            "Translation Beam Size: %d\t"
                            "Translation Beam Alpha: %d\n\t"
                            "Recognition Loss: %4.5f\t"
                            "Translation Loss: %4.5f\t"
                            "PPL: %4.5f\n\t"
                            "Eval Metric: %s\n\t"
                            "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                            "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                            "CHRF %.2f\t"
                            "ROUGE %.2f",
                            epoch_no + 1,
                            self.steps,
                            valid_duration,
                            self.eval_recognition_beam_size if self.do_recognition else -1,
                            self.eval_translation_beam_size if self.do_translation else -1,
                            self.eval_translation_beam_alpha if self.do_translation else -1,
                            val_res["valid_recognition_loss"]
                            if self.do_recognition
                            else -1,
                            val_res["valid_translation_loss"]
                            if self.do_translation
                            else -1,
                            val_res["valid_ppl"] if self.do_translation else -1,
                            self.eval_metric.upper(),
                            # WER
                            val_res["valid_scores"]["wer"] if self.do_recognition else -1,
                            val_res["valid_scores"]["wer_scores"]["del_rate"]
                            if self.do_recognition
                            else -1,
                            val_res["valid_scores"]["wer_scores"]["ins_rate"]
                            if self.do_recognition
                            else -1,
                            val_res["valid_scores"]["wer_scores"]["sub_rate"]
                            if self.do_recognition
                            else -1,
                            # BLEU
                            val_res["valid_scores"]["bleu"] if self.do_translation else -1,
                            val_res["valid_scores"]["bleu_scores"]["bleu1"]
                            if self.do_translation
                            else -1,
                            val_res["valid_scores"]["bleu_scores"]["bleu2"]
                            if self.do_translation
                            else -1,
                            val_res["valid_scores"]["bleu_scores"]["bleu3"]
                            if self.do_translation
                            else -1,
                            val_res["valid_scores"]["bleu_scores"]["bleu4"]
                            if self.do_translation
                            else -1,
                            # Other
                            val_res["valid_scores"]["chrf"] if self.do_translation else -1,
                            val_res["valid_scores"]["rouge"] if self.do_translation else -1,
                        )

                        self._log_examples(
                            sequences=[s for s in valid_data.sequence],
                            gls_references=val_res["gls_ref"]
                            if self.do_recognition
                            else None,
                            gls_hypotheses=val_res["gls_hyp"]
                            if self.do_recognition
                            else None,
                            txt_references=val_res["txt_ref"]
                            if self.do_translation
                            else None,
                            txt_hypotheses=val_res["txt_hyp"]
                            if self.do_translation
                            else None,
                        )

                        valid_seq = [s for s in valid_data.sequence]
                        # store validation set outputs and references
                        if self.do_recognition:
                            self._store_outputs(
                                "dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls"
                            )
                            self._store_outputs(
                                "references.dev.gls", valid_seq, val_res["gls_ref"]
                            )

                        if self.do_translation:
                                self._store_outputs(
                                    "dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt"
                                )
                                self._store_outputs(
                                    "references.dev.txt", valid_seq, val_res["txt_ref"]
                                )
                    

                    if distributed:
                        torch.distributed.barrier()
                        
                if self.stop: 
                    break
                

            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else: #executed when the loop is not terminated by 'break'
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        

        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )
        if is_main_process():
            self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch, update: bool = True) -> (Tensor, Tensor):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        :return normalized_translation_loss: Normalized translation loss
        """
        recognition_loss, translation_loss = get_loss_for_batch(
            model=self.model,
            batch=batch,
            recognition_loss_function=self.recognition_loss_function
            if self.do_recognition
            else None,
            translation_loss_function=self.translation_loss_function
            if self.do_translation
            else None,
            recognition_loss_weight=self.recognition_loss_weight
            if self.do_recognition
            else None,
            translation_loss_weight=self.translation_loss_weight
            if self.do_translation
            else None,
        )

        # normalize translation loss
        if self.do_translation:
            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        # TODO (Cihan): Add Gloss Token normalization (?)
        #   I think they are already being normalized by batch
        #   I need to think about if I want to normalize them by token.

        # (Yutong):
        # ctcloss_level: 
        # sentence reduction='sum'  / num_sent (batch_size)*(batch_multiplier)
        # token  reduction='mean' / (batch_multiplier)
        if self.do_recognition:
            if self.ctcloss_level=='sentence':
                gls_normalization_factor = batch.num_seqs
            else: #'token'
                gls_normalization_factor = 1
            normalized_recognition_loss = recognition_loss / (
                gls_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        # compute gradients
        total_loss.backward()


        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.module.parameters())

        if update:

            self.optimizer.step()
            self.optimizer.zero_grad()
            # increment step counter
            self.steps += 1

        # increment token counter
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(
        self,
        current_lr,
        valid_scores: Dict,
        valid_recognition_loss: float,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: Dictionary of validation scores
        :param valid_recognition_loss: validation loss (sum over whole validation set)
        :param valid_translation_loss: validation loss (sum over whole validation set)
        :param valid_ppl: validation perplexity
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition Loss: {:.5f}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_recognition_loss if self.do_recognition else -1,
                    valid_translation_loss if self.do_translation else -1,
                    valid_ppl if self.do_translation else -1,
                    eval_metric,
                    # WER
                    valid_scores["wer"] if self.do_recognition else -1,
                    valid_scores["wer_scores"]["del_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["ins_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["sub_rate"]
                    if self.do_recognition
                    else -1,
                    # BLEU
                    valid_scores["bleu"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                    # Other
                    valid_scores["chrf"] if self.do_translation else -1,
                    valid_scores["rouge"] if self.do_translation else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.

        :param sequences: sign video sequence names (list of strings)
        :param txt_hypotheses: decoded txt hypotheses (list of strings)
        :param txt_references: decoded txt references (list of strings)
        :param gls_hypotheses: decoded gls hypotheses (list of strings)
        :param gls_references: decoded gls references (list of strings)
        """

        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            assert len(txt_references) == len(txt_hypotheses)
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)
            if self.do_translation:
                txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info(
                    "\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    distributed = 'WORLD_SIZE' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        cfg['device'] = torch.device('cuda:{}'.format(local_rank))
        device = cfg['device']
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else: 
        cfg['device'] = torch.device('cuda:{}'.format(0))
        device = cfg['device'] #please set CUDA_VISIBLE_DEVICES=? in the script

    train_data, dev_data, test_data, gls_vocab, txt_vocab, gls_counter, txt_counter = load_data(
        data_cfg=cfg["data"]
        )

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg, distributed=distributed)

    # store copy of original training config in model dir

    # DDP
    if distributed:
        trainer.logger.info('Distributed training, world_size={}, local_rank={}, \
                rank={}'.format(os.environ['WORLD_SIZE'], os.environ['LOCAL_RANK'],
                                          os.environ['RANK']))
    else:
        trainer.logger.info('Single-gpu training, gpu_id=0')  
    

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )

    trainer.logger.info(str(model))


    if is_main_process():
        shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
        # store the vocabs
        gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
        gls_vocab.to_file(gls_vocab_file)
        txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
        txt_vocab.to_file(txt_vocab_file)
        gls_counter_file = "{}/gls.counter".format(cfg["training"]["model_dir"])
        import json
        with open(gls_counter_file,'w') as f:
            json.dump(gls_counter, f)
        txt_counter_file = "{}/txt.counter".format(cfg["training"]["model_dir"])
        with open(txt_counter_file,'w') as f:
            json.dump(txt_counter, f)

    if distributed:
        torch.distributed.barrier()

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data, distributed=distributed)
    # Delete to speed things up as we don't need training data anymore
    del train_data, dev_data, test_data

    # predict with the best model on validation and test
    # (if test data is available)
    if is_main_process():
        ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
        output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
        output_path = os.path.join(trainer.model_dir, output_name)
        logger = trainer.logger
        del trainer
        test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    assert 'LOCAL_RANK' in os.environ, 'Only support distributed training now!'
    train(cfg_file=args.config)

