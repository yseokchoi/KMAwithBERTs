import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from math import sqrt
import functools


def build_torch_optimizer_for_bert(model, opt):
    """
    no_decay = ["bias", "LayerNorm.weight"]
    encoder_params = [
            {
                "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
            ]
    decoder_params = [
            {
                "params": [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]
    etc_params = [p for p in model.generator.parameters() if p.requires_grad] + [p for p in model.separator.parameters() if p.requires_grad]
    """
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    # decoder_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad and 'cross' not in n]
    # etc_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad and 'cross' in n] + [p for p in model.generator.parameters() if p.requires_grad] + [p for p in model.separator.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]
    etc_params = [p for p in model.generator.parameters() if p.requires_grad] + [p for p in model.separator.parameters() if p.requires_grad]

    betas = [opt.adam_beta1, opt.adam_beta2]
    encoder_lr = opt.learning_rate if opt.enc_learning_rate == 0.0 else opt.enc_learning_rate
    decoder_lr = opt.learning_rate if opt.dec_learning_rate == 0.0 else opt.dec_learning_rate
    etc_lr = opt.learning_rate if opt.etc_learning_rate == 0.0 else opt.etc_learning_rate

    if opt.optim == 'sgd':
        if len(encoder_params) > 0 and len(decoder_params) > 0:
            optimizer = {
                "encoder": optim.SGD(encoder_params, lr=encoder_lr),
                "decoder": optim.SGD(decoder_params, lr=decoder_lr),
                "etc": optim.SGD(etc_params, lr=etc_lr)
            }
        elif len(decoder_params) > 0:
            optimizer = {
                "decoder": optim.SGD(decoder_params, lr=decoder_lr),
                "etc": optim.SGD(etc_params, lr=etc_lr)
            }
        else:
            optimizer = {
                "etc": optim.SGD(etc_params, lr=etc_lr)
            }
    elif opt.optim == 'adagrad':
        if len(encoder_params) > 0 and len(decoder_params) > 0:
            optimizer = {
                "encoder": optim.Adagrad(
                    encoder_params,
                    lr=encoder_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init),
                "decoder": optim.Adagrad(
                    decoder_params,
                    lr=decoder_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init),
                "etc": optim.Adagrad(
                    etc_params,
                    lr=etc_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init)
            }
        elif len(decoder_params) > 0:
            optimizer = {
                "decoder": optim.Adagrad(
                    decoder_params,
                    lr=decoder_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init),
                "etc": optim.Adagrad(
                    etc_params,
                    lr=etc_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init)
            }
        else:
            optimizer = {
                "etc": optim.Adagrad(
                    etc_params,
                    lr=etc_lr,
                    initial_accumulator_value=opt.adagrad_accumlator_init)
            }
    elif opt.optim == 'adadelta':
        if llen(encoder_params) > 0 and len(decoder_params) > 0:
            optimizer = {
                "encoder": optim.Adadelta(encoder_params, lr=encoder_lr),
                "decoder": optim.Adadelta(decoder_params, lr=decoder_lr),
                "etc": optim.Adadelta(etc_params, lr=etc_lr)
            }
        elif len(decoder_params) > 0:
            optimizer = {
                "decoder": optim.Adadelta(decoder_params, lr=decoder_lr),
                "etc": optim.Adadelta(etc_params, lr=etc_lr)
            }
        else:
            optimizer = {
                "etc": optim.Adadelta(etc_params, lr=etc_lr)
            }
    elif opt.optim == 'adam':
        if len(encoder_params) > 0 and len(decoder_params) > 0:
            optimizer = {
                "encoder": optim.Adam(encoder_params, lr=encoder_lr, betas=betas, eps=1e-12),
                "decoder": optim.Adam(decoder_params, lr=decoder_lr, betas=betas, eps=1e-12),
                "etc": optim.Adam(etc_params, lr=etc_lr, betas=betas, eps=1e-12)
            }
        elif len(decoder_params) > 0:
            optimizer = {
                "decoder": optim.Adam(decoder_params, lr=decoder_lr, betas=betas, eps=1e-12),
                "etc": optim.Adam(etc_params, lr=etc_lr, betas=betas, eps=1e-12)
            }
        else:
            optimizer = {
                "etc": optim.Adam(etc_params, lr=etc_lr, betas=betas, eps=1e-12)
            }
    else:
        raise ValueError("Invalid optimizer type: " + opt.optim)

    return optimizer


def make_learning_rate_decay_fn(opt):
    """Returns the learning decay function from options."""
    if opt.decay_method == 'noam':
        return functools.partial(
            noam_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)
    elif opt.decay_method == 'noamwd':
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)
    elif opt.decay_method == 'rsqrt':
        return functools.partial(
            rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)
    else:
        return None


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)))


def noamwd_decay(step, warmup_steps,
                 model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)) *
        rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))

class BertOptimizer(object):
    def __init__(self, optimizer, learning_rate, learning_rate_decay_fn=None, max_grad_norm=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1

    @classmethod
    def from_opt(cls, model, opt, checkpoint=None):
        optim_opt = opt
        optim_state_dict = None

        if opt.train_from and checkpoint is not None:
            optim = checkpoint["optim"]
            ckpt_opt = checkpoint["opt"]
            ckpt_state_dict = {}
            if isinstance(optim, BertOptimizer):
                ckpt_state_dict["training_step"] = optim._step + 1
                ckpt_state_dict["decay_step"] = optim._step + 1
                ckpt_state_dict["optimizer"] = optim.optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            if opt.reset_optim == 'none':
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
            elif opt.reset_optim == 'all':
                pass
            elif opt.reset_optim == 'states':
                optim_opt = ckpt_opt
                optim_state_dict = ckpt_state_dict
                del optim_state_dict["optimizer"]
            elif opt.reset_optim == 'keep_states':
                optim_state_dict = ckpt_state_dict

        lr = {
            "encoder": optim_opt.learning_rate if optim_opt.enc_learning_rate == 0.0 else optim_opt.enc_learning_rate,
            "decoder": optim_opt.learning_rate if optim_opt.dec_learning_rate == 0.0 else optim_opt.dec_learning_rate,
            "etc": optim_opt.learning_rate if optim_opt.etc_learning_rate == 0.0 else optim_opt.etc_learning_rate
        }
        optimizer = cls(
            build_torch_optimizer_for_bert(model, optim_opt),
            lr,
            learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt),
            max_grad_norm=optim_opt.max_grad_norm
        )
        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)

        return optimizer

    @property
    def training_step(self):
        return self._training_step

    def set_learning_rate(self, lr):
        self._learning_rate["encoder"] = lr["encoder"]
        self._learning_rate["decoder"] = lr["decoder"]
        self._learning_rate["etc"] = lr["etc"]

    def get_learning_rate(self):
        return self._learning_rate

    def learning_rate(self, rate=None):
        if rate is not None:
            if self._learning_rate_decay_fn is None:
                return rate
            scale = self._learning_rate_decay_fn(self._decay_step)
            lr = scale * rate
            return lr
        else:
            if self._learning_rate_decay_fn is None:
                return self._learning_rate
            scale = self._learning_rate_decay_fn(self._decay_step)
            lr = {stack: scale * r if stack in self._optimizer.keys() else r for stack, r in self._learning_rate.items()}
            return lr

    def state_dict(self):
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": {stack:optimizer.state_dict() for stack, optimizer in self._optimizer.items()}
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored.
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            for stack, optimizer in self._optimizer.items():
                self._optimizer[stack].load_state_dict(state_dict['optimizer'][stack])

    def zero_grad(self):
        for stack, optimizer in self._optimizer.items():
            optimizer.zero_grad()

    def backward(self, loss):
        loss.backward()

    def step(self):
        for stack, optimizer in self._optimizer.items():
            learning_rate = self.learning_rate(self._learning_rate[stack])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
                if self._max_grad_norm > 0:
                    clip_grad_norm_(group['params'], self._max_grad_norm)
            optimizer.step()
        self._decay_step += 1
        self._training_step += 1