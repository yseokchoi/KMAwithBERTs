import os

import random
import easydict
import numpy as np
from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from loss import NMTLossCompute, LabelSmoothingLoss
from optimizers import BertOptimizer
from model_saver import build_model_saver
from statistics import Statistics
from logging_upper import init_logger, logger
import report_manager as rm

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertModel

#torch.set_printoptions(edgeitems=20)

batch_item = namedtuple("batch", ["sentences", "batch_size", "src_inputs", "tgt_inputs", "tgt_outputs", "tgt_spcs", "src_lengths", "tgt_lengths"])


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)

def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            if param.requires_grad:
                enc += param.nelement()
        else:
            if param.requires_grad:
                dec += param.nelement()
    return enc + dec, enc, dec

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Sentence(object):
    def __init__(self, sent_id, src_sentence, tgt_sentence, src_id, tgt_id, tgt_spc, src_length, tgt_length):
        self.sent_id = sent_id
        self.src_sentence = src_sentence
        self.tgt_sentence = tgt_sentence
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.tgt_spc = tgt_spc
        self.src_length = src_length
        self.tgt_length = tgt_length

    def get_src_length(self):
        return self.src_length

    def get_tgt_length(self):
        return self.tgt_length

    def get_length(self):
        return (self.src_length, self.tgt_length)


class MorphologicalDataset(Dataset):
    def __init__(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_voc_size = len(src_vocab)
        self.tgt_voc_size = len(tgt_vocab)

        self.sentence = []
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.sentence[idx]

        return (data.sent_id, data.src_sentence, data.tgt_sentence, data.src_id, data.tgt_id, data.tgt_spc, data.src_length, data.tgt_length)

    def insert(self, sent_id, src_sentence, tgt_sentence, src_id, tgt_id, tgt_spc, src_length, tgt_length):
        sent = Sentence(sent_id, src_sentence, tgt_sentence, src_id, tgt_id, tgt_spc, src_length, tgt_length)
        self.sentence.append(sent)
        self.length = len(self.sentence)

    def read(self, src_file, tgt_file, tgt_spc_file):
        logger.info("Reading file: {} / {} / {}".format(src_file, tgt_file, tgt_spc_file))
        f_src = [x.strip() for x in open(src_file).readlines() if x != ""]
        f_tgt = [x.strip() for x in open(tgt_file).readlines() if x != ""]
        f_t_spc = [[int(y) for y in x.strip().split(" ")] for x in open(tgt_spc_file).readlines() if x != ""]

        for sent_id, (src_sent, tgt_sent, tgt_spc) in enumerate(zip(f_src, f_tgt, f_t_spc)):
            if (sent_id + 1) % 1000 == 0:
                logger.info("Reading line: {}".format(sent_id + 1))
            src_sentence = src_sent
            tgt_sentence = tgt_sent
            src_sent = ["[CLS]"] + src_sent.split(" ") + ["[SEP]"]
            tgt_sent = ["[CLS]"] + tgt_sent.split(" ") + ["[SEP]"]

            src_sent_id = [self.src_vocab.get(x, self.src_vocab["[UNK]"]) for x in src_sent]
            tgt_sent_id = [self.tgt_vocab.get(x, self.tgt_vocab["[UNK]"]) for x in tgt_sent]
            tgt_spc = [1] + tgt_spc + [1]

            src_length = len(src_sent)
            tgt_length = len(tgt_sent)

            self.insert(sent_id=sent_id,
                        src_sentence=src_sentence,
                        tgt_sentence=tgt_sentence,
                        src_id=src_sent_id,
                        tgt_id=tgt_sent_id,
                        tgt_spc=tgt_spc,
                        src_length=src_length,
                        tgt_length=tgt_length)


class NMAModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMAModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _run_encoder(self, src, src_lengths):
        src_mask = sequence_mask(src_lengths, src.size(1))
        enc_states = self.encoder(input_ids=src,
                                  attention_mask=src_mask,
                                  token_type_ids=None,
                                  position_ids=None,
                                  head_mask=None)[0]
        return enc_states

    def _run_decoder(self, tgt, memory_bank, memory_lengths, step=None):
        input_shape = tgt.size()
        memory_mask = sequence_mask(memory_lengths, memory_bank.size(1))
        if step is not None:
            position_ids = torch.tensor([step], dtype=torch.long, device=tgt.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        outputs = self.decoder(input_ids=tgt,
                               attention_mask=None,
                               token_type_ids=None,
                               position_ids=position_ids,
                               head_mask=None,
                               encoder_hidden_states=memory_bank,
                               encoder_attention_mask=memory_mask,
                               output_attentions=True,
                               step=step)
        dec_out = outputs[0]
        attns = outputs[-1][-1][-1][:, 0, :, :]

        return dec_out, attns

    def forward(self, src, tgt, src_lengths, tgt_lengths, step=None):
        """
        :param src (LongTensor): (batch, src_len)
        :param tgt (LongTensor): (batch, tgt_len)
        :param lengths: (LongTensor): (batch, )
        :return:
        """
        src_mask = sequence_mask(src_lengths)
        enc_states = self.encoder(input_ids=src,
                                  attention_mask=src_mask,
                                  token_type_ids=None,
                                  position_ids=None,
                                  head_mask=None)[0]

        tgt_mask = sequence_mask(tgt_lengths)
        outputs = self.decoder(input_ids=tgt,
                               attention_mask=tgt_mask,
                               token_type_ids=None,
                               position_ids=None,
                               head_mask=None,
                               encoder_hidden_states=enc_states,
                               encoder_attention_mask=src_mask,
                               output_attentions=True,
                               step=step)

        dec_out = outputs[0]
        attns = outputs[-1][-1][-1][:, 0, :, :]

        return dec_out, attns

class Trainer(object):
    def __init__(self, model, train_loss, valid_loss, optim, shard_size=0, norm_method="sents", accum_count=1, accum_steps=0, n_gpu=1, gpu_rank=1, report_manager=None, model_saver=None, model_dtype='fp32', tgt_pad_idx=0):
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = 1
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.model_dtype = model_dtype
        self.tgt_pad_idx = tgt_pad_idx

        self.model.train()

    def _accum_count(self, step):
        if step > self.accum_steps:
            _accum = self.accum_count_l
        else:
            _accum = self.accum_count
        return _accum


    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == 'tokens':
                num_tokens = batch.tgt_outputs.ne(self.tgt_pad_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size

            if len(batches) == self.accum_count:
                yield batches, normalization
                batches = []
                normalization = 0

        if batches:
            yield batches, normalization

    def _start_report_manager(self, start_time=None):
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        if self.report_manager is not None:
            return self.report_manager.report_training(step, num_steps, learning_rate, report_stats, multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        if self.report_manager is not None:
            return self.report_manager.report_step(learning_rate, step, train_stats=train_stats, valid_stats=valid_stats)

    def validate(self, valid_iter, moving_average=None):
        valid_model = self.model

        valid_model.eval()

        with torch.no_grad():
            stats = Statistics()

            for batch in valid_iter:
                target_size = batch.tgt_inputs.size(1)
                trunc_size = batch.tgt_inputs.size(1)

                src_inputs = batch.src_inputs
                src_lengths = batch.src_lengths

                tgt_inputs = batch.tgt_inputs
                tgt_lengths = batch.tgt_lengths

                outputs, attns = valid_model(src_inputs, tgt_inputs, src_lengths, tgt_lengths)

                _, batch_stats = valid_loss(batch, outputs, attns)

                stats.update(batch_stats)

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats, report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            if self.accum_count == 1:
                self.optim.zero_grad()

            target_size = batch.tgt_inputs.size(1)
            trunc_size = batch.tgt_inputs.size(1)

            src_inputs = batch.src_inputs
            src_lengths = batch.src_lengths

            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_inputs = batch.tgt_inputs
            tgt_lengths = batch.tgt_lengths

            outputs, attns = self.model(src_inputs, tgt_inputs, src_lengths, tgt_lengths)

            loss, batch_stats = self.train_loss(batch, outputs, attns, normalization=normalization, shard_size=self.shard_size, trunc_start=0, trunc_size=trunc_size)

            if loss is not None:
                self.optim.backward(loss)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if self.accum_count == 1:
                self.optim.step()

        if self.accum_count > 1:
            self.optim.step()

    def train(self, train_iter, train_steps, save_checkpoint_steps=5000, valid_iter=None, valid_steps=10000):
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        any_finished = True
        while any_finished:
            logger.info("training epochs..")
            for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
                step = self.optim.training_step

                self._gradient_accumulation(batches, normalization, total_stats, report_stats)

                report_stats = self._maybe_report_training(step, train_steps, self.optim.learning_rate(), report_stats)

                if valid_iter is not None and step % valid_steps == 0:
                    valid_stats = self.validate(valid_iter)

                    self._report_step(self.optim.learning_rate(), step, valid_stats=valid_stats)

                if (self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0)):
                    self.model_saver.save(step)

                if train_steps > 0 and step >= train_steps:
                    any_finished = False
                    break

        # if (self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0)):
        #     self.model_saver.save(step)


def read_vocab(vocab_path):
    vocab = {}

    with open(vocab_path, "r") as f:
        lines = f.readlines()[2:]

    for idx, line in enumerate(lines, start=0):
        token = line.strip().split("\t")[0]
        vocab.setdefault(token, idx)

    return vocab


def collate_fn(batches, device=-1, token_size=512, src_pad_id=0, tgt_pad_id=0):
    batch_size = len(batches)
    sentences = []
    src_inputs = np.empty([batch_size, token_size], dtype=np.int64)
    tgt_inputs = np.empty([batch_size, token_size], dtype=np.int64)
    tgt_outputs = np.empty([batch_size, token_size], dtype=np.int64)
    tgt_spcs = np.empty([batch_size, token_size], dtype=np.int64)

    src_lengths = np.zeros(batch_size, dtype=np.int64)
    tgt_lengths = np.zeros(batch_size, dtype=np.int64)

    # sent_id, src_sent, tgt_sent, src_ids, tgt_ids, tgt_spcs, src_len, tgt_len = batch
    for idx, batch in enumerate(batches):
        sent_id, src_sent, tgt_sent, src_id, tgt_id, tgt_spc, src_len, tgt_len = batch

        sentences.append(batch)

        tgt_len_ = tgt_len - 1
        src_inputs[idx, :src_len] = src_id
        src_inputs[idx, src_len:] = src_pad_id

        tgt_inputs[idx, :tgt_len_] = tgt_id[:-1]
        tgt_inputs[idx, tgt_len_:] = tgt_pad_id

        tgt_outputs[idx, :tgt_len_] = tgt_id[1:]
        tgt_outputs[idx, tgt_len_:] = tgt_pad_id

        tgt_spcs[idx, :tgt_len_] = tgt_spc[1:]
        tgt_spcs[idx, tgt_len_:] = tgt_pad_id

        src_lengths[idx] = src_len
        tgt_lengths[idx] = tgt_len_

    _max_src_length = src_lengths.max()
    _max_tgt_length = tgt_lengths.max()

    # truncated
    src_inputs = src_inputs[:, :_max_src_length]
    tgt_inputs = tgt_inputs[:, :_max_tgt_length]
    tgt_outputs = tgt_outputs[:, :_max_tgt_length]
    tgt_spcs = tgt_spcs[:, :_max_tgt_length]

    # numpy to tensor
    src_inputs = torch.tensor(src_inputs)
    tgt_inputs = torch.tensor(tgt_inputs)
    tgt_outputs = torch.tensor(tgt_outputs)
    tgt_spcs = torch.tensor(tgt_spcs)
    src_lengths = torch.tensor(src_lengths)
    tgt_lengths = torch.tensor(tgt_lengths)

    # cpu or gpu
    if device != -1:
        t_device = torch.device("cuda", device)
    else:
        t_device = torch.device("cpu")

    f_batches = batch_item(
        sentences=sentences,
        batch_size=batch_size,
        src_inputs=src_inputs.to(t_device),
        tgt_inputs=tgt_inputs.to(t_device),
        tgt_outputs=tgt_outputs.to(t_device),
        tgt_spcs=tgt_spcs.to(t_device),
        src_lengths=src_lengths.to(t_device),
        tgt_lengths=tgt_lengths.to(t_device)
    )

    return f_batches


def build_base_model(model_opt, gpu, checkpoint=None):
    # cpu or gpu
    if gpu != -1:
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")

    enc_config = BertConfig.from_pretrained(model_opt.encoder_config_file)
    dec_config = BertConfig.from_pretrained(model_opt.decoder_config_file)

    encoder = BertModel(enc_config)
    decoder = BertModel(dec_config)

    # Build Generator.
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_config.hidden_size, dec_config.vocab_size),
        Cast(torch.float32),
        gen_func
    )

    # Build Seperator.
    sep_func = nn.LogSoftmax(dim=-1)
    separator = nn.Sequential(
        nn.Linear(dec_config.hidden_size, 2),
        Cast(torch.float32),
        sep_func
    )

    model = NMAModel(encoder, decoder)

    # Load the model states from checkpoint
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)
        generator.load_state_dict(checkpoint["generator"], strict=False)
        separator.load_state_dict(checkpoint["separator"], strict=False)

    # Initialization Encoder, Decoder
    else:
         # Initialization Generator, Separator
        if model_opt.param_init != 0.0:
            range = model_opt.param_init
            for p in generator.parameters():
                p.data.uniform_(-range, range)
                
            for p in separator.parameters():
                p.data.uniform_(-range, range)
                
        if model_opt.param_init_glorot:
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            
            for p in separator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        
        if model_opt.encoder_bert_path != "":
            enc_state_dict = torch.load(model_opt.encoder_bert_path, map_location="cpu")
            enc_dict = {".".join(k.split(".")[1:]): v for k, v in enc_state_dict.items()}
            for n, p in model.encoder.named_parameters():
                if n in enc_dict:
                    p.data.copy_(enc_dict[n])
                    logger.info("Updated parameter: encoder.{}".format(n))
        else:
            if model_opt.param_init != 0.0:
                range = model_opt.param_init
                for p in model.encoder.parameters():
                    p.data.uniform_(-range, range)
                
                    
            if model_opt.param_init_glorot:
                for p in model.encoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                
               
        if model_opt.decoder_bert_path != "":
            dec_state_dict = torch.load(model_opt.decoder_bert_path, map_location="cpu")
            dec_dict = {".".join(k.split(".")[1:]): v for k, v in dec_state_dict.items()}
            for n, p in model.decoder.named_parameters():
                if n in dec_dict:   
                    p.data.copy_(dec_dict[n])
                    logger.info("Updated parameter: decoder.{}".format(n))
        else:
            if model_opt.param_init != 0.0:
                range = model_opt.param_init
                for p in model.decoder.parameters():
                    p.data.uniform_(-range, range)
                
                    
            if model_opt.param_init_glorot:
                for p in model.decoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        
    model.generator = generator
    model.separator = separator

    model.to(device)

    return model, enc_config, dec_config


# test model
args = easydict.EasyDict({
    "encoder_config_file": "models/12_4_model/encoder_config.json",
    "decoder_config_file": "models/12_4_model/decoder_config.json",
    "encoder_bert_path": "/media/4T/data_pool/ETRI_BERT/003_bert_eojeol_pytorch/pytorch_model.bin",
    "decoder_bert_path": "/media/4T/data_pool/ETRI_BERT/001_bert_morp_pytorch/pytorch_model.bin",
    "encoder_vocabs": "/media/4T/data_pool/ETRI_BERT/003_bert_eojeol_pytorch/vocab.korean.rawtext.list",
    "decoder_vcoabs": "/media/4T/data_pool/ETRI_BERT/001_bert_morp_pytorch/vocab.korean_morp.list",
    "save_model": "models/12_4_model/model-70-both-bert",
    "save_checkpoint_steps": 2000,
    "log_file": "models/12_4_model/log-70-both-bert.txt",
    "keep_checkpoint":-1,
    "gpu": 0,
    "lambda_coverage":0.0,
    "optim": "adam",
    "learning_rate": 0.0,
    "adam_beta1": 0.99,
    "adam_beta2": 0.998,
    "enc_learning_rate": 5e-3,
    "dec_learning_rate": 1e-3,
    "etc_learning_rate": 1e-3,
    "max_grad_norm": 10.0,
    "train_from": "",
    "tensorboard": False,
    "train_steps": 100000,
    "valid_steps": 2000,
    "report_every": 100,
    "train_batch_size": 64,
    "valid_batch_size": 10,
    "eval_batch_size": 10,
    "max_generator_batches": 8,
    "param_init": 0.1,
    "param_init_glorot": True,
    "decay_method": "none",
    "start_decay_steps": 20000,
    "decay_steps": 20000,
    "learning_rate_decay": 0.5,
    "warmup_steps": 4000,
    "label_smoothing": 0.0,
    "accum_count": 1,
    "accum_steps": 0,
    "normalization": "sents",
    "seed": 0

})

if args.seed > 0:
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.gpu != -1:
        torch.cuda.manual_seed(seed)

init_logger(args.log_file)

src_vocab_path = args.encoder_vocabs
tgt_vocab_path = args.decoder_vocabs

train_src_file_path = "./datasets/train.sent.tok.0.7"
train_tgt_file_path = "./datasets/train.tagging.tok.0.7"
train_spc_file_path = "./datasets/train.tagging.tok.spc.0.7"

valid_src_file_path = "./datasets/dev.sent.tok"
valid_tgt_file_path = "./datasets/dev.tagging.tok"
valid_spc_file_path = "./datasets/dev.tagging.tok.spc"

tr_batch_size = args.train_batch_size
va_batch_size = args.valid_batch_size
# ev_batch_size = args.eval_batch_size
shard_size = args.max_generator_batches

device = 0

src_vocab = read_vocab(src_vocab_path)
tgt_vocab = read_vocab(tgt_vocab_path)

train_dataset = MorphologicalDataset(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
valid_dataset = MorphologicalDataset(src_vocab=src_vocab, tgt_vocab=tgt_vocab)

train_dataset.read(train_src_file_path, train_tgt_file_path, train_spc_file_path)
valid_dataset.read(valid_src_file_path, valid_tgt_file_path, valid_spc_file_path)

train_sampler = RandomSampler(train_dataset)
valid_sampler = SequentialSampler(valid_dataset)

train_iterator = DataLoader(train_dataset, sampler=train_sampler, batch_size=tr_batch_size, collate_fn=partial(collate_fn, device=device, src_pad_id=src_vocab["[PAD]"], tgt_pad_id=tgt_vocab["[PAD]"]))
valid_iterator = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=va_batch_size, collate_fn=partial(collate_fn, device=device, src_pad_id=src_vocab["[PAD]"], tgt_pad_id=tgt_vocab["[PAD]"]))

# cpu or gpu
if args.gpu != -1:
    device = torch.device("cuda", args.gpu)
else:
    device = torch.device("cpu")

model, enc_config, dec_config = build_base_model(args, args.gpu)

optim = BertOptimizer.from_opt(model, args, checkpoint=None)

print(model)
n_params, enc, dec = _tally_parameters(model)
logger.info('encoder: %d' % enc)
logger.info('decoder: %d' % dec)
logger.info('* number of parameters: %d' % n_params)

model_args = args
model_saver = build_model_saver(model_args, args, model, optim)

if args.label_smoothing > 0:
    gen_criterion = LabelSmoothingLoss(args.label_smoothing, dec_config.vocab_size, ignore_index=tgt_vocab["[PAD]"])
else:
    gen_criterion = nn.NLLLoss(ignore_index=tgt_vocab["[PAD]"], reduction='sum')
sep_criterion = nn.NLLLoss(reduction='none')

train_loss = NMTLossCompute(gen_criterion, sep_criterion, model.generator, model.separator, lambda_coverage=args.lambda_coverage)
train_loss.to(device)

valid_loss = NMTLossCompute(gen_criterion, sep_criterion, model.generator, model.separator, lambda_coverage=args.lambda_coverage)
valid_loss.to(device)

report_manager = rm.build_report_manager(args, args.gpu)

trainer = Trainer(model, train_loss, valid_loss, optim, shard_size=shard_size, norm_method="tokens", accum_count=args.accum_count, accum_steps=args.accum_steps, n_gpu=1, gpu_rank=args.gpu, report_manager=report_manager, model_saver=model_saver, model_dtype='fp32', tgt_pad_idx=tgt_vocab["[PAD]"])
trainer.train(train_iterator, args.train_steps, save_checkpoint_steps=args.save_checkpoint_steps, valid_iter=valid_iterator, valid_steps=args.valid_steps)





