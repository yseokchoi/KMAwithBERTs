import os, tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import easydict
import numpy as np
from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from beam import GNMTGlobalScorer
from utils import tile
from beam_search import BeamSearch
from statistics import Statistics
from logging_upper import init_logger, logger
import report_manager as rm

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertModel

batch_item = namedtuple("batch", ["sentences", "src_inputs", "tgt_inputs", "tgt_outputs", "tgt_spcs", "src_lengths", "tgt_lengths"])


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
        f_src = [x.strip() for x in open(src_file).readlines() if x != ""]
        f_tgt = [x.strip() for x in open(tgt_file).readlines() if x != ""]
        f_t_spc = [[int(y) for y in x.strip().split(" ")] for x in open(tgt_spc_file).readlines() if x != ""]

        for sent_id, (src_sent, tgt_sent, tgt_spc) in enumerate(zip(f_src, f_tgt, f_t_spc)):
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
        src_mask = sequence_mask(src_lengths)
        enc_states = self.encoder(input_ids=src,
                                  attention_mask=src_mask,
                                  token_type_ids=None,
                                  position_ids=None,
                                  head_mask=None)[0]
        return enc_states

    def _run_decoder(self, tgt, memory_bank, memory_lengths, max_length, step=None):
        input_shape = tgt.size()
        memory_mask = sequence_mask(memory_lengths, max_length)
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
        src_inputs=src_inputs.to(t_device),
        tgt_inputs=tgt_inputs.to(t_device),
        tgt_outputs=tgt_outputs.to(t_device),
        tgt_spcs=tgt_spcs.to(t_device),
        src_lengths=src_lengths.to(t_device),
        tgt_lengths=tgt_lengths.to(t_device)
    )

    return f_batches


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.from_model

    checkpoint = torch.load(model_path, map_location= lambda storage, loc: storage)
    model_opt = checkpoint["opt"]

    if model_opt.encoder_bert_path != opt.encoder_bert_path:
        model_opt.encoder_bert_path = opt.encoder_bert_path
    if model_opt.decoder_bert_path != opt.decoder_bert_path:
        model_opt.decoder_bert_path = opt.decoder_bert_path

    model = build_base_model(model_opt, opt.gpu, checkpoint)

    model.eval()
    model.generator.eval()
    model.separator.eval()

    return model, model_opt


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
                if p.dim() > 1:
                    p.data.uniform_(-range, range)

            for p in separator.parameters():
                if p.dim() > 1:
                    p.data.uniform_(-range, range)

        if model_opt.param_init_glorot:
            for p in model.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

            for p in model.decoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if model_opt.encoder_bert_path != "":
            enc_state_dict = torch.load(model_opt.encoder_bert_path, map_location="cpu")
            enc_dict = {".".join(k.split(".")[1:]): v for k, v in enc_state_dict.items()}
            for n, p in model.encoder.named_parameters():
                if n in enc_dict:
                    p.data.copy_(enc_dict[n])
                    logger.info("Updated parameter: encoder.{}".format(n))

        if model_opt.decoder_bert_path != "":
            dec_state_dict = torch.load(model_opt.decoder_bert_path, map_location="cpu")
            dec_dict = {".".join(k.split(".")[1:]): v for k, v in dec_state_dict.items()}
            for n, p in model.decoder.named_parameters():
                if n in dec_dict:
                    p.data.copy_(dec_dict[n])
                    logger.info("Updated parameter: decoder.{}".format(n))

    model.generator = generator
    model.separator = separator

    model.to(device)

    return model


def decode_and_generate(model, decoder_input, memory_bank, batch, memory_lengths, max_length, step, batch_offset):
    dec_out, dec_attn = model._run_decoder(decoder_input, memory_bank, memory_lengths, max_length=max_length, step=step)

    attn = dec_attn
    log_gen_probs = model.generator(dec_out.squeeze(1))
    log_sep_probs = model.separator(dec_out.squeeze(1))

    return log_gen_probs, log_sep_probs, attn

def translate_batch(model, data_iter, global_scorer, max_length, min_length=0, ratio=0., beam_size=1, nbest=1, tgt_vocab=None, out_file=None, out_sep_file=None):
    tgt_vocab_ids = {y:x for x, y in tgt_vocab.items()}
    with torch.no_grad():
        for batch in tqdm.tqdm(data_iter):
            src_inputs = batch.src_inputs
            src_lengths = batch.src_lengths

            batch_size = src_inputs.size(0)

            memory_bank = model._run_encoder(src_inputs, src_lengths)

            model.decoder.init_state(src_inputs)

            results = {
                "predictions": None,
                "separator_predictions": None,
                "scores": None,
                "attention": None,
                "batch": batch
            }

            model.decoder.map_state(lambda state, dim: tile(state, beam_size, dim=dim))

            memory_bank = tile(memory_bank, beam_size, dim=0)
            mb_device = memory_bank.device
            memory_lengths = tile(src_lengths, beam_size, dim=0)
            max_memory_length = memory_lengths.max()

            beam = BeamSearch(
                beam_size,
                n_best=nbest,
                batch_size=batch_size,
                global_scorer=global_scorer,
                pad=tgt_vocab["[PAD]"] if tgt_vocab is not None else None,
                eos=tgt_vocab["[SEP]"] if tgt_vocab is not None else None,
                bos=tgt_vocab["[CLS]"] if tgt_vocab is not None else None,
                min_length=min_length,
                max_length=max_length,
                ratio=ratio,
                mb_device=mb_device,
                return_attention=False,
                stepwise_penalty=False,
                block_ngram_repeat=0,
                exclusion_tokens={},
                memory_lengths=memory_lengths
            )

            for step in range(max_length):
                decoder_input = beam.current_predictions.view(-1, 1) #batch x beam, 1

                log_gen_probs, log_sep_probs, attn = decode_and_generate(
                    model,
                    decoder_input,
                    memory_bank,
                    batch,
                    memory_lengths=memory_lengths,
                    max_length=max_memory_length,
                    step=step,
                    batch_offset=beam._batch_offset
                )

                beam.advance(log_gen_probs, log_sep_probs, attn)

                any_beam_is_finished = beam.is_finished.any()

                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                select_indices = beam.current_origin

                if any_beam_is_finished:
                    memory_bank = memory_bank.index_select(0, select_indices)
                    memory_lengths = memory_lengths.index_select(0, select_indices)

                model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

            results["scores"] = beam.scores
            results["predictions"] = beam.predictions
            results["separator_predictions"] = beam.sep_predictions
            results["attention"] = beam.attention

            if out_file is not None:
                for b in range(batch_size):
                    prediction = results["predictions"][b][0][:-1]
                    pred_tokens = [tgt_vocab_ids[x.item()] for x in prediction]
                    out_file.write("{}\n".format(" ".join(pred_tokens)))

            if out_sep_file is not None:
                for b in range(batch_size):
                    out_sep_file.write("{}\n".format(" ".join([str(x.item()) for x in results["separator_predictions"][b][0][:-1]])))

# test model
args = easydict.EasyDict({
    "encoder_config_file": "./models/12_12_model/encoder_config.json",
    "decoder_config_file": "./models/12_12_model/decoder_config.json",
    "encoder_bert_path": "",
    "decoder_bert_path": "",
    "gpu": 0,
    "from_model": "./models/12_4_model/model-accumulation-allbert_epoch_32000.pt",
    "batch_size": 32,
    "output": "./models/12_4_model/pred-accumulation-allbert_epoch_32000.txt",
    "nbest": 1,
    "beam_size": 5,
    "length_penalty": "none",
    "coverage_penalty": "none",
    "alpha": 0.,
    "beta": 0.,
    "max_length": 300,
    "min_length": 1
})

#test datasets
src_vocab_path = "/media/4T/data_pool/ETRI_BERT/003_bert_eojeol_pytorch/vocab.korean.rawtext.list"
tgt_vocab_path = "/media/4T/data_pool/ETRI_BERT/001_bert_morp_pytorch/vocab.korean_morp.list"

eval_src_file_path = "./datasets/test.sent.tok"
eval_tgt_file_path = "./datasets/test.tagging.tok"
eval_spc_file_path = "./datasets/test.tagging.tok.spc"

ev_batch_size = args.batch_size

device = args.gpu

src_vocab = read_vocab(src_vocab_path)
tgt_vocab = read_vocab(tgt_vocab_path)

eval_dataset = MorphologicalDataset(src_vocab=src_vocab, tgt_vocab=tgt_vocab)

eval_dataset.read(eval_src_file_path, eval_tgt_file_path, eval_spc_file_path)

eval_sampler = SequentialSampler(eval_dataset)

eval_iterator = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=ev_batch_size, collate_fn=partial(collate_fn, device=device))

model, model_opt = load_test_model(args)
print("Loaded Model: {}".format(args.from_model))

out_file = open(args.output, "w")
out_file_spc = open("{}.spc".format(args.output), "w")

scorer = GNMTGlobalScorer.from_opt(args)

translate_batch(model, eval_iterator, global_scorer=scorer, max_length=args.max_length, min_length=args.min_length, beam_size=args.beam_size, nbest=args.nbest, tgt_vocab=tgt_vocab, out_file=out_file, out_sep_file=out_file_spc)

out_file.close()
out_file_spc.close()