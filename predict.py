import os
import sys
import time
import json
import errno
import shutil
import argparse

import librosa
import soundfile as sf

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.data_loader import SpectrogramDatasetPrediction as Dataset
from decoder import GreedyDecoder, BeamCTCDecoder, Scorer, KenLMScorer
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--turns_path', default='',
                    help='Turns json file for the utterances in audio file. Ignore if whole file should be processed in one go')
parser.add_argument('--tmp_dir', default='./tmp',
                    help='Temporary directory to store individual turns\' audio')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for forward pass. If using only 1 CPU this can be ignored.=')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str, help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--trie_path', default=None, type=str, help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta1', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--lm_beta2', default=1, type=float, help='Language model word bonus (IV words)')
args = parser.parse_args()

def main():
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        scorer = None
        if args.lm_path is not None:
            scorer = KenLMScorer(labels, args.lm_path, args.trie_path)
            scorer.set_lm_weight(args.lm_alpha)
            scorer.set_word_weight(args.lm_beta1)
            scorer.set_valid_word_weight(args.lm_beta2)
        else:
            scorer = Scorer()
        decoder = BeamCTCDecoder(labels, scorer, beam_width=args.beam_width, top_paths=1, space_index=labels.index(' '), blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))

    duration = slice_audio(args.audio_path, args.turns_path, audio_conf['sample_rate'])
    turns_data = Dataset(audio_conf=audio_conf, manifest_filepath=os.path.join(args.tmp_dir,'manifest.tmp'), normalize=True)
    
    turns_loader = DataLoader(turns_data, batch_size=args.batch_size, collate_fn=_collate_fn)

    t0 = time.time()

    for data in turns_loader:
        if data.size(3) < 32:
            print('Utterance too short')
            continue
        out = model(Variable(data, volatile=True))
        out = out.transpose(0, 1)
        decoded_output = decoder.decode(out.data)
        print(decoded_output[0])
    # spect = parser.parse_audio(args.audio_path).contiguous()
    # spect = spect.view(1, 1, spect.size(0), spect.size(1))
    # out = model(Variable(spect, volatile=True))
    # out = out.transpose(0, 1)  # TxNxH
    # decoded_output = decoder.decode(out.data)
    t1 = time.time()
    # print(decoded_output[0])
    print("Decoded {0:.2f} seconds of audio in {1:.2f} seconds".format(duration, t1-t0), file=sys.stderr)
    shutil.rmtree(args.tmp_dir)

def slice_audio(audio_path, turns_path, sr):
    tmp_dir = args.tmp_dir
    make_dir(tmp_dir)
    y, _ = librosa.load(audio_path, sr = sr)
    duration = librosa.get_duration(y,sr)

    if turns_path != '':
        with open(turns_path) as f:
            turns = json.load(f)['turns']
    else:
        # Make a fake turns json
        turns = [{'start': 0.0, 'stop': duration}]
    with open(os.path.join(tmp_dir, 'manifest.tmp'),'w') as f:
        for i, turn in enumerate(turns):
            start = turn['start']
            stop = turn['stop']
            segment = y[int(start*sr):int(stop*sr)]
            write_audio(i, segment, sr, tmp_dir)
            f.write(os.path.join(tmp_dir, str(i)+'.wav\n'))
    return duration

def write_audio(name, y, sr, directory):
    audio_path = os.path.join(directory, str(name)+'.wav')
    sf.write(audio_path, y, sr, subtype='PCM_16')
    return audio_path

def make_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

def _collate_fn(batch):
    def func(p):
        return p.size(1)
    longest_sample = max(batch, key=func)
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    for x in range(minibatch_size):
        tensor = batch[x]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
    return inputs

if __name__ == '__main__':
    main()