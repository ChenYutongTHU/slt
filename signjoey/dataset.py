# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        input_data: str,
        path: str,
        fields: Tuple,
        downsample: int=1,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            input_data: feature/image,
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            if input_data == 'feature':
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                ]
            else:
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                    ("num_frames", fields[5])
                ]
        elif input_data == 'images':
            fields = fields[:2]+fields[3:]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }
                    if input_data=='image':
                        assert 'num_frames' in s
                    if 'num_frames' in s:
                        samples[seq_id]['num_frames'] = s['num_frames']
                #downsample
                if input_data=='feature':
                    samples[seq_id]['sign'] = samples[seq_id]['sign'][0::downsample,:] # L',d
                    samples[seq_id]['num_frames'] = samples[seq_id]['sign'].shape[0]
                #samples[seq_id]['num_frames']
                #samples[seq_id]['sign'] 

        examples = []
        for s in samples:
            sample = samples[s]
            if input_data == 'feature':
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            # This is for numerical stability
                            sample["sign"] + 1e-8,
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                        ],
                        fields,
                    )
                )
            else:
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                            sample["num_frames"]
                        ],
                        fields,
                    )
                )
        super().__init__(examples, fields, **kwargs)
