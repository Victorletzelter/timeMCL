# This file was adpated from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/dataset/loader.py
# under MIT License.

from typing import Optional
import itertools
from torch.utils.data import IterableDataset
from gluonts.dataset.common import Dataset
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import Cyclic, PseudoShuffled, Cached

# this class is basically doing the same as GluonTS. We authors of pytorch-ts just combined Transformation and creation of iterator in the same function.


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset)),
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            shuffled = PseudoShuffled(
                self.transformed_dataset,
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
            return iter(shuffled)
