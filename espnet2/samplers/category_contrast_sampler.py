# Sampler that for categorical data will sample mini-batches such that
# there are at least batch_size/n categories in each mini-batch.
# there will be (n-1) positive pairs for each anchor point in the batch.

# The implementation is based on the category_balanced_sampler

import random
from collections import defaultdict
from typing import Iterator, Optional, Tuple

from typeguard import typechecked
from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler

class CategoryContrastiveSampler(AbsSampler):
    @typechecked
    def __init__(
        self,
        batch_size: int,
        category2utt_file: str,
        min_batch_size: int = 1,
        utt_per_category: int = 4,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_size > 0
        assert batch_size % utt_per_category == 0, "batch_size must be divisible by utt_per_category"
        
        random.seed(epoch)

        assert category2utt_file is not None

        self.batch_size = batch_size
        self.utt_per_category = utt_per_category
        self.category_per_batch = batch_size // utt_per_category

        # dictionary with categories as keys and corresponding utterances
        # as values
        self.category2utt = read_2columns_text(category2utt_file)
        
        # Filter out categories with insufficient utterances
        self.category2utt = {spk: utts.split() for spk, utts in self.category2utt.items() 
                            if len(utts.split()) >= self.utt_per_category}
        
        self.categories = list(self.category2utt.keys())
        
        if len(self.categories) < self.category_per_batch:
            raise ValueError(f"Not enough categories with {self.utt_per_category} utterances each. "
                             f"Found {len(self.categories)}, need at least {self.category_per_batch}.")

        self.create_batches()

    def create_batches(self):
        self.batches = []
        categories_pool = self.categories.copy()

        # compute total number of utterances
        total_utts = sum([len(utts) for utts in self.category2utt.values()])

        while len(categories_pool) >= self.category_per_batch:
            # Randomly sample categories for the batch
            batch_categories = random.sample(categories_pool, self.category_per_batch)
            batch = []
            
            for category in batch_categories:
                category_utts = random.sample(self.category2utt[category], self.utt_per_category)
                batch.extend(category_utts)
                
                # Remove used utterances from the pool
                for utt in category_utts:
                    self.category2utt[category].remove(utt)
                
                # If category has no more utterances, remove from the pool
                if len(self.category2utt[category]) < self.utt_per_category:
                    categories_pool.remove(category)

            self.batches.append(batch)

        # compute and report the % utilization of total utterances
        utilized_utts = self.batch_size * len(self.batches)
        utilization = (utilized_utts / total_utts) * 100
        print(f"CategoryContrastiveSampler: Utilization = {utilized_utts}/{total_utts} or {utilization:.2f}% of total utterances")


    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
        )

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batches)
