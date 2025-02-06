"""
Trainer module for speaker recognition.
In speaker recognition (embedding extractor training/inference),
calculating validation loss in closed set is not informative since
generalization in unseen utterances from known speakers are good in most cases.
Thus, we measure open set equal error rate (EER) using unknown speakers by
overriding validate_one_epoch.
"""

from typing import Dict, Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import to_device
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore
from espnet2.utils.a_dcf import calculate_a_dcf
from tqdm import tqdm

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


@dataclass
class SASVCostModel:
    Pspf: float = 0.05
    Pnontrg: float = 0.05
    Ptrg: float = 0.9
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 20


class SpkTrainer(Trainer):
    """Trainer designed for speaker recognition.

    Training will be done as closed set classification.
    Validation will be open set EER calculation.
    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    @typechecked
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:

        ngpu = options.ngpu
        distributed = distributed_option.distributed
        device = "cuda" if ngpu > 0 else "cpu"

        trial_info_dir = options.output_dir + "/trial_info"

        # dicts for enrollment and trial info
        spk_to_enroll_utts = {}  # {spkID: [utt_id1, utt_id2, ...]}
        trial_pairs = []  # [(spkID, test_utt_id, label), ...]
        spk_embed_dict = {}  # {spk_id: averaged_embedding}

        # Read spk2enroll file
        spk2enroll_path = f"{trial_info_dir}/spk2enroll"
        with open(spk2enroll_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                spk_id, enroll_utts = parts
                spk_to_enroll_utts[spk_id] = enroll_utts.split(',')

        # Read trial_label file
        trial_label_path = f"{trial_info_dir}/trial_label"
        with open(trial_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                pair_str, label = parts
                spk_id, test_utt = pair_str.split('*')
                label = int(label)
                trial_pairs.append((spk_id, test_utt, label))
    
        print(f"Validating with {len(spk_to_enroll_utts)} enrolled speakers and {len(trial_pairs)} trials")

        # task (spk or sasv)
        task = None
        unique_labels = set([label for _, _, label in trial_pairs])
        if len(unique_labels) == 2:
            task = "spk"
        elif len(unique_labels) == 3:
            task = "sasv"
        else:
            raise ValueError(f"Unknown task with {unique_labels} unique labels")

        model.eval()

        scores = []
        labels = []
        utt_embed_dict = {}
        bs = 0

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        task_token = None

        for utt_id, batch in tqdm(iterator):
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]
            assert isinstance(batch, dict), type(batch)
            speech_batch = to_device(batch["speech"], device)
            org_shape = (speech_batch.size(0), speech_batch.size(1))
            speech_batch = speech_batch.flatten(0, 1)

            # Prepare task tokens if needed
            if task_token is not None:
                task_tokens = to_device(
                    task_token.repeat(speech_batch.size(0)), 
                    "cuda" if ngpu > 0 else "cpu"
                ).unsqueeze(1)
            else:
                task_tokens = None

            spk_embds = model(
                speech=speech_batch,
                spk_labels=None,
                extract_embd=True,
                task_tokens=task_tokens,
            )

            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for _utt_id, _spk_embd in zip(utt_id, spk_embds):
                utt_embed_dict[_utt_id] = _spk_embd

        torch.cuda.empty_cache()

        # different gpus have different utterances in the dictionary
        # so we need to gather all embeddings to all gpus
        if distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Prepare list to gather all dictionaries
            all_utt_embed_dicts = [None] * world_size

            # Gather dictionaries from all ranks
            torch.distributed.all_gather_object(all_utt_embed_dicts, utt_embed_dict)

            # Merge all gathered dictionaries
            utt_embed_dict = {}
            for d in all_utt_embed_dicts:
                utt_embed_dict.update(d)

        print(f"{len(utt_embed_dict)} embeddings extracted")

        # make speaker embeddings by averaging utterance embeddings
        for spk_id, enroll_utts in spk_to_enroll_utts.items():
            # Stack all enrollment embeddings for this speaker
            enroll_embds = torch.stack([utt_embed_dict[utt] for utt in enroll_utts])
            # Average them to get speaker embedding
            spk_embed_dict[spk_id] = enroll_embds.mean(dim=0)

        # calculate similarity scores
        # first, sort trials
        sorted_trials = sorted(trial_pairs, key=lambda x: (x[0], x[1]))  # Sort by speaker_id, then test_utt
        current_spk = None
        current_spk_embd = None

        try:
            for spk_id, test_utt, label in sorted_trials:
                if distributed:
                    torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                    if iterator_stop > 0:
                        break
        
                # Only get speaker embedding if speaker changes
                if spk_id != current_spk:
                    current_spk = spk_id
                    current_spk_embd = spk_embed_dict[spk_id].unsqueeze(0).mean(dim=1)
        
                # Get test embedding and compute score
                test_embd = utt_embed_dict[test_utt].unsqueeze(0).mean(dim=1)
                current_spk_embd = current_spk_embd.to(device)
                test_embd = test_embd.to(device)
                score = F.cosine_similarity(current_spk_embd, test_embd, dim=1)
                scores.append(score)
                labels.append(torch.tensor([label], device=device))
            
        except RuntimeError as e:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
            raise e

        torch.cuda.empty_cache()

        scores = torch.cat(scores).type(torch.float32)
        labels = torch.cat(labels).type(torch.int32).flatten()

        if distributed:
            # get the number of trials assigned on each GPU
            length = to_device(
                torch.tensor([labels.size(0)], dtype=torch.int32), "cuda"
            )
            lengths_all = [
                to_device(torch.zeros(1, dtype=torch.int32), "cuda")
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(lengths_all, length)

            scores_all = [
                to_device(torch.zeros(i, dtype=torch.float32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(scores_all, scores)
            scores = torch.cat(scores_all)

            labels_all = [
                to_device(torch.zeros(i, dtype=torch.int32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(labels_all, labels)
            labels = torch.cat(labels_all)
            # rank = torch.distributed.get_rank()
            torch.distributed.barrier()

        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # calculate statistics in target and nontarget classes.
        n_trials = len(scores)
        scores_trg = []
        scores_nontrg = []
        scores_spf = []

        if task == "spk":
            for _s, _l in zip(scores, labels):
                if _l == 1:
                    scores_trg.append(_s)
                elif _l == 0:
                    scores_nontrg.append(_s)
                else:
                    raise ValueError(f"{_l}, {type(_l)}")
        else:
            for _s, _l in zip(scores, labels):
                if _l == 0:
                    scores_trg.append(_s)
                elif _l == 1:
                    scores_nontrg.append(_s)
                elif _l == 2:
                    scores_spf.append(_s)
                else:
                    raise ValueError(f"{_l}, {type(_l)}")

        trg_mean = float(np.mean(scores_trg))
        trg_std = float(np.std(scores_trg))
        nontrg_mean = float(np.std(scores_nontrg))
        nontrg_std = float(np.std(scores_nontrg))
        spf_mean = float(np.mean(scores_spf))
        spf_std = float(np.std(scores_spf))

        # exception for collect_stats.
        if len(scores) == 1:
            if task == "spk":
                reporter.register(stats=dict(eer=1.0, mindcf=1.0))
                return
            elif task == "sasv":
                reporter.register(stats=dict(min_a_dcf=1.0))
                return
        
        if task == "spk":
            # predictions, ground truth, and the false acceptance rates to calculate
            results = tuneThresholdfromScore(scores, labels, [1, 0.1])
            eer = results[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

            # p_target, c_miss, and c_falsealarm in NIST minDCF calculation
            p_trg, c_miss, c_fa = 0.05, 1, 1
            mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)

            reporter.register(
                stats=dict(
                    eer=eer,
                    mindcf=mindcf,
                    n_trials=n_trials,
                    trg_mean=trg_mean,
                    trg_std=trg_std,
                    nontrg_mean=nontrg_mean,
                    nontrg_std=nontrg_std,
                )
            )
        
        elif task == "sasv":
            # write the scores to a file
            with open(f"{options.output_dir}/scores.txt", "w") as f:
                for (spk_id, test_utt, label), score in zip(trial_pairs, scores):
                    f.write(f"{spk_id} {test_utt} {score} {label}\n")

            # calculate a-DCF
            results_dict = calculate_a_dcf(sasv_score_txt=f"{options.output_dir}/scores.txt",
                                           cost_model=SASVCostModel())
            reporter.register(
            stats=dict(
                min_a_dcf=results_dict["min_a_dcf"],
                min_a_dcf_thresh=results_dict["min_a_dcf_thresh"],
                n_trials=n_trials,
                trg_mean=trg_mean,
                trg_std=trg_std,
                nontrg_mean=nontrg_mean,
                nontrg_std=nontrg_std,
                spf_mean=spf_mean,
                spf_std=spf_std,
                )
            )

        # added to reduce GRAM usage. May have minor speed boost when
        # this line is commented in case GRAM is not fully used.
        torch.cuda.empty_cache()

    @classmethod
    @torch.no_grad()
    @typechecked
    def extract_embed(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: str,
        custom_bs: int,
    ) -> None:
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        trial_info_dir = options.output_dir + "/trial_info"
        device = "cuda" if ngpu > 0 else "cpu"

        model.eval()
        utt_embed_dict = {}

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        # iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # dicts for enrollment and trial info
        spk_to_enroll_utts = {}  # {spkID: [utt_id1, utt_id2, ...]}
        trial_pairs = []  # [(spkID, test_utt_id, label), ...]
        spk_embed_dict = {}  # {spk_id: averaged_embedding}

        # Read spk2enroll file
        spk2enroll_path = f"{trial_info_dir}/spk2enroll"
        with open(spk2enroll_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                spk_id, enroll_utts = parts
                spk_to_enroll_utts[spk_id] = enroll_utts.split(',')

        # Read trial_label file
        trial_label_path = f"{trial_info_dir}/trial_label"
        with open(trial_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                pair_str, label = parts
                spk_id, test_utt = pair_str.split('*')
                label = int(label)
                trial_pairs.append((spk_id, test_utt, label))

        print(f"Computing inference embeddings for {len(spk_to_enroll_utts)} enrolled speakers and {len(trial_pairs)} trials")

        task_token = None
        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        idx = 0
        for utt_id, batch in tqdm(iterator):
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]
            assert isinstance(batch, dict), type(batch)

            speech_batch = to_device(batch["speech"], device)
            org_shape = (speech_batch.size(0), speech_batch.size(1))
            speech_batch = speech_batch.flatten(0, 1)

            # Prepare task tokens if needed
            if task_token is not None:
                task_tokens = to_device(
                    task_token.repeat(speech_batch.size(0)), 
                    "cuda" if ngpu > 0 else "cpu"
                ).unsqueeze(1)
            else:
                task_tokens = None

            spk_embds = model(
                speech=speech_batch,
                spk_labels=None,
                extract_embd=True,
                task_tokens=task_tokens,
            )

            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for _utt_id, _spk_embd in zip(utt_id, spk_embds):
                utt_embed_dict[_utt_id] = _spk_embd

        torch.cuda.empty_cache()

        # different gpus have different utterances in the dictionary
        # so we need to gather all embeddings to all gpus
        if distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Prepare list to gather all dictionaries
            all_utt_embed_dicts = [None] * world_size

            # Gather dictionaries from all ranks
            torch.distributed.all_gather_object(all_utt_embed_dicts, utt_embed_dict)

            # Merge all gathered dictionaries
            utt_embed_dict = {}
            for d in all_utt_embed_dicts:
                utt_embed_dict.update(d)

        print(f"{len(utt_embed_dict)} embeddings extracted")

        # make speaker embeddings by averaging utterance embeddings
        for spk_id, enroll_utts in spk_to_enroll_utts.items():
            # Stack all enrollment embeddings for this speaker
            enroll_embds = torch.stack([utt_embed_dict[utt] for utt in enroll_utts])
            # Average them to get speaker embedding
            utt_embed_dict[spk_id] = enroll_embds.mean(dim=0)

        utt_embed_dict_cpu = {k: v.cpu().numpy() for k, v in utt_embed_dict.items()}
        np.savez(output_dir + f"/embeddings{rank}", **utt_embed_dict_cpu)

        torch.cuda.empty_cache()