"""
Trainer module for speaker recognition.
In speaker recognition (embedding extractor training/inference),
calculating validation loss in closed set is not informative since
generalization in unseen utterances from known speakers are good in most cases.
Thus, we measure open set equal error rate (EER) using unknown speakers by
overriding validate_one_epoch.
"""

from typing import Dict, Iterable

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

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


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
        print(f"Validating with {len(spk_to_enroll_utts)} enrolled speakers")
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
        print(f"Validating with {len(trial_pairs)} trials")


        model.eval()

        scores = []
        labels = []
        utt_embed_dict = {}
        bs = 0

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        utt_id_list = []
        speech_list = []
        task_token = None

        for utt_id, batch in iterator:
            bs = max(bs, len(utt_id))
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]
            assert isinstance(batch, dict), type(batch)
            utt_id_list.extend(utt_id)
            speech_list.extend(batch["speech"])
        
        stacked_speech = torch.stack(speech_list, dim=0)
        stacked_speech = to_device(stacked_speech, device)

        n_utt = len(utt_id_list)

        # extract speaker embeddings.
        for start in range(0, n_utt, bs):
            end = start + bs
            _utt_ids = utt_id_list[start:end]
            _speechs = stacked_speech[start:end]
            org_shape = (_speechs.size(0), _speechs.size(1))
            _speechs = _speechs.flatten(0, 1)

            if task_token is None:
                task_tokens = None
            else:
                task_tokens = to_device(
                    task_token.repeat(_speechs.size(0)), "cuda" if ngpu > 0 else "cpu"
                ).unsqueeze(1)

            spk_embds = model(
                speech=_speechs,
                spk_labels=None,
                extract_embd=True,
                task_tokens=task_tokens,
            )

            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for _utt_id, _spk_embd in zip(_utt_ids, spk_embds):
                utt_embed_dict[_utt_id] = _spk_embd

        del utt_id_list
        del speech_list
        torch.cuda.empty_cache()

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
                    current_spk_embd = spk_embed_dict[spk_id].unsqueeze(0)
        
                # Get test embedding and compute score
                test_embd = utt_embed_dict[test_utt].unsqueeze(0)
                score = torch.cdist(current_spk_embd, test_embd)
                score = -1.0 * score.mean()
                scores.append(score.view(1))
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
        for _s, _l in zip(scores, labels):
            if _l == 1:
                scores_trg.append(_s)
            elif _l == 0:
                scores_nontrg.append(_s)
            else:
                raise ValueError(f"{_l}, {type(_l)}")

        trg_mean = float(np.mean(scores_trg))
        trg_std = float(np.std(scores_trg))
        nontrg_mean = float(np.std(scores_nontrg))
        nontrg_std = float(np.std(scores_nontrg))

        # exception for collect_stats.
        if len(scores) == 1:
            reporter.register(stats=dict(eer=1.0, mindcf=1.0))
            return

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
        average: bool = False,
    ) -> None:
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        model.eval()
        utt_embed_dict = {}

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        # iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # fill dictionary with speech samples
        utt_id_list = []
        utt_id_whole_list = []
        speech_list = []
        task_token = None
        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        idx = 0
        for utt_id, batch in iterator:
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]

            assert isinstance(batch, dict), type(batch)
            for _utt_id, _speech, _speech2 in zip(
                utt_id, batch["speech"], batch["speech2"]
            ):
                _utt_id_1, _utt_id_2 = _utt_id.split("*")
                if _utt_id_1 not in utt_id_whole_list:
                    utt_id_whole_list.append(_utt_id_1)
                    if idx % world_size == rank:
                        utt_id_list.append(_utt_id_1)
                        speech_list.append(_speech)

                    if len(utt_id_list) == custom_bs:
                        speech_list = torch.stack(speech_list, dim=0)
                        org_shape = (speech_list.size(0), speech_list.size(1))
                        speech_list = speech_list.flatten(0, 1)
                        speech_list = to_device(
                            speech_list, "cuda" if ngpu > 0 else "cpu"
                        )
                        if task_token is None:
                            task_tokens = None
                        else:
                            task_tokens = to_device(
                                task_token.repeat(speech_list.size(0)),
                                "cuda" if ngpu > 0 else "cpu",
                            ).unsqueeze(1)
                        spk_embds = model(
                            speech=speech_list,
                            spk_labels=None,
                            extract_embd=True,
                            task_tokens=task_tokens,
                        )
                        # removed to be use magnitude in qmf
                        # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                        spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                        for uid, _spk_embd in zip(utt_id_list, spk_embds):
                            if average:
                                utt_embed_dict[uid] = (
                                    _spk_embd.mean(0).detach().cpu().numpy()
                                )
                            else:
                                utt_embed_dict[uid] = _spk_embd.detach().cpu().numpy()

                        utt_id_list = []
                        speech_list = []

                    idx += 1
                if _utt_id_2 not in utt_id_whole_list:
                    utt_id_whole_list.append(_utt_id_2)
                    if idx % world_size == rank:
                        utt_id_list.append(_utt_id_2)
                        speech_list.append(_speech2)

                    if len(utt_id_list) == custom_bs:
                        speech_list = torch.stack(speech_list, dim=0)
                        org_shape = (speech_list.size(0), speech_list.size(1))
                        speech_list = speech_list.flatten(0, 1)
                        speech_list = to_device(
                            speech_list, "cuda" if ngpu > 0 else "cpu"
                        )
                        if task_token is None:
                            task_tokens = None
                        else:
                            task_tokens = to_device(
                                task_token.repeat(speech_list.size(0)),
                                "cuda" if ngpu > 0 else "cpu",
                            ).unsqueeze(1)
                        spk_embds = model(
                            speech=speech_list,
                            spk_labels=None,
                            extract_embd=True,
                            task_tokens=task_tokens,
                        )
                        # removed to be use magnitude in qmf
                        # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                        spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                        for uid, _spk_embd in zip(utt_id_list, spk_embds):
                            if average:
                                utt_embed_dict[uid] = (
                                    _spk_embd.mean(0).detach().cpu().numpy()
                                )
                            else:
                                utt_embed_dict[uid] = _spk_embd.detach().cpu().numpy()

                        utt_id_list = []
                        speech_list = []

                    idx += 1

        if len(utt_id_list) != 0:
            speech_list = torch.stack(speech_list, dim=0)
            org_shape = (speech_list.size(0), speech_list.size(1))
            speech_list = speech_list.flatten(0, 1)
            speech_list = to_device(speech_list, "cuda" if ngpu > 0 else "cpu")
            if task_token is None:
                task_tokens = None
            else:
                task_tokens = to_device(
                    task_token.repeat(speech_list.size(0)),
                    "cuda" if ngpu > 0 else "cpu",
                ).unsqueeze(1)
            spk_embds = model(
                speech=speech_list,
                spk_labels=None,
                extract_embd=True,
                task_tokens=task_tokens,
            )
            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                if average:
                    utt_embed_dict[uid] = _spk_embd.mean(0).detach().cpu().numpy()
                else:
                    utt_embed_dict[uid] = _spk_embd.detach().cpu().numpy()

        np.savez(output_dir + f"/embeddings{rank}", **utt_embed_dict)
