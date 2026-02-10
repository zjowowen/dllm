import functools
import os
from dataclasses import dataclass, field

import accelerate
import transformers

import dllm
from dllm.pipelines.oneflow import OneFlowTrainer
from dllm.pipelines.oneflow.models import OneFlowConfig, OneFlowModel
from dllm.pipelines.oneflow.utils import (
    ONEFLOW_IMAGE_EOM,
    ONEFLOW_IMAGE_SOM,
    ONEFLOW_IMAGE_TOKEN,
    OneFlowCollator,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = "gpt2"

    # OneFlow model size (toy defaults)
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8

    # image latent dim is unused for text-only pretraining, but kept for config completeness
    dim_latent: int = 4

    # Tie to_q_logits.weight to text_embed.weight (EditFlow-inspired weight sharing).
    tie_q_logits_to_embedding: bool = False


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "Trelis/tiny-shakespeare"
    text_field: str = "Text"
    max_length: int = 256
    streaming: bool = False
    # If True, treat `dataset_args` as a local path produced by 🤗 datasets `save_to_disk`,
    # and assume it is ALREADY preprocessed into PT format (contains `input_ids`).
    # This is useful for offline clusters to avoid downloading + tokenization on cluster.
    load_preprocessed_data: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(default=True)


@dataclass
class TrainingArguments(OneFlowTrainer.OneFlowConfig):
    output_dir: str = None  # overwrite this
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    # Text-only PT default: we usually don't have a test split (and OneFlow is label-free),
    # so disable eval by default to avoid HF Trainer init errors.
    eval_strategy: str = "no"
    do_eval: bool = False
    scheduler_cls: str = field(
        default="LinearKappaScheduler",
        metadata={
            "help": (
                "The scheduler class controlling κ(t). "
                "Available options: see `dllm/core/schedulers/kappa.py`"
            )
        },
    )
    # ---- perf profiling helpers (off by default) ---------------------------------
    profile_timing: bool = False
    profile_timing_sync: bool = False
    profile_log_optimizer_time: bool = False

    # Torch profiler trace (rank0 only). Writes to `${output_dir}/profile/`.
    profile_trace: bool = False
    profile_trace_warmup: int = 2
    profile_trace_steps: int = 10


def build_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizer:
    # Helpful guard: if user passes an absolute/relative filesystem path, ensure it exists.
    # (Repo IDs like "org/name" are allowed and may contain "/".)
    p = os.path.expanduser(str(tokenizer_name_or_path))
    is_path_like = os.path.isabs(p) or p.startswith(".") or p.startswith("~") or os.path.exists(p)
    if is_path_like and (not os.path.exists(p)):
        raise ValueError(
            f"tokenizer_name_or_path looks like a local path but does not exist: {p}\n"
            "If you intended to use an offline bundle, make sure you set the shell variable first, e.g.:\n"
            "  BUNDLE=/abs/path/to/bundle\n"
            "  ... --tokenizer_name_or_path \"$BUNDLE/tokenizer\" --dataset_args \"$BUNDLE/dataset\""
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, padding_side="right"
    )

    # ensure pad/eos/bos exist (mirror dllm.utils.get_tokenizer behavior)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    # add OneFlow special tokens (safe even for text-only)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ONEFLOW_IMAGE_TOKEN, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_EOM]}
    )
    return tokenizer


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # necessary when batch does not contain "labels" field
    training_args.label_names = []
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    # necessary for streaming dataset
    training_args.accelerator_config.dispatch_batches = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    # ----- Dataset (PT-style) ------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        # Helpful guard for offline/preprocessed path.
        if bool(data_args.load_preprocessed_data):
            dp = os.path.expanduser(str(data_args.dataset_args))
            if (os.path.isabs(dp) or dp.startswith(".") or dp.startswith("~") or os.path.exists(dp)) and (
                not os.path.exists(dp)
            ):
                raise ValueError(
                    f"load_preprocessed_data=True but dataset_args path does not exist: {dp}\n"
                    "Expected a directory produced by `datasets.save_to_disk(...)`.\n"
                    "If you intended to use an offline bundle, make sure you set the shell variable first, e.g.:\n"
                    "  BUNDLE=/abs/path/to/bundle\n"
                    "  ... --dataset_args \"$BUNDLE/dataset\""
                )
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )

        if not data_args.load_preprocessed_data:
            map_fn = functools.partial(
                dllm.utils.tokenize_and_group,
                tokenizer=tokenizer,
                text_field=data_args.text_field,
                seq_length=data_args.max_length,
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail,
                add_special_tokens=False,
            )

            dataset = dataset.map(
                map_fn,
                batched=True,
                remove_columns=dataset["train"].column_names,
                **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
                **({} if data_args.streaming else {"desc": "Mapping dataset to PT format"}),
            )
        else:
            # Offline/preprocessed path: dataset should already contain `input_ids`.
            if "input_ids" not in dataset["train"].column_names:
                raise ValueError(
                    "load_preprocessed_data=True but dataset does not contain `input_ids`."
                )

        # Ensure each sample starts with BOS (required for insertion slot semantics)
        bos_id = int(tokenizer.bos_token_id)

        def add_bos(row):
            ids = row["input_ids"]
            if not ids:
                return row
            if ids[0] != bos_id:
                row["input_ids"] = [bos_id] + ids
            return row

        # NOTE: For streaming datasets (IterableDataset/IterableDatasetDict),
        # `datasets` does NOT support multiprocessing `num_proc` in `.map(...)`.
        if data_args.streaming:
            dataset = dataset.map(add_bos)
        else:
            dataset = dataset.map(
                add_bos,
                num_proc=data_args.num_proc,
                desc="Prepending BOS",
            )

        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Model ------------------------------------------------------------------
    cfg = OneFlowConfig(
        vocab_size=len(tokenizer),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        unk_token_id=int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
        dim=model_args.dim,
        depth=model_args.depth,
        dim_head=model_args.dim_head,
        heads=model_args.heads,
        dim_latent=model_args.dim_latent,
        tie_q_logits_to_embedding=bool(getattr(model_args, "tie_q_logits_to_embedding", False)),
    )
    model = OneFlowModel(cfg)

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow text-only training...")

    # If user enables eval_strategy but no eval split exists, disable evaluation to avoid
    # HF Trainer's hard error on init.
    eval_ds = dataset.get("test", None)
    if eval_ds is None:
        try:
            es = str(getattr(training_args, "eval_strategy", "no") or "no")
        except Exception:
            es = "no"
        if es.lower() != "no":
            logger.warning(
                f"No eval_dataset found in loaded dataset (splits={list(dataset.keys())}); "
                "forcing eval_strategy='no' for safety."
            )
            training_args.eval_strategy = "no"
            training_args.do_eval = False

    trainer = OneFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=OneFlowCollator(tokenizer=tokenizer),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
    )

    # Optional torch profiler trace (rank0 only).
    if bool(getattr(training_args, "profile_trace", False)) and accelerate.PartialState().is_main_process:
        try:
            from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

            acts = [ProfilerActivity.CPU]
            if hasattr(ProfilerActivity, "NPU"):
                acts.append(getattr(ProfilerActivity, "NPU"))
            elif hasattr(ProfilerActivity, "CUDA"):
                acts.append(ProfilerActivity.CUDA)

            prof_dir = os.path.join(training_args.output_dir, "profile")
            os.makedirs(prof_dir, exist_ok=True)

            wait = 0
            warmup = max(0, int(getattr(training_args, "profile_trace_warmup", 2)))
            active = max(1, int(getattr(training_args, "profile_trace_steps", 10)))
            prof = profile(
                activities=acts,
                schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=tensorboard_trace_handler(prof_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )

            from transformers import TrainerCallback

            class _ProfilerCallback(TrainerCallback):
                def __init__(self, p):
                    self.p = p

                def on_train_begin(self, args, state, control, **kwargs):
                    self.p.__enter__()

                def on_step_end(self, args, state, control, **kwargs):
                    self.p.step()

                def on_train_end(self, args, state, control, **kwargs):
                    try:
                        self.p.__exit__(None, None, None)
                    except Exception:
                        pass

            trainer.add_callback(_ProfilerCallback(prof))
            logger.info(f"[profile_trace] enabled (rank0). traces -> {prof_dir}")
        except Exception as e:
            logger.warning(f"[profile_trace] failed to enable torch profiler: {e}")

    trainer.train()

    # Save
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()


