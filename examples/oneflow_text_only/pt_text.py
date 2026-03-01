import functools
import os
from dataclasses import dataclass, field

import accelerate
import transformers

import dllm
from dllm.pipelines.oneflow.utils import (
    ONEFLOW_IMAGE_EOM,
    ONEFLOW_IMAGE_SOM,
    ONEFLOW_IMAGE_TOKEN,
    OneFlowCollator,
)
from dllm.pipelines.oneflow_text_only.models import OneFlowTextOnlyConfig, OneFlowTextOnlyModel
from dllm.pipelines.oneflow_text_only.trainer import OneFlowTextOnlyTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = "gpt2"
    init_model_dir: str | None = None

    # Reference text model style size params
    hidden_size: int = 768
    cond_dim: int = 128
    n_blocks: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    mlp_ratio: int = 4
    rotary_base: int = 10000

    # Keep output contract compatible with OneFlow trainer/sampler.
    dim_latent: int = 4
    tie_q_logits_to_embedding: bool = False


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "Trelis/tiny-shakespeare"
    text_field: str = "Text"
    max_length: int = 256
    streaming: bool = False
    load_preprocessed_data: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(default=True)


@dataclass
class TrainingArguments(OneFlowTextOnlyTrainer.OneFlowConfig):
    output_dir: str = None
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_strategy: str = "no"
    do_eval: bool = False
    scheduler_cls: str = field(
        default="LinearKappaScheduler",
        metadata={
            "help": (
                "The scheduler class controlling kappa(t). "
                "Available options: see `dllm/core/schedulers/kappa.py`"
            )
        },
    )


def build_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizer:
    p = os.path.expanduser(str(tokenizer_name_or_path))
    is_path_like = os.path.isabs(p) or p.startswith(".") or p.startswith("~") or os.path.exists(p)
    if is_path_like and (not os.path.exists(p)):
        raise ValueError(
            f"tokenizer_name_or_path looks like a local path but does not exist: {p}\n"
            "If you intended to use an offline bundle, make sure you set the shell variable first, e.g.:\n"
            "  BUNDLE=/abs/path/to/bundle\n"
            "  ... --tokenizer_name_or_path \"$BUNDLE/tokenizer\" --dataset_args \"$BUNDLE/dataset\""
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    # Keep oneflow special tokens for compatibility with existing sampler/train code paths.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ONEFLOW_IMAGE_TOKEN, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_EOM]}
    )
    return tokenizer


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.label_names = []
    training_args.remove_unused_columns = False
    training_args.accelerator_config.dispatch_batches = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    with accelerate.PartialState().local_main_process_first():
        if bool(data_args.load_preprocessed_data):
            dp = os.path.expanduser(str(data_args.dataset_args))
            if (os.path.isabs(dp) or dp.startswith(".") or dp.startswith("~") or os.path.exists(dp)) and (
                not os.path.exists(dp)
            ):
                raise ValueError(
                    f"load_preprocessed_data=True but dataset_args path does not exist: {dp}\n"
                    "Expected a directory produced by `datasets.save_to_disk(...)`."
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
            if "input_ids" not in dataset["train"].column_names:
                raise ValueError("load_preprocessed_data=True but dataset does not contain `input_ids`.")

        bos_id = int(tokenizer.bos_token_id)

        def add_bos(row):
            ids = row["input_ids"]
            if ids and ids[0] != bos_id:
                row["input_ids"] = [bos_id] + ids
            return row

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

    init_model_dir = getattr(model_args, "init_model_dir", None)
    if init_model_dir:
        init_model_dir = os.path.expanduser(str(init_model_dir))
        if not os.path.exists(init_model_dir):
            raise ValueError(f"init_model_dir does not exist: {init_model_dir}")
        logger.info(f"Loading init model from checkpoint: {init_model_dir}")
        model = OneFlowTextOnlyModel.from_pretrained(init_model_dir, map_location="cpu")
    else:
        cfg = OneFlowTextOnlyConfig(
            vocab_size=len(tokenizer),
            bos_token_id=int(tokenizer.bos_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            unk_token_id=int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
            hidden_size=model_args.hidden_size,
            cond_dim=model_args.cond_dim,
            n_blocks=model_args.n_blocks,
            n_heads=model_args.n_heads,
            dropout=model_args.dropout,
            mlp_ratio=model_args.mlp_ratio,
            rotary_base=model_args.rotary_base,
            dim_latent=model_args.dim_latent,
            tie_q_logits_to_embedding=bool(model_args.tie_q_logits_to_embedding),
        )
        model = OneFlowTextOnlyModel(cfg)

    if int(getattr(model.config, "vocab_size", 0)) != int(len(tokenizer)):
        logger.warning(
            "Tokenizer/model vocab mismatch detected "
            f"(model={getattr(model.config, 'vocab_size', None)}, tokenizer={len(tokenizer)}). "
            "Resizing token embeddings to tokenizer size."
        )
        model.resize_token_embeddings(len(tokenizer))

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow text-only (reference-style backbone) training...")

    eval_ds = dataset.get("test", None)
    try:
        es = str(getattr(training_args, "eval_strategy", "no") or "no")
    except Exception:
        es = "no"
    if eval_ds is None and es.lower() != "no":
        logger.warning(
            f"No eval_dataset found in loaded dataset (splits={list(dataset.keys())}); "
            "forcing eval_strategy='no' for safety."
        )
        training_args.eval_strategy = "no"
        training_args.do_eval = False

    trainer = OneFlowTextOnlyTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=OneFlowCollator(tokenizer=tokenizer),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
    )

    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()

