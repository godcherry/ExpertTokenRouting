import json
import transformers
import torch
import argparse
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from typing import Dict
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from ETR.qwen_7b_extra.modeling_qwen import QWenLMHeadModel
from ETR.qwen_7b_extra.tokenization_qwen import QWenTokenizer

from transformers import (
    default_data_collator,
    get_scheduler,
    SchedulerType
)
from torch.utils.data import DataLoader
import math
from tqdm import tqdm

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * (len(system)-3) + [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID]
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID]
            elif role == '<|im_start|>assistant':
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID]
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"].long()
        self.labels = data_dict["labels"].long()
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def main():
    args = parse_args()

    # accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
    #                           gradient_accumulation_steps=args.gradient_accumulation_steps)

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs],
                              gradient_accumulation_steps=args.gradient_accumulation_steps)
    # accelerator.init_trackers(project_name=f"{args.project}",
    #                           init_kwargs={"wandb": {"name": f"exp", "config": args}}, )

    tokenizer = QWenTokenizer.from_pretrained(
        "Qwen-7B-Chat",
        model_max_length=args.model_max_len,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    train_json = json.load(open(args.train_file, "r"))
    train_dataset = SupervisedDataset(train_json, tokenizer=tokenizer, max_len=args.model_max_len)


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,
        drop_last=True
    )


    model = QWenLMHeadModel.from_pretrained("qwen_7b_extra",
                                            device_map="cuda:0")
    mean_vector = torch.mean(model.lm_head.weight.data, dim=0, keepdim=True)
    model.extra_lm_head.weight.data = mean_vector.repeat(model.extra_vocab_size, 1)

    trainable_params = []

    for n, param in model.named_parameters():
        param.requires_grad = False
        if 'extra_lm_head' in n:
            param.requires_grad = True
            trainable_params.append(param)

    optimizer_grouped_parameters = [{"params": trainable_params,
                                     "weight_decay": args.weight_decay}]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(max_train_steps * args.gradient_accumulation_steps))
    accum_loss = 0.

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                accum_loss = accum_loss + loss.item()
                progress_bar.update(1)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                accelerator.print(f"Accum Loss: {accum_loss / args.gradient_accumulation_steps}")
                accum_loss = 0.


    accelerator.wait_for_everyone()
    parameters_to_save = {name: param for name, param in accelerator.unwrap_model(model).cpu().state_dict().items() if 'extra_lm_head' in name}
    torch.save(parameters_to_save, 'qwen_7b_extra/extra/lm_head_150.pth')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="Testproject")
    parser.add_argument("--model_name", type=str, default='qwen_7b_extra')
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--model_max_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_file", type=str, default='synthetic_mmlu/training_data_clean/train.json')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        choices=["linear", "cosine", 'cosine_with_restarts', "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=5)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    transformers.logging.set_verbosity_error()
    main()
