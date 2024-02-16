import transformers
from transformers import AutoModelForCausalLM
import copy
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union
import torch
from tqdm import tqdm
import os
import pandas as pd
import json
import re
from thefuzz import process
from types import SimpleNamespace
import torch
import asyncio

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages):
    if all(m['role'] != "user" for m in messages):
        raise Exception(f"Invalid request: Expecting at least one user message.")

    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0]['role'] == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content = m['role'], m['content']
        if content:
            content = content.lstrip("\n").rstrip()
        if role == "assistant":
            if len(messages) == 0:
                raise Exception(f"Invalid request: Expecting role user before role assistant.")
            if messages[-1]['role'] == "user":
                messages.append(
                    {'role': "assistant", 'content': content.lstrip("\n").rstrip()}
                )
            else:
                messages[-1]['content'] += content
        elif role == "user":
            messages.append(
                {'role': "user", 'content': content.lstrip("\n").rstrip()}
            )
        else:
            raise Exception(f"Invalid request: Incorrect role {role}.")

    query = _TEXT_COMPLETION_CMD
    if messages[-1]['role'] == "user":
        query = messages[-1]['content']
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise Exception("Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i]['role'] == "user" and messages[i + 1]['role'] == "assistant":
            usr_msg = messages[i]['content'].lstrip("\n").rstrip()
            bot_msg = messages[i + 1]['content'].lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            history.append([usr_msg, bot_msg])
        else:
            raise Exception("Invalid request: Expecting exactly one user "
                            "(or function) role before every assistant role.", )
    if system:
        query = f"{system}\n\nQuestion: {query}"
    return query, history


def format_example(line, prompts):
    example = prompts + f"{line['question']}\n"
    for choice in choices:
        example += f'({choice}) {line[f"{choice}"]} '
    example = example.rstrip() + '\n'
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response, {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


class MultiAgentSynergy:
    DecodingStatus = SimpleNamespace(GENERAL=1, EXPERT=2, FINISH=3)

    def __init__(self, general_llm, experts, tokenizer, agent_tokens):
        self.general_llm = general_llm
        self.experts = experts
        self.tokenizer = tokenizer
        self.agent_tokens = agent_tokens

    def chat_completion(self,
                        messages,
                        temperature=0.0,
                        top_p=None,
                        stop=None):

        gen_kwargs = {}
        if temperature is not None:
            if temperature < 0.01:
                gen_kwargs['top_k'] = 1  # greedy decoding
            else:
                # Not recommended. Please tune top_p instead.
                gen_kwargs['temperature'] = temperature
        if top_p is not None:
            gen_kwargs['top_p'] = top_p

        messages = copy.deepcopy(messages)
        agent_token_tuple = tuple(self.agent_tokens.keys())

        stop_words = add_extra_stop_words(stop)
        stop_words_ids = [self.tokenizer.encode(s) for s in stop_words] if stop_words else None
        agent_stop_words_ids = [self.tokenizer.encode(s) for s in agent_token_tuple]
        stop_words_ids_all = stop_words_ids + agent_stop_words_ids if stop_words_ids else agent_stop_words_ids

        decoding_status = MultiAgentSynergy.DecodingStatus.GENERAL
        decoding_llm = None

        expert_synergy = []

        while decoding_status != MultiAgentSynergy.DecodingStatus.FINISH:
            if decoding_status == MultiAgentSynergy.DecodingStatus.GENERAL:
                decoding_llm = self.general_llm

            query, history = parse_messages(messages)

            if query is _TEXT_COMPLETION_CMD:
                response = self.text_complete_last_message(
                    decoding_llm,
                    history,
                    stop_words_ids=stop_words_ids_all,
                    gen_kwargs=gen_kwargs)
            else:
                response, _ = decoding_llm.chat(
                    self.tokenizer,
                    query,
                    history=history,
                    stop_words_ids=stop_words_ids_all,
                    **gen_kwargs
                )
            print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")

            if response.endswith(agent_token_tuple):
                agent_token = response[-len(agent_token_tuple[0]):]
                response = response[:-len(agent_token_tuple[0])]
                decoding_llm = self.experts[self.agent_tokens[agent_token]]
                expert_synergy.append(self.agent_tokens[agent_token])

                decoding_status = MultiAgentSynergy.DecodingStatus.EXPERT
            else:
                decoding_status = MultiAgentSynergy.DecodingStatus.FINISH

            if messages[-1]['role'] != 'assistant':
                messages.append({'role': 'assistant', 'content': response})
            else:
                messages[-1]['content'] = response

        response = messages[-1]['content']
        response = trim_stop_words(response, stop_words)
        return response, expert_synergy


    def text_complete_last_message(self, llm, history, stop_words_ids, gen_kwargs):
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
        for i, (query, response) in enumerate(history):
            query = query.lstrip("\n").rstrip()
            response = response.lstrip("\n").rstrip()
            prompt += f"\n{im_start}user\n{query}{im_end}"
            prompt += f"\n{im_start}assistant\n{response}{im_end}"
        prompt = prompt[: -len(im_end)]

        _stop_words_ids = [self.tokenizer.encode(im_end)]
        if stop_words_ids:
            for s in stop_words_ids:
                _stop_words_ids.append(s)
        stop_words_ids = _stop_words_ids

        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(llm.device)
        output = llm.generate(input_ids, stop_words_ids=stop_words_ids, **gen_kwargs).tolist()[0]
        output = self.tokenizer.decode(output, errors="ignore")
        assert output.startswith(prompt)
        output = output[len(prompt):]
        output = trim_stop_words(output, ["<|endoftext|>", im_end])
        print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
        return output


@torch.no_grad()
def eval_subject(ma,
                 test_df,
                 prompt):
    result = []
    conversations = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, prompt)
        messages = [{"role": "user", "content": question}]
        response, expert_list = ma.chat_completion(
            messages=messages,
            temperature=0.0,
        )
        messages.append({'role': 'assistant', 'content': response})
        conv = {"id": f"identity_{_}", "conversations": [], "expert": expert_list}
        for m in messages:
            conv["conversations"].append({"from": m["role"], "value": m["content"]})

        conversations.append(conv)
        pred = extract_answer(response, row)

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            result.append(correct)

    return conversations, result


SUBJECTS = ("astronomy", "electrical_engineering", "security_studies",
            "prehistory", "international_law", "human_sexuality")

choices = ["A", "B", "C", "D"]

AGENT_TOKEN_DICT = {
    "<|extra_0|>": "astronomy",
    "<|extra_1|>": "electrical_engineering",
    "<|extra_2|>": "security_studies",
    "<|extra_3|>": "prehistory",
    "<|extra_4|>": "international_law",
    "<|extra_5|>": "human_sexuality",
}


async def load_model(model_path, idx):
    loop = asyncio.get_event_loop()
    # 使用线程池执行阻塞性的加载操作
    model = await loop.run_in_executor(None,
                                       lambda: AutoModelForCausalLM.from_pretrained(f"{model_path}",
                                                                                    device_map=f"cuda:{idx}",
                                                                                    trust_remote_code=True,
                                                                                    bf16=True,
                                                                                    use_safetensors=True))
    return model


async def main():
    expert_dict = {}
    split = "test"
    acc_dict = {}
    model_path_list = []

    for subject in SUBJECTS:
        model_path_list.append(f"expert_models/mmlu/Qwen_{subject}")

    model_path_list.append(f"qwen_7b_extra")
    tasks = [asyncio.create_task(load_model(m, _)) for _, m in enumerate(model_path_list)]
    models = await asyncio.gather(*tasks)

    for _, subject in enumerate(SUBJECTS):
        expert_dict[subject] = models[_].eval()

    general_llm = models[-1]
    extra_lm_head = torch.load('qwen_7b_extra/extra/lm_head.pth')
    general_llm.load_state_dict(extra_lm_head, strict=False)
    general_llm = general_llm.eval()

    ma = MultiAgentSynergy(general_llm=general_llm,
                           experts=expert_dict,
                           tokenizer=transformers.AutoTokenizer.from_pretrained("Qwen-7B-Chat",
                                                                                use_fast=False,
                                                                                trust_remote_code=True),
                           agent_tokens=AGENT_TOKEN_DICT)

    for subject in tqdm(SUBJECTS):
        test_file_path = os.path.join(
            "dataset/mmlu", f"{split}", f"{subject}_test.csv",
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)

        zero_shots_prompts = ("The following is a multiple-choice question. Please choose the most "
                              "suitable one among A, B, C and D as the answer to this question.\n\n")

        conversations, result = eval_subject(
            ma,
            test_df,
            zero_shots_prompts,
        )

        if not os.path.exists(f'evaluation/mmlu/multi'):
            os.makedirs(f'evaluation/mmlu/multi')

        with open(f'evaluation/mmlu/multi/{subject}_{split}.json', 'w') as json_file:
            json.dump(conversations, json_file, indent=4)
        acc = sum(result) / len(result)
        acc_dict[subject] = acc

        with open(f"evaluation/mmlu/multi/acc_{split}.json", 'w') as json_file:
            json.dump(acc_dict, json_file, indent=4)


if __name__ == '__main__':
    asyncio.run(main())
