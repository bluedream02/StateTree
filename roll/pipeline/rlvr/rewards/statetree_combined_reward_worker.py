"""
StateTree combined reward: r = max(r_EM, r_LLM).

Paper Section 3.2 — Exact Match OR binary LLM-as-a-Judge (CORRECT/WRONG).
Compatible with ROLL RLVR pipeline reward workers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.datasets.chat_template import get_chat_template
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider
from roll.platforms import current_platform
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import prompt_maps


def extract_boxed(response: str) -> str:
    idx = response.find(r"\boxed{")
    if idx < 0:
        # Fallback: last non-empty line after </think>
        cut = response
        if "</think>" in response:
            cut = response.split("</think>")[-1]
        lines = [ln.strip() for ln in cut.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(response) and depth > 0:
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return ""
    return response[start : i - 1].strip()


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def exact_match(pred: str, gold: str) -> float:
    if not gold:
        return 1.0
    # Support multiple golds joined by ||
    gts = [g.strip() for g in str(gold).split("||") if g.strip()] or [str(gold)]
    p = normalize_text(pred)
    return 1.0 if any(p == normalize_text(g) for g in gts) else 0.0


def parse_judge_binary(text: str) -> float:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        label = str(obj.get("label", "")).upper()
        if "CORRECT" in label and "WRONG" not in label.replace("CORRECT", ""):
            return 1.0
        if label == "CORRECT":
            return 1.0
        return 0.0
    except Exception:
        pass
    upper = text.upper()
    if re.search(r"\bYES\b", upper) and not re.search(r"\bNO\b", upper):
        return 1.0
    if "CORRECT" in upper and "WRONG" not in upper:
        return 1.0
    if re.search(r"\bNO\b", upper):
        return 0.0
    if "WRONG" in upper:
        return 0.0
    return 0.0


class StateTreeCombinedRewardWorker(Worker):
    """
    Combined Exact-Match + LLM-as-a-Judge reward for StateTree / LoCoMo QA.

    Config knobs (via WorkerConfig / YAML):
      - judge_prompt: key in prompt_maps (default: StateTree-LoCoMo-judge)
      - judge_model_type: "api" | "inference" | "em_only"
      - judge_model_name / judge_api_url / judge_api_key for API mode
      - skip_llm_if_em: if true (default), skip judge when EM already matches
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        judge_key = getattr(self.worker_config, "judge_prompt", None) or "StateTree-LoCoMo-judge"
        self.judge_prompt = prompt_maps.get(judge_key, prompt_maps.get("StateTree-LoCoMo-judge"))
        self.judge_model_type = getattr(self.worker_config, "judge_model_type", None) or "em_only"
        self.judge_model_name = getattr(self.worker_config, "judge_model_name", None)
        self.judge_api_url = getattr(self.worker_config, "judge_api_url", None)
        self.judge_api_key = getattr(self.worker_config, "judge_api_key", None)
        self.skip_llm_if_em = bool(getattr(self.worker_config, "skip_llm_if_em", True))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)

        if self.judge_model_type == "em_only":
            self.tokenizer = self.actor_tokenizer
            print(f"{self.worker_name} initialized (EM only)")
            return

        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            print(f"{self.worker_name} initialized with API judge")
            return

        if self.judge_model_type == "inference":
            async_strategy = self.worker_config.strategy_args.strategy_name in ["vllm", "sglang"]
            if self.worker_config.strategy_args.strategy_name == "sglang":
                self.worker_config.strategy_args.strategy_config["enable_weights_cpu_backup"] = True
            if self.worker_config.strategy_args.strategy_name == "vllm":
                self.worker_config.strategy_args.strategy_config["sleep_level"] = 1
            self.strategy = create_strategy(worker=self, sync_wrapper=async_strategy)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            self.strategy.offload_states()
            current_platform.init()
            print(f"{self.worker_name} initialized with local inference judge")
            return

        raise ValueError(f"Unsupported judge_model_type: {self.judge_model_type}")

    def _call_api_model(self, messages: List[Dict[str, str]], retry_times: int = 3) -> str:
        from openai import OpenAI

        output = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API judge")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(api_key=self.judge_api_key, base_url=self.judge_api_url)
                completion = client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=messages,
                    temperature=0.0,
                )
                output = completion.choices[0].message.content or ""
                if output:
                    break
            except Exception as e:
                print(e)
                continue
        return output

    def _run_local_inference(self, messages: List[Dict[str, str]]) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")
        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)
        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(current_platform.device_type)
        attention_mask = tokenized["attention_mask"].to(current_platform.device_type)
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        generation_config["temperature"] = 0.0
        data = DataProto(
            batch=TensorDict(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                batch_size=input_ids.shape[0],
            )
        )
        data = data.to(current_platform.device_type)
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}
        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]) :]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]
        return self.tokenizer.decode(generate_ids[0], skip_special_tokens=True).strip()

    def _llm_judge(self, question: str, pred: str, gold: str) -> float:
        if not self.judge_prompt:
            return 0.0
        formatted = self.judge_prompt.format(
            question=question,
            response=pred,
            reference=gold,
            gold_answer=gold,
            generated_answer=pred,
        )
        messages = [{"role": "user", "content": formatted}]
        if self.judge_model_type == "api":
            raw = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            raw = self._run_local_inference(messages)
        else:
            return 0.0
        return parse_judge_binary(raw)

    def _score_one(self, prompt_txt: str, response: str, gold: str) -> Dict[str, Any]:
        y_ans = extract_boxed(response)
        r_em = exact_match(y_ans, gold)
        r_llm = 0.0
        if self.judge_model_type != "em_only" and not (self.skip_llm_if_em and r_em >= 1.0):
            # Prefer original question if available in prompt tail
            question = prompt_txt
            if "Question:" in prompt_txt:
                question = prompt_txt.split("Question:")[-1].strip()
            r_llm = self._llm_judge(question, y_ans or response, gold)
        reward = max(r_em, r_llm)
        return {"reward": reward, "r_em": r_em, "r_llm": r_llm, "extracted": y_ans}

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics: Dict[str, Any] = {}
        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict[str, Any]):
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        scores = []
        em_scores = []
        llm_scores = []
        for prompt_txt, response, gold in zip(
            prompts_text_list, response_text_list, data.non_tensor_batch["ground_truth"]
        ):
            out = self._score_one(prompt_txt, response, str(gold))
            scores.append(out["reward"])
            em_scores.append(out["r_em"])
            llm_scores.append(out["r_llm"])
            self.logger.info(
                json.dumps(
                    {
                        "reward": out["reward"],
                        "r_em": out["r_em"],
                        "r_llm": out["r_llm"],
                        "extracted": out["extracted"],
                        "gold": gold,
                    },
                    ensure_ascii=False,
                )
            )

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        metrics["statetree/r_em_mean"] = float(sum(em_scores) / max(len(em_scores), 1))
        metrics["statetree/r_llm_mean"] = float(sum(llm_scores) / max(len(llm_scores), 1))
        metrics["statetree/reward_mean"] = float(sum(scores) / max(len(scores), 1))

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": scores_tensor,
                "scores": scores_tensor,
            }
        )
        output.meta_info = {"metrics": metrics}
        return output
