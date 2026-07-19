"""The judge: score a conversation against the rubric with two backends.

``HeuristicJudge`` is deterministic and GPU-free. It judges the three criteria that
are mechanically detectable - unsafe data asks and over-promises (compliance),
vague non-answers (completeness), and cold responses to complaints (tone) - and
*abstains* (``na``) on the two that need real reading, intent resolution and policy
accuracy. That abstention is the honest floor: it is exactly the gap the LLM judge
has to fill, and the gap the kappa validation measures.

``LLMJudge`` runs a local Qwen2.5-3B-Instruct, grounded in the retrieved policy
passages, and returns a verdict + verbatim evidence quote per criterion. Its output
is validated against the typed schema and repaired once if malformed.

``score_conversation`` ties it together: gather context (retrieved or full),
run the backend, time it, and - if a LangSmith key is present - trace it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

from qaudit import config
from qaudit.data.load import Conversation
from qaudit.judge.prompts import SYSTEM, build_judge_prompt
from qaudit.judge.schema import CriterionVerdict, JudgeResult
from qaudit.kb import Passage, load_kb
from qaudit.retrieval import Retriever, build_retriever
from qaudit.rubric.rubric import Criterion, load_rubric

logger = logging.getLogger(__name__)

# --- Heuristic detectors (deterministic) ------------------------------------
# Sensitive credentials an agent must never ask for in chat, and over-promises the
# policy does not support. Both are compliance_safety violations.
_SENSITIVE_RE = re.compile(
    r"\b(password|pin|cvv|security code|full card|card number|account details|"
    r"account information|card details)\b",
    re.I,
)
_ASK_RE = re.compile(
    r"\b(provide|share|send|give|tell me|enter your|confirm your|reply with|reply here|"
    r"with your|need your|read out|verify your)\b",
    re.I,
)
_OVERPROMISE_RE = re.compile(
    r"\b(immediately|instantly|instant refund|right away|guarantee[ds]?|"
    r"straight away|refund now)\b",
    re.I,
)
# Vague deflections vs. concrete, actionable guidance.
_VAGUE_RE = re.compile(
    r"(refer to the (terms|contract)|terms and conditions|contact (our|customer|us)|"
    r"get in touch|reach out to (our|us)|provide me with (your )?account)",
    re.I,
)
_CONCRETE_RE = re.compile(
    r"(your account|follow these|these steps|step \d|\b1\.|log in|sign in|visit|"
    r"navigate|checkout|settings|download)",
    re.I,
)
# Empathy markers expected in a reply to a complaint.
_EMPATHY_RE = re.compile(r"\b(apolog|sorry|understand|appreciate|we're here|regret)\b", re.I)


def _first_sentence_matching(text: str, pattern: re.Pattern[str]) -> str:
    """Return the first sentence in ``text`` that matches ``pattern`` (evidence quote)."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if pattern.search(sentence):
            return sentence.strip()[:200]
    return ""


class Judge(Protocol):
    """Common interface: verdicts for one conversation given policy context."""

    name: str

    def score(
        self, conversation: Conversation, criteria: list[Criterion], passages: list[Passage]
    ) -> tuple[list[CriterionVerdict], bool]: ...


class HeuristicJudge:
    """Deterministic rule-based judge - the CI floor and the LLM's baseline."""

    name = "heuristic"

    def score(
        self, conversation: Conversation, criteria: list[Criterion], passages: list[Passage]
    ) -> tuple[list[CriterionVerdict], bool]:
        agent = conversation.agent_text
        verdicts: list[CriterionVerdict] = []
        for c in criteria:
            verdicts.append(self._score_one(c.id, conversation, agent))
        return verdicts, True

    def _score_one(self, cid: str, conv: Conversation, agent: str) -> CriterionVerdict:
        if cid == "compliance_safety":
            ask = _first_sentence_matching(agent, _SENSITIVE_RE)
            if ask and _ASK_RE.search(ask):
                return CriterionVerdict(criterion_id=cid, verdict="violation", evidence=ask,
                                        rationale="Asks the customer for sensitive credentials in chat.")
            over = _first_sentence_matching(agent, _OVERPROMISE_RE)
            if over:
                return CriterionVerdict(criterion_id=cid, verdict="violation", evidence=over,
                                        rationale="Promises an outcome faster/stronger than policy supports.")
            return CriterionVerdict(criterion_id=cid, verdict="pass", evidence="",
                                    rationale="No unsafe data request or over-promise detected.")
        if cid == "completeness":
            vague = _first_sentence_matching(agent, _VAGUE_RE)
            if vague and not _CONCRETE_RE.search(agent):
                return CriterionVerdict(criterion_id=cid, verdict="violation", evidence=vague,
                                        rationale="Vague deflection with no concrete next step.")
            return CriterionVerdict(criterion_id=cid, verdict="pass", evidence="",
                                    rationale="Gives a concrete step or the needed detail.")
        if cid == "tone_empathy":
            if conv.intent == "complaint" and not _EMPATHY_RE.search(agent):
                return CriterionVerdict(criterion_id=cid, verdict="violation", evidence=agent[:200],
                                        rationale="Complaint met without an apology or acknowledgement.")
            return CriterionVerdict(criterion_id=cid, verdict="pass", evidence="",
                                    rationale="Tone is acceptable for the situation.")
        # intent_resolution and policy_accuracy need real reading: the heuristic abstains.
        return CriterionVerdict(criterion_id=cid, verdict="na", evidence="",
                                rationale="Heuristic backend does not assess this criterion.")


class LLMJudge:
    """Local instruct-model judge (Qwen2.5-3B-Instruct), grounded in policy excerpts."""

    name = "transformers"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
        if config.JUDGE_LOAD_4BIT:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            except Exception as err:  # noqa: BLE001 - fall back to full precision
                logger.warning("4-bit unavailable (%s); loading full precision.", err)
        self.tokenizer = AutoTokenizer.from_pretrained(config.JUDGE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(config.JUDGE_MODEL, **kwargs)

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=config.JUDGE_MAX_NEW_TOKENS, do_sample=False
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def score(
        self, conversation: Conversation, criteria: list[Criterion], passages: list[Passage]
    ) -> tuple[list[CriterionVerdict], bool]:
        prompt = build_judge_prompt(conversation, criteria, passages)
        raw = self._generate(prompt)
        verdicts, valid = _parse_verdicts(raw, criteria)
        if not valid:  # one repair attempt with an explicit instruction
            repair = prompt + "\n\nYour previous output was not valid JSON. Return only the JSON object."
            verdicts, _ = _parse_verdicts(self._generate(repair), criteria)
        return verdicts, valid


# --- Parsing / validation ---------------------------------------------------
def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences/prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(candidate[start : end + 1])


# Per-object recovery: a small 4-bit model reliably emits well-formed
# ``{"criterion_id": ..., "verdict": ..., "evidence": ...}`` objects but sometimes
# emits corrupted tokens *between* them, which breaks a whole-document JSON parse.
# This pulls each object out individually so one glitch doesn't lose every verdict.
_OBJ_RE = re.compile(
    r'"criterion_id"\s*:\s*"(?P<cid>[a-z_]+)"\s*,\s*"verdict"\s*:\s*"(?P<verdict>[a-zA-Z]+)"'
    r'(?:\s*,\s*"evidence"\s*:\s*"(?P<ev>[^"]*)")?',
    re.I,
)


def _recover_verdicts(raw: str) -> dict[str, CriterionVerdict]:
    """Recover one verdict per criterion from a possibly-noisy model response.

    Tries a clean whole-document parse first, then falls back to (and tops up from) a
    per-object regex scan that tolerates junk between the objects.
    """
    by_id: dict[str, CriterionVerdict] = {}
    try:
        for item in _extract_json(raw).get("verdicts", []):
            try:
                v = CriterionVerdict.model_validate(item)
                by_id.setdefault(v.criterion_id, v)
            except Exception:  # noqa: BLE001, S112 - skip a bad row, keep the rest
                continue
    except Exception:  # noqa: BLE001, S110 - whole-document parse failed; use the regex
        pass
    for m in _OBJ_RE.finditer(raw):
        if m.group("cid") in by_id:
            continue
        try:
            by_id[m.group("cid")] = CriterionVerdict(
                criterion_id=m.group("cid"),
                verdict=m.group("verdict"),
                evidence=(m.group("ev") or "").strip(),
            )
        except Exception:  # noqa: BLE001, S112 - not a real criterion/verdict; skip
            continue
    return by_id


def _parse_verdicts(
    raw: str, criteria: list[Criterion]
) -> tuple[list[CriterionVerdict], bool]:
    """Validate the model output into one verdict per criterion.

    Missing entries are filled with a safe ``na`` so the result always covers every
    criterion; ``valid`` reports whether a verdict was recovered for *all* criteria
    without needing a fill (the honest structured-output reliability signal).
    """
    by_id = _recover_verdicts(raw)
    ordered = [
        by_id.get(
            c.id,
            CriterionVerdict(criterion_id=c.id, verdict="na", evidence="",
                             rationale="Missing from model output."),
        )
        for c in criteria
    ]
    valid = all(c.id in by_id for c in criteria)
    return ordered, valid


# --- Orchestration ----------------------------------------------------------
def build_judge(backend: str | None = None) -> Judge:
    """Resolve and construct a judge backend ('heuristic', 'transformers', 'auto')."""
    backend = (backend or config.JUDGE_BACKEND).lower()
    if backend == "auto":
        backend = "transformers" if _llm_available() else "heuristic"
    if backend == "heuristic":
        return HeuristicJudge()
    if backend == "transformers":
        return LLMJudge()
    raise ValueError(f"Unknown judge backend {backend!r}.")


def _llm_available() -> bool:
    """True when a CUDA GPU and the transformers stack are importable."""
    try:
        import torch  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _gather_context(
    conversation: Conversation, context_mode: str, retriever: Retriever, k: int
) -> tuple[list[Passage], list[str]]:
    """Return the policy passages to ground the judge on, and their ids.

    ``retrieved`` gives the top-k passages for the conversation; ``full`` stuffs the
    entire KB. This is the switch the ablation flips.
    """
    if context_mode == "full":
        passages = load_kb()
        return passages, [p.id for p in passages]
    hits = retriever.search(conversation.customer_text, k=k)
    passages = [h.passage for h in hits]
    return passages, [p.id for p in passages]


def score_conversation(
    conversation: Conversation,
    judge: Judge | None = None,
    *,
    backend: str | None = None,
    context_mode: str | None = None,
    retriever: Retriever | None = None,
    k: int = config.TOP_K,
) -> JudgeResult:
    """Score one conversation and return a typed, timed :class:`JudgeResult`."""
    judge = judge or build_judge(backend)
    context_mode = context_mode or config.JUDGE_CONTEXT_MODE
    retriever = retriever or build_retriever()
    criteria = load_rubric()
    passages, ids = _gather_context(conversation, context_mode, retriever, k)
    prompt_chars = len(build_judge_prompt(conversation, criteria, passages))

    def _run() -> tuple[list[CriterionVerdict], bool]:
        return judge.score(conversation, criteria, passages)

    started = time.perf_counter()
    verdicts, valid = _traced(_run, conversation.id, judge.name, context_mode)
    latency = time.perf_counter() - started

    return JudgeResult(
        conversation_id=conversation.id,
        backend=judge.name,
        context_mode=context_mode,
        retrieved_passage_ids=ids,
        verdicts=verdicts,
        schema_valid_first_try=valid,
        prompt_chars=prompt_chars,
        latency_s=round(latency, 3),
    )


def _traced(fn, conv_id: str, backend: str, context_mode: str):  # type: ignore[no-untyped-def]
    """Run ``fn`` under a LangSmith trace when a key is present, else run it plainly.

    Tracing is best-effort and entirely optional: with no ``LANGCHAIN_API_KEY`` set,
    or if ``langsmith`` is not installed, this is a no-op. The free Developer plan is
    hard-capped at 5,000 traces/month for a personal org with no card, so it cannot
    bill - see the README.
    """
    if not config.TRACING_ENABLED:
        return fn()
    try:
        from langsmith import traceable

        wrapped = traceable(
            run_type="llm",
            name="qaudit.judge",
            project_name=config.LANGSMITH_PROJECT,
            metadata={"conversation_id": conv_id, "backend": backend, "context_mode": context_mode},
        )(fn)
        return wrapped()
    except Exception as err:  # noqa: BLE001 - never let tracing break scoring
        logger.warning("LangSmith tracing skipped: %s", err)
        return fn()
