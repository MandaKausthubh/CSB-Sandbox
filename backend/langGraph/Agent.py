"""
creative_sandbox_langgraph.py

Single-file LangGraph-style multi-agent workflow implementation for:
"Creative Safety Sandbox" (PersonaReactionAgent, PlatformPolicyAgent,
LegalEthicsAgent, BrandAgent, AggregatorAgent, FixerAgent)

Usage:
    - pip install pydantic openai requests (optional)
    - export OPENAI_API_KEY or configure another LLM client
    - python creative_sandbox_langgraph.py

This file is intentionally modular:
 - Agent classes are small and replaceable.
 - LLM client layer is pluggable (OpenAI or simple stub).
 - The GraphRunner executes nodes in a deterministic LangGraph-like order.
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Optional: import OpenAI SDK if available
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------- Data models ----------
class PersonaFeedback(BaseModel):
    audience: str
    sentiment: str
    backlash_likelihood: float
    key_concerns: List[str]
    details: Optional[str] = None


class PlatformFeedback(BaseModel):
    platform: str
    demonetization_likelihood: float
    removal_risk: float
    flagged_policy_categories: List[str]
    details: Optional[str] = None


class LegalFeedback(BaseModel):
    region: str
    compliance_risk: float
    violating_areas: List[str]
    notes: Optional[str] = None


class BrandFeedback(BaseModel):
    tone_alignment_score: float
    style_violations: List[str]
    missing_disclaimers: List[str]
    forbidden_terms_found: List[str]
    details: Optional[str] = None


class SandboxState(BaseModel):
    input_text: str
    project_name: Optional[str] = None
    audiences: List[str] = []
    platforms: List[str] = []
    legal_regions: List[str] = []
    brand_rules: Dict[str, Any] = {}
    persona_feedback: List[PersonaFeedback] = []
    platform_feedback: List[PlatformFeedback] = []
    legal_feedback: List[LegalFeedback] = []
    brand_feedback: Optional[BrandFeedback] = None
    aggregated_summary: Optional[str] = None
    overall_risk: float = 0.0
    risk_breakdown: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


# ---------- LLM client (pluggable) ----------
class LLMClient:
    """
    Minimal wrapper for LLM calls. By default tries to use OpenAI if OPENAI_API_KEY is set.
    You can subclass/replace this with Google PaLM or local model logic.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        if self.provider == "openai" and OPENAI_AVAILABLE:
            # set API key if present
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                openai.api_key = key

    def analyze(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Do a synchronous LLM call; returns text.
        If OpenAI isn't available, returns a deterministic stub summary.
        """
        logging.debug("LLMClient.analyze called with provider=%s", self.provider)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            # Safe minimal OpenAI call:
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are an assistant that analyzes creative content for safety and brand alignment."},
                              {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp["choices"][0]["message"]["content"].strip()
                logging.debug("OpenAI response length=%d", len(text))
                return text
            except Exception as e:
                logging.warning("OpenAI call failed: %s. Falling back to stub.", e)
        # Fallback deterministic mock response:
        return self._stub_response(prompt)

    def _stub_response(self, prompt: str) -> str:
        # Very simple heuristic stub: look for risky keywords and craft a short summary
        lowered = prompt.lower()
        risk_terms = ["kill", "bomb", "drugs", "hate", "violence", "attack", "misinformation"]
        hits = [t for t in risk_terms if t in lowered]
        if hits:
            return f"RISK_FOUND: keywords={hits}. Recommend rewrite to remove/clarify claims. Likelihood: high."
        if "brand" in lowered and "tone" in lowered:
            return "Brand tone looks friendly; slight mismatch on formality. Likelihood: low."
        # default
        return "No major issues detected. Sentiment: neutral to positive. Likelihood: low."


# ---------- Base Agent ----------
class Agent:
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm

    def run(self, state: SandboxState) -> SandboxState:
        """
        Should be overridden by subclasses. Must return (and may mutate) the SandboxState.
        """
        raise NotImplementedError


# ---------- Concrete Agents ----------
class PersonaReactionAgent(Agent):
    """
    Simulates audience reactions. For each audience in state.audiences produces a PersonaFeedback.
    """

    def __init__(self, llm: LLMClient, audiences_to_simulate: Optional[List[str]] = None):
        super().__init__("PersonaReactionAgent", llm)
        self.default_audiences = audiences_to_simulate or ["Gen Z", "Parents", "Teachers", "Global markets", "Sensitive groups"]

    def run(self, state: SandboxState) -> SandboxState:
        audiences = state.audiences or self.default_audiences
        outputs: List[PersonaFeedback] = []
        for a in audiences:
            prompt = self._build_prompt(audience=a, state=state)
            resp_text = self.llm.analyze(prompt)
            # Parse response heuristically - robust parsing needed in real product
            sentiment, likelihood, concerns = self._heuristic_parse_persona(resp_text)
            pf = PersonaFeedback(
                audience=a,
                sentiment=sentiment,
                backlash_likelihood=likelihood,
                key_concerns=concerns,
                details=resp_text[:1000],
            )
            outputs.append(pf)
            logging.info("Persona %s: sentiment=%s, backlash=%.2f", a, sentiment, likelihood)
        state.persona_feedback = outputs
        # update intermediate risk
        avg = sum(p.backlash_likelihood for p in outputs) / (len(outputs) or 1)
        state.risk_breakdown["persona"] = round(avg, 4)
        return state

    def _build_prompt(self, audience: str, state: SandboxState) -> str:
        return (
            f"Simulate how the '{audience}' audience would react to the following creative text. "
            f"Return a short analysis including: sentiment (positive/neutral/negative), "
            f"backlash likelihood as a percent (0-100), and up to 5 key concerns. Text:\n\n{state.input_text}\n\n"
            f"Brand rules: {json.dumps(state.brand_rules)}"
        )

    def _heuristic_parse_persona(self, text: str) -> Tuple[str, float, List[str]]:
        txt = text.lower()
        if "risk" in txt or "high" in txt or "backlash" in txt:
            likelihood = min(1.0, 0.7 + 0.1 * txt.count("!"))
            sentiment = "negative"
        elif "low" in txt or "no major" in txt or "none" in txt:
            likelihood = 0.05
            sentiment = "positive"
        else:
            likelihood = 0.2
            sentiment = "neutral"
        concerns = []
        if ":" in text:
            # try to extract after 'key concerns' etc - naive
            parts = text.split("key concerns")
            if len(parts) > 1:
                concerns = [c.strip() for c in parts[1].split(",")][:5]
        # fallback: derive from words
        if not concerns:
            for w in ["safety", "offensive", "misinformation", "tone", "sexual"]:
                if w in txt:
                    concerns.append(w)
        return sentiment, round(likelihood, 4), concerns


class PlatformPolicyAgent(Agent):
    """
    Evaluates risk per platform (YouTube, Instagram, TikTok, etc.)
    """

    def __init__(self, llm: LLMClient, platforms_to_check: Optional[List[str]] = None):
        super().__init__("PlatformPolicyAgent", llm)
        self.default_platforms = platforms_to_check or ["YouTube", "Instagram", "TikTok"]

    def run(self, state: SandboxState) -> SandboxState:
        platforms = state.platforms or self.default_platforms
        outputs: List[PlatformFeedback] = []
        for p in platforms:
            prompt = self._build_prompt(platform=p, state=state)
            resp_text = self.llm.analyze(prompt)
            demonet, removal, cats = self._heuristic_parse_platform(resp_text)
            pf = PlatformFeedback(
                platform=p,
                demonetization_likelihood=demonet,
                removal_risk=removal,
                flagged_policy_categories=cats,
                details=resp_text[:1000],
            )
            outputs.append(pf)
            logging.info("Platform %s: demonet=%.2f removal=%.2f cats=%s", p, demonet, removal, cats)
        state.platform_feedback = outputs
        # average platform risk
        avg = sum((p.demonetization_likelihood + p.removal_risk) / 2.0 for p in outputs) / (len(outputs) or 1)
        state.risk_breakdown["platform"] = round(avg, 4)
        return state

    def _build_prompt(self, platform: str, state: SandboxState) -> str:
        return (
            f"You are an expert on {platform} content moderation rules. For the text below, "
            f"return (1) demonetization likelihood 0-100, (2) removal risk 0-100, (3) policy categories triggered "
            f"as a comma-separated list, and (4) a one-line reason. Text:\n\n{state.input_text}"
        )

    def _heuristic_parse_platform(self, text: str) -> Tuple[float, float, List[str]]:
        txt = text.lower()
        demonet = 0.1
        removal = 0.05
        cats = []
        if "hate" in txt or "violence" in txt or "child" in txt:
            demonet = 0.8
            removal = 0.6
            cats = ["hate", "violence", "child-safety"]
        elif "misinformation" in txt or "false" in txt:
            demonet = 0.5
            removal = 0.2
            cats = ["misinformation"]
        return round(demonet, 4), round(removal, 4), cats

class LegalRAGEngine:
    """
    RAG engine for retrieving legal/policy content based on similarity search.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.emb = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_db = None
        self.documents = []

    def ingest(self, docs: List[str], metadata: List[Dict[str, Any]]):
        """
        docs: List of raw text chunks
        metadata: List of metadata dicts for each doc
        Creates FAISS index
        """

        ds = []
        for text, meta in zip(docs, metadata):
            ds.append(Document(page_content=text, metadata=meta))

        self.vector_db = FAISS.from_documents(ds, self.emb)
        self.documents = ds

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve top-k most relevant legal/policy chunks.
        """
        if self.vector_db is None:
            raise RuntimeError("LegalRAGEngine: vector DB not built! Call ingest() first.")
        return self.vector_db.similarity_search(query, k=k)

class LegalEthicsAgentRAG:
    """
    Uses RAG + LLM to analyze legal, safety, and compliance issues across jurisdictions.
    """

    def __init__(self, llm_client, rag_engine: LegalRAGEngine, regions_to_check=None):
        self.name = "LegalEthicsAgentRAG"
        self.llm = llm_client
        self.rag = rag_engine
        self.default_regions = regions_to_check or ["US", "EU", "IN"]

    def run(self, state):
        regions = state.legal_regions or self.default_regions
        outputs = []

        for region in regions:
            retrieved_docs = self.rag.retrieve(
                query=state.input_text + f"\nJurisdiction: {region}",
                k=5
            )

            legal_context = "\n\n".join(
                [f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in retrieved_docs]
            )

            prompt = self.build_prompt(region, state.input_text, legal_context)
            resp_text = self.llm.analyze(prompt, temperature=0.2)

            compliance_risk, violating_areas, citations = self.parse_json(resp_text)

            outputs.append(
                {
                    "region": region,
                    "compliance_risk": compliance_risk,
                    "violating_areas": violating_areas,
                    "citations": citations,
                    "raw": resp_text,
                }
            )

        # Store back into state
        state.legal_feedback = outputs
        avg_risk = sum(o["compliance_risk"] for o in outputs) / len(outputs)
        state.risk_breakdown["legal"] = round(avg_risk, 4)

        return state

    def build_prompt(self, region: str, content: str, context: str):
        return f"""
You are a legal compliance AI specializing in {region} content law.

Analyze the user's text **using ONLY the legal context provided**.

==========================
LEGAL CONTEXT (RAG)
==========================
{context}

==========================
USER CONTENT
==========================
{content}

==========================
TASK
==========================
Return a STRICT JSON with:

- compliance_risk: float (0 to 1)
- violating_areas: list[str]
- citations: list[str]

Example:
{{
  "compliance_risk": 0.62,
  "violating_areas": ["child-safety", "regulated-claim"],
  "citations": ["FTC_Advertising_4.2", "EU_DSA_Article_14"]
}}

IMPORTANT:
- You may ONLY use knowledge from the supplied context.
- Cite each finding using the metadata keys in the context.

Now generate the JSON:
"""

    def parse_json(self, text):
        try:
            js = json.loads(text)
            return (
                float(js.get("compliance_risk", 0)),
                js.get("violating_areas", []),
                js.get("citations", []),
            )
        except Exception:
            # fallback if model fails
            return 0.3, ["uncertain"], []



#
# class LegalEthicsAgent(Agent):
#     """
#     Checks region-specific compliance and ethical flags.
#     """
#
#     def __init__(self, llm: LLMClient, regions_to_check: Optional[List[str]] = None):
#         super().__init__("LegalEthicsAgent", llm)
#         self.default_regions = regions_to_check or ["US", "EU", "IN"]
#
#     def run(self, state: SandboxState) -> SandboxState:
#         regions = state.legal_regions or self.default_regions
#         outputs: List[LegalFeedback] = []
#         for r in regions:
#             prompt = self._build_prompt(region=r, state=state)
#             resp_text = self.llm.analyze(prompt)
#             risk, viols = self._heuristic_parse_legal(resp_text)
#             lf = LegalFeedback(
#                 region=r,
#                 compliance_risk=risk,
#                 violating_areas=viols,
#                 notes=resp_text[:1000],
#             )
#             outputs.append(lf)
#             logging.info("Legal %s: compliance_risk=%.2f violations=%s", r, risk, viols)
#         state.legal_feedback = outputs
#         avg = sum(l.compliance_risk for l in outputs) / (len(outputs) or 1)
#         state.risk_breakdown["legal"] = round(avg, 4)
#         return state
#
#     def _build_prompt(self, region: str, state: SandboxState) -> str:
#         return (
#             f"As a lawyer familiar with {region} laws, analyze the below text for compliance issues: child-safety, defamation, "
#             f"regulated claims, hate speech, medical claims, and consumer protection. For each issue, return probability 0-100 "
#             f"and short justification. Text:\n\n{state.input_text}"
#         )
#
#     def _heuristic_parse_legal(self, txt: str) -> Tuple[float, List[str]]:
#         t = txt.lower()
#         risk = 0.05
#         viols = []
#         if "defamation" in t or "false claim" in t:
#             risk = 0.6
#             viols.append("defamation")
#         if "child" in t or "minor" in t:
#             risk = max(risk, 0.7)
#             viols.append("child-safety")
#         if "medical" in t or "health" in t:
#             risk = max(risk, 0.4)
#             viols.append("regulated-claims")
#         return round(risk, 4), viols
#

class BrandAgent(Agent):
    """
    Checks brand consistency, forbidden terms, disclaimers and tone.
    """

    def __init__(self, llm: LLMClient):
        super().__init__("BrandAgent", llm)

    def run(self, state: SandboxState) -> SandboxState:
        prompt = self._build_prompt(state)
        resp_text = self.llm.analyze(prompt)
        tone_score, style_violations, missing, forbidden = self._heuristic_parse_brand(resp_text)
        bf = BrandFeedback(
            tone_alignment_score=tone_score,
            style_violations=style_violations,
            missing_disclaimers=missing,
            forbidden_terms_found=forbidden,
            details=resp_text[:2000],
        )
        state.brand_feedback = bf
        state.risk_breakdown["brand"] = round(max(0.0, 1.0 - tone_score), 4)
        logging.info("Brand: tone_score=%.2f violations=%s forbidden=%s", tone_score, style_violations, forbidden)
        return state

    def _build_prompt(self, state: SandboxState) -> str:
        return (
            "You are a brand analyst. Consider the brand rules below and the creative text. "
            "Return a JSON-like short analysis containing: tone_alignment_score (0-1), list of style_violations, "
            "list of missing_disclaimers, list of forbidden_terms_found, and one-line advice.\n\n"
            f"Brand rules: {json.dumps(state.brand_rules)}\n\nText:\n{state.input_text}"
        )

    def _heuristic_parse_brand(self, text: str) -> Tuple[float, List[str], List[str], List[str]]:
        t = text.lower()
        tone = 0.8
        violations = []
        missing = []
        forbidden = []
        if "formal" in t and "casual" in t:
            tone = 0.5
        for w in ["disclaimer", "terms"]:
            if w in t and "missing" in t:
                missing.append(w)
        for w in ["slur", "offensive", "forbidden"]:
            if w in t:
                forbidden.append(w)
        if "violation" in t:
            violations.append("style-guideline")
        return round(tone, 4), violations, missing, forbidden


class AggregatorAgent(Agent):
    """
    Aggregates risk signals into overall_risk and a human-readable summary.
    """

    def __init__(self, llm: LLMClient, weights: Optional[Dict[str, float]] = None):
        super().__init__("AggregatorAgent", llm)
        # weights sum should ideally be 1.0; defaults if not provided
        self.weights = weights or {"persona": 0.3, "platform": 0.25, "legal": 0.3, "brand": 0.15}

    def run(self, state: SandboxState) -> SandboxState:
        # compute weighted sum of risk_breakdown
        breakdown = state.risk_breakdown
        logging.debug("Aggregator sees breakdown: %s", breakdown)
        total = 0.0
        total_weight = 0.0
        for k, w in self.weights.items():
            val = float(breakdown.get(k, 0.0))
            total += val * w
            total_weight += w
        overall = (total / total_weight) if total_weight else 0.0
        # clamp 0-1
        overall = max(0.0, min(1.0, overall))
        state.overall_risk = round(overall, 4)
        # build summary
        summary = self._build_summary(state)
        state.aggregated_summary = summary
        logging.info("Aggregated overall_risk=%.4f", state.overall_risk)
        return state

    def _build_summary(self, state: SandboxState) -> str:
        lines = []
        lines.append(f"Overall risk score: {state.overall_risk:.4f} (0 safe — 1 high risk)")
        lines.append("Breakdown:")
        for k, v in state.risk_breakdown.items():
            lines.append(f" - {k}: {v:.4f}")
        # include top concerns
        concerns = set()
        for p in state.persona_feedback:
            concerns.update(p.key_concerns)
        for pf in state.platform_feedback:
            concerns.update(pf.flagged_policy_categories)
        for lf in state.legal_feedback:
            concerns.update(lf.violating_areas)
        if state.brand_feedback:
            concerns.update(state.brand_feedback.forbidden_terms_found)
            concerns.update(state.brand_feedback.style_violations)
        if concerns:
            lines.append("Top aggregated concerns: " + ", ".join([str(c) for c in list(concerns)[:8]]))
        return "\n".join(lines)


class FixerAgent(Agent):
    """
    Produces safer rewrites and suggestions when risk exceeds threshold.
    """

    def __init__(self, llm: LLMClient, risk_threshold: float = 0.35):
        super().__init__("FixerAgent", llm)
        self.risk_threshold = risk_threshold

    def run(self, state: SandboxState) -> SandboxState:
        if state.overall_risk <= self.risk_threshold:
            logging.info("FixerAgent: overall risk (%.4f) <= threshold (%.4f): skipping fixer.", state.overall_risk, self.risk_threshold)
            return state
        # Create a prompt that instructs the model to produce a safer rewrite
        prompt = self._build_prompt(state)
        resp_text = self.llm.analyze(prompt, temperature=0.6, max_tokens=800)
        # store rewrite in metadata
        state.metadata["fixer_rewrite"] = resp_text
        # naive reduction of risk after fix (in real system you'd re-run agents)
        state.overall_risk = max(0.0, state.overall_risk - 0.35)
        state.risk_breakdown = {k: max(0.0, v - 0.25) for k, v in state.risk_breakdown.items()}
        logging.info("FixerAgent produced rewrite; new overall_risk=%.4f", state.overall_risk)
        return state

    def _build_prompt(self, state: SandboxState) -> str:
        return (
            "You are a content safety editor. Create a safer rewrite of the text below that: "
            "(1) reduces legal/policy/brand risks, (2) preserves the main creative message and tone if possible, "
            "(3) highlight changes you made in a short bullet list. Return 'REWRITE:' followed by the text, "
            "then 'CHANGES:' and bullets.\n\nOriginal Text:\n" + state.input_text + "\n\nBrand rules: " + json.dumps(state.brand_rules)
        )


# ---------- Graph runner (LangGraph-style) ----------
@dataclass
class GraphRunner:
    """
    Very small LangGraph-like runner.
    Nodes are executed in the order: initialization -> persona -> platform -> legal -> brand -> aggregator -> conditional fixer -> end.
    """

    llm_client: LLMClient
    risk_threshold: float = 0.35

    def build_agents(self) -> List[Agent]:
        # create agents with shared LLM client
        return [
            PersonaReactionAgent(self.llm_client),
            PlatformPolicyAgent(self.llm_client),
            LegalEthicsAgentRAG(self.llm_client, rag_engine=self.rag_engine), # type: ignore
            BrandAgent(self.llm_client),
            AggregatorAgent(self.llm_client),
            FixerAgent(self.llm_client, risk_threshold=self.risk_threshold),
        ]

    def run(self, state: SandboxState) -> SandboxState:
        logging.info("GraphRunner starting for project: %s", state.project_name)
        agents = self.build_agents()

        # Map names to agents for clarity (optional)
        name_to_agent = {a.name: a for a in agents}

        # Execute sequentially: persona -> platform -> legal -> brand -> aggregator
        for agent_name in ["PersonaReactionAgent", "PlatformPolicyAgent", "LegalEthicsAgent", "BrandAgent", "AggregatorAgent"]:
            agent = name_to_agent[agent_name]
            logging.info("Running agent: %s", agent_name)
            state = agent.run(state)

        # conditional fixer
        if state.overall_risk > self.risk_threshold:
            logging.info("overall_risk %.4f > threshold %.4f -> running FixerAgent", state.overall_risk, self.risk_threshold)
            state = name_to_agent["FixerAgent"].run(state)
        else:
            logging.info("overall_risk %.4f <= threshold %.4f -> skipping FixerAgent", state.overall_risk, self.risk_threshold)

        logging.info("GraphRunner finished. overall_risk=%.4f", state.overall_risk)
        return state


# ---------- Utilities / Example usage ----------
def example_input() -> SandboxState:
    txt = (
        "Launching our new energy drink! Guaranteed to boost your brain power and outpace competitors. "
        "Use daily for best results. Also, in some places, the new formula gave people headaches but that's just temporary."
    )
    brand = {
        "name": "Sparkle Labs",
        "tone": "upbeat, witty",
        "forbidden_terms": ["guaranteed", "cure", "patent-pending-claims"],
        "required_disclaimers": ["Not medical advice", "Consult a doctor if you have health issues"]
    }
    s = SandboxState(
        input_text=txt,
        project_name="Sparkle Launch",
        audiences=["Gen Z", "Parents", "Health-conscious adults"],
        platforms=["YouTube", "Instagram"],
        legal_regions=["US", "EU"],
        brand_rules=brand,
    )
    return s


def print_report(state: SandboxState):
    print("\n==== Creative Safety Sandbox Report ====\n")
    print(f"Project: {state.project_name}")
    print(f"Overall risk: {state.overall_risk:.4f}\n")
    print("Risk breakdown:")
    for k, v in state.risk_breakdown.items():
        print(f"  - {k}: {v:.4f}")
    print("\nPersona feedback (sample):")
    for p in state.persona_feedback:
        print(f" * {p.audience}: sentiment={p.sentiment} backlash={p.backlash_likelihood} concerns={p.key_concerns}")
    print("\nPlatform feedback (sample):")
    for pf in state.platform_feedback:
        print(f" * {pf.platform}: demonet={pf.demonetization_likelihood} removal={pf.removal_risk} cats={pf.flagged_policy_categories}")
    print("\nLegal feedback (sample):")
    for lf in state.legal_feedback:
        print(f" * {lf.region}: risk={lf.compliance_risk} violations={lf.violating_areas}")
    if state.brand_feedback:
        print("\nBrand feedback:")
        print(json.dumps(state.brand_feedback.dict(), indent=2))
    if "fixer_rewrite" in state.metadata:
        print("\nSafer rewrite (FixerAgent):\n", state.metadata["fixer_rewrite"][:1000])
    print("\nAggregated summary:\n")
    print(state.aggregated_summary)
    print("\n==== End Report ====\n")


def main():
    # Choose LLM provider: "openai" or "stub"
    provider = os.environ.get("LLM_PROVIDER", "openai")
    # If openai package isn't available or key missing, fallback to stub
    if provider == "openai" and not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        logging.warning("OpenAI provider requested but not available/authorized; falling back to stub provider.")
        provider = "stub"
    llm_client = LLMClient(provider=provider, model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))

    state = example_input()
    runner = GraphRunner(llm_client=llm_client, risk_threshold=0.35)
    final_state = runner.run(state)
    print_report(final_state)


if __name__ == "__main__":
    main()
