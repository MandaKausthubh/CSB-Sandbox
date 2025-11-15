# pyright: ignoreall
# Place in app/services/analysis_engine.py (or router_analysis.py) and import where needed.

import os
import json
from typing import Dict, Any, List

# Try to import the real LangGraph agent classes (adjust path if you placed file elsewhere)
# try:
from langGraph.Agent import (
    SandboxState,
    PersonaReactionAgent,
    PlatformPolicyAgent,
    LegalEthicsAgentRAG,
    BrandAgent,
    AggregatorAgent,
    FixerAgent,
    LLMClient as LangGraphLLMClient,
    LegalRAGEngine as LangGraphRAGEngine,
)
LANGGRAPH_AVAILABLE = True
# except Exception:
#     LANGGRAPH_AVAILABLE = False

# Try to import your minimal RAG / LLM stubs (fallback)
try:
    from langGraph.llm_client import LLMClientStub  # type: ignore
except Exception:
    # minimal fallback stub
    class LLMClientStub:
        def analyze(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0):
            lower = prompt.lower()
            risky = ["bomb", "kill", "drugs", "attack", "hate", "cure", "guarantee"]
            if any(w in lower for w in risky):
                return json.dumps({
                    "compliance_risk": 0.85,
                    "violating_areas": ["safety", "illegal-content"],
                    "suggestions": ["Remove violent claims", "Add disclaimers"],
                    "enhanced_script": prompt[:200] + " (edited)"
                })
            return json.dumps({
                "compliance_risk": 0.12,
                "violating_areas": [],
                "suggestions": ["Add brand disclaimer"],
                "enhanced_script": prompt[:200] + " (polished)"
            })

try:
    from langGraph.rag_engine import MinimalRAG # type: ignore
except Exception:
    # simple in-file MinimalRAG fallback
    class MinimalRAG:
        def __init__(self):
            self.docs = []

        def ingest(self, texts, metadatas):
            for t, m in zip(texts, metadatas):
                self.docs.append({"text": t, "meta": m})

        def retrieve(self, query: str, k: int = 3):
            q_words = set(query.lower().split())
            scored = []
            for d in self.docs:
                score = len(q_words.intersection(set(d["text"].lower().split())))
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:k]]

# Lazy singletons so we only build RAG once
_RAG_ENGINE = None
_LLM_CLIENT = None


def get_rag_engine():
    global _RAG_ENGINE
    if _RAG_ENGINE is not None:
        return _RAG_ENGINE

    # Prefer the LangGraph RAG engine if available + legal_ingest
    if LANGGRAPH_AVAILABLE:
        try:
            # if you have legal_ingest.build_rag
            from legal_ingest import build_rag
            _RAG_ENGINE = build_rag()
            return _RAG_ENGINE
        except Exception:
            pass

    # fallback: build a MinimalRAG and ingest any local legal_docs/*.txt files
    rag = MinimalRAG()
    try:
        import glob
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # optional; if not installed, do plain
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        files = glob.glob("./legal_docs/*.txt")
        texts, metas = [], []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                raw = fh.read()
            try:
                chunks = splitter.split_text(raw)
            except Exception:
                chunks = [raw]
            for c in chunks:
                texts.append(c)
                metas.append({"source": os.path.basename(f)})
        if texts:
            rag.ingest(texts, metas)
    except Exception:
        # no files or splitters; leave rag empty
        pass

    _RAG_ENGINE = rag
    return _RAG_ENGINE


def get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    # Prefer LangGraph's LLM client if available and env configured
    if LANGGRAPH_AVAILABLE:
        # instantiate with "openai" only if key exists; else use stub
        provider = os.environ.get("LLM_PROVIDER", "openai")
        try:
            lg_llm = LangGraphLLMClient(provider=provider, model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
            _LLM_CLIENT = lg_llm
            return _LLM_CLIENT
        except Exception:
            pass

    # Fallback to your LLMClientStub or local stub
    _LLM_CLIENT = LLMClientStub()
    return _LLM_CLIENT


def _build_suggestions_from_state(state) -> List[Dict[str, Any]]:
    suggestions = []
    # persona suggestions (if any)
    for p in getattr(state, "persona_feedback", []) or []:
        suggestions.append({
            "category": "engagement",
            "title": f"Audience: {p.audience}",
            "description": f"Consider: {', '.join(p.key_concerns[:3]) or 'tone adjustment'}",
            "priority": "medium"
        })
    # brand suggestions
    bf = getattr(state, "brand_feedback", None)
    if bf:
        suggestions.append({
            "category": "compliance",
            "title": "Brand alignment",
            "description": f"Missing disclaimers: {', '.join(bf.missing_disclaimers or [])}",
            "priority": "high" if bf.missing_disclaimers else "low"
        })
    # platform suggestions - if any flagged categories exist
    for pf in getattr(state, "platform_feedback", []) or []:
        if pf.flagged_policy_categories:
            suggestions.append({
                "category": "platform",
                "title": f"Platform: {pf.platform}",
                "description": f"Flagged categories: {', '.join(pf.flagged_policy_categories)}",
                "priority": "medium"
            })
    return suggestions


def _build_violations_from_state(state, params) -> List[Dict[str, Any]]:
    violations = []
    # Legal feedback may be list of dicts (from RAG agent) or LegalFeedback models
    for lf in getattr(state, "legal_feedback", []) or []:
        # lf might be dict or object
        if isinstance(lf, dict):
            region = lf.get("region")
            viols = lf.get("violating_areas", [])
            risk = lf.get("compliance_risk", 0.0)
        else:
            region = getattr(lf, "region", None)
            viols = getattr(lf, "violating_areas", [])
            risk = getattr(lf, "compliance_risk", 0.0)
        for v in viols:
            violations.append({
                "type": "legal",
                "severity": "high" if risk > 0.7 else "medium" if risk > 0.4 else "low",
                "description": f"Potential issue: {v}",
                "affectedRegions": [region] if region else params.get("region", []),
                "recommendation": "Review this area and consult legal counsel if needed."
            })
    return violations


def run_graph_analysis(content: str, params: Dict[str, Any], user_id: int = None) -> Dict[str, Any]:
    """
    Runs the full agent pipeline (Persona -> Platform -> Legal RAG -> Brand -> Aggregator -> Fixer)
    and returns a FE-compatible result dict.
    """
    llm = get_llm_client()
    rag = get_rag_engine()

    # Build a SandboxState (fallback to minimal if class not available)
    if LANGGRAPH_AVAILABLE:
        # use the real SandboxState dataclass / pydantic model
        state = SandboxState(
            input_text=content,
            project_name=None,
            audiences=params.get("targetAudience", []),
            platforms=params.get("platform", []),
            legal_regions=params.get("region", []),
            brand_rules={},  # you may pass sponsor/brand rules here
        )
    else:
        # Minimal stand-in state (duck-typed)
        class _S:
            pass
        state = _S()
        state.input_text = content
        state.project_name = None
        state.audiences = params.get("targetAudience", [])
        state.platforms = params.get("platform", [])
        state.legal_regions = params.get("region", [])
        state.brand_rules = {}

    # Build agents (do not use GraphRunner build_agents to avoid signature mismatch)
    agents = []
    agents.append(PersonaReactionAgent(llm) if LANGGRAPH_AVAILABLE else PersonaReactionAgent(llm))
    agents.append(PlatformPolicyAgent(llm))
    # Legal RAG agent: if we have a special constructor requiring rag, pass it; otherwise try best-effort
    try:
        leg_agent = LegalEthicsAgentRAG(llm, rag)
    except TypeError:
        # maybe signature is (llm,) only
        leg_agent = LegalEthicsAgentRAG(llm)
    agents.append(leg_agent)
    agents.append(BrandAgent(llm))
    agents.append(AggregatorAgent(llm))
    agents.append(FixerAgent(llm))

    # Execute agents sequentially and update `state`
    for a in agents:
        try:
            state = a.run(state)
        except Exception as e:
            # safe fallback: log and continue
            import logging
            logging.exception("Agent %s failed: %s", getattr(a, "name", str(a)), e)
            continue

    # Build final result from state
    overall_risk = getattr(state, "overall_risk", 0.0)
    score = float(overall_risk) * 100.0

    compliance_status = "violation" if overall_risk > 0.7 else "warning" if overall_risk > 0.4 else "compliant"

    violations = _build_violations_from_state(state, params)
    suggestions = _build_suggestions_from_state(state)

    # prefer fixer rewrite if available
    ai_script = state.metadata.get("fixer_rewrite") if hasattr(state, "metadata") and state.metadata.get("fixer_rewrite") else None
    if not ai_script:
        # maybe the Legal RAG supplied an enhanced_script in its last output
        try:
            last_legal = (state.legal_feedback[-1] if getattr(state, "legal_feedback", None) else None)
            if isinstance(last_legal, dict):
                ai_script = last_legal.get("raw") or last_legal.get("enhanced_script")
        except Exception:
            ai_script = None

    ai_script = ai_script or (getattr(state, "input_text", content))

    result = {
        "score": round(score, 2),
        "complianceStatus": compliance_status,
        "violations": violations,
        "suggestions": suggestions,
        "aiEnhancedScript": ai_script,
    }
    return result
