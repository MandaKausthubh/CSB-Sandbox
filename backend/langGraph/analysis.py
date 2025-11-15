import json
from typing import Dict, Any
from langGraph.rag_engine import MinimalRAG
from langGraph.llm_client import LLMClientStub

rag = MinimalRAG()
rag.ingest(
    texts=[
        "FTC Guidelines prohibit unsubstantiated medical claims.",
        "YouTube prohibits violent or harmful content.",
        "EU DSA requires transparency in content moderation."
    ],
    metas=[{"src":"FTC"},{"src":"YT"},{"src":"EU_DSA"}]
)

llm = LLMClientStub()

def run_analysis(content: str, params: Dict[str, Any]) -> Dict[str, Any]:
    region = params["region"][0] if params["region"] else "US"
    query = f"{content}\nRegion: {region}"

    retrieved = rag.retrieve(query, k=3)
    context = "\n".join([d["text"] for d in retrieved])

    prompt = f"""
CONTEXT:
{context}

CONTENT:
{content}

Return JSON:
{{
  "compliance_risk": ...,
  "violating_areas": [...],
  "suggestions": [...],
  "enhanced_script": "..."
}}
"""
    raw = llm.analyze(prompt)

    try:
        base = json.loads(raw)
    except:
        base = {
            "compliance_risk": 0.5,
            "violating_areas": ["unknown"],
            "suggestions": ["General improvement recommended."],
            "enhanced_script": content + "\n\n(Edited)",
        }

    # Map violations into FE format
    violations = []
    for v in base.get("violating_areas", []):
        violations.append({
            "type": "legal",
            "severity": "medium",
            "description": f"Potential violation detected: {v}",
            "affectedRegions": params["region"],
            "recommendation": "Review the content and adjust wording for compliance."
        })

    # Map suggestions into FE format
    suggestions = []
    for s in base.get("suggestions", []):
        suggestions.append({
            "category": "compliance",
            "title": "Content Suggestion",
            "description": s,
            "priority": "medium"
        })

    return {
        "score": float(base["compliance_risk"]) * 100,
        "complianceStatus": (
            "violation" if base["compliance_risk"] > 0.7 else
            "warning" if base["compliance_risk"] > 0.4 else
            "compliant"
        ),
        "violations": violations,
        "suggestions": suggestions,
        "aiEnhancedScript": base.get("enhanced_script", content),
    }

