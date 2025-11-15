import json
import os

class LLMClientStub:
    """Simple deterministic fallback for hackathon."""
    def analyze(self, prompt: str):
        lower = prompt.lower()
        risky = ["bomb", "kill", "drugs", "attack", "hate", "cure", "guarantee"]

        if any(w in lower for w in risky):
            return json.dumps({
                "compliance_risk": 0.85,
                "violating_areas": ["safety", "illegal-content"],
                "suggestions": ["Remove violent claims", "Add disclaimers"],
                "enhanced_script": prompt[:200] + " (edited)"
            })
        else:
            return json.dumps({
                "compliance_risk": 0.12,
                "violating_areas": [],
                "suggestions": ["Add brand disclaimer"],
                "enhanced_script": prompt[:200] + " (polished)"
            })

