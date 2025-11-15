# **Creative Safety Sandbox — Summary for PPT Generation**

## **Project Title:**

**Creative Safety Sandbox (CSS): A Multi-Agent AI System for Content Risk Analysis**

## **One-Line Pitch:**

An AI-powered multi-agent platform that stress-tests creative content—scripts, captions, ads—against audience reactions, brand rules, platform policies, and legal/ethical risks.

---

# **Problem Statement:**

Creators and brands constantly produce content for YouTube, Instagram, TikTok, and other platforms. But:

* Content can unintentionally offend specific audiences
* Platforms often remove or demonetize content due to policy violations
* Legal issues (hate speech, minors, misinformation) can arise
* Brand tone and guidelines are often ignored
* Teams lack tools to reliably **validate content before posting**

This leads to **rework, takedowns, bad PR, and brand damage**.

---

# **Solution — Creative Safety Sandbox (CSS):**

A **multi-agent AI review board** that evaluates content BEFORE it goes live.

The user uploads a script, text caption, or ad copy.
CSS then runs a coordinated team of AI agents—modeled like a real creative review team:

1. **Persona Reaction Agents**
   Simulate how different audience groups (Gen Z, parents, teachers, global regions) would react.

2. **Platform Policy Agents**
   Evaluate content using YouTube, Instagram, TikTok moderation rules.

3. **Legal + Ethics Agents**
   Check for hate speech, minors safety, harmful claims, bias, misinformation.

4. **Brand Consistency Agent**
   Ensures content matches tone, avoids forbidden words, and respects brand style.

5. **Aggregator Agent**
   Summarizes all findings and computes an Overall Risk Score.

6. **Fixer Agent** *(conditional)*
   If risk is high, the agent automatically produces a **safer, on-brand rewrite** of the content.

---

# **Final Output to User:**

### **A complete safety report including:**

* Overall risk score (0–100)
* Audience reaction breakdown
* Platform violation warnings
* Legal/Ethical flags
* Brand consistency check
* Highlighted risky sentences
* AI-generated safe rewrite of the content

### **Plus an agent timeline:**

Shows how each AI agent contributed to the decision.

---

# **User Flow:**

1. User uploads text/script content
2. Selects audiences, platforms, legal regions
3. Clicks “Run Sandbox Analysis”
4. Multi-agent system runs in backend
5. User sees dynamic dashboard + safety report
6. Optional: Download PDF or copy the safe rewrite

---

# **Tech Stack:**

**Backend:** FastAPI + LangGraph
**AI Models:** Open-source LLMs + toxicity classifiers
**Orchestration:** LangGraph for multi-agent workflow
**Database:** PostgreSQL / SQLite
**Frontend:** Next.js + Tailwind CSS
**Vector Storage (optional):** FAISS
**Architecture:** Multi-agent graph with shared state (SandboxState)

---

# **Why This Idea Stands Out:**

* Truly multi-agent, collaborative system
* Solves a real problem in marketing + creator economy
* Unique “stress-test” perspective instead of typical content generation
* Produces actionable outputs (risk score + safe rewrite)
* Very visual and demo-friendly for hackathon judges
* Strong alignment with ethical AI and brand safety trends

---

# **Impact:**

CSS empowers creators and brands to publish confidently—reducing takedowns, legal issues, and PR disasters, while improving content quality across platforms.

