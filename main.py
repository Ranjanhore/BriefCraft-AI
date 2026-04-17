# =========================
# AI CREATIVE STUDIO - MAIN
# =========================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# MODELS
# =========================

class UserInput(BaseModel):
    input: str

class SelectConcept(BaseModel):
    concept_index: int


# =========================
# PROJECT STATE (MEMORY)
# =========================

class ProjectState:
    def __init__(self):
        self.brief: Optional[str] = None
        self.analysis: Optional[str] = None
        self.concepts: Optional[str] = None
        self.selected_concept: Optional[str] = None
        self.layout: Optional[str] = None
        self.graphics: Optional[str] = None
        self.render: Optional[str] = None
        self.production: Optional[str] = None


# =========================
# BASE LLM CALL
# =========================

def llm(prompt):
    res = client.chat.completions.create(
        model="gpt-5.3",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# =========================
# AGENTS
# =========================

class AnalysisAgent:
    def run(self, brief):
        prompt = f"""
        You are a senior creative strategist.

        Analyze this brief:
        {brief}

        Output:
        - Audience
        - Goal
        - Tone
        - Constraints
        """
        return llm(prompt)


class ConceptAgent:
    def run(self, analysis):
        prompt = f"""
        Based on this analysis:

        {analysis}

        Create 3 high-end creative concepts.

        Each should include:
        - Name
        - Theme
        - Short description
        """
        return llm(prompt)


class LayoutAgent:
    def run(self, concept):
        prompt = f"""
        Create spatial CAD layout plan for:

        {concept}

        Include:
        - Zones
        - Visitor flow
        - Dimensions
        """
        return llm(prompt)


class GraphicsAgent:
    def run(self, concept):
        prompt = f"""
        Create 2D graphics plan for:

        {concept}

        Include:
        - LED content
        - Branding panels
        """
        return llm(prompt)


class RenderAgent:
    def run(self, concept):
        prompt = f"""
        Create ultra-realistic 3D render prompts for:

        {concept}

        Include:
        - Camera angle
        - Lighting
        - Materials
        """
        return llm(prompt)


class ProductionAgent:
    def run(self, concept):
        prompt = f"""
        Create production drawings and execution plan for:

        {concept}

        Include:
        - Materials
        - Fabrication
        """
        return llm(prompt)


# =========================
# ORCHESTRATOR
# =========================

class Orchestrator:

    def __init__(self):
        self.state = ProjectState()

        self.analysis_agent = AnalysisAgent()
        self.concept_agent = ConceptAgent()
        self.layout_agent = LayoutAgent()
        self.graphics_agent = GraphicsAgent()
        self.render_agent = RenderAgent()
        self.production_agent = ProductionAgent()

    def run(self, user_input):

        # STEP 1: STORE BRIEF
        if not self.state.brief:
            self.state.brief = user_input

        # STEP 2: ANALYSIS
        if not self.state.analysis:
            self.state.analysis = self.analysis_agent.run(self.state.brief)
            return {"stage": "analysis", "data": self.state.analysis}

        # STEP 3: CONCEPTS
        if not self.state.concepts:
            self.state.concepts = self.concept_agent.run(self.state.analysis)
            return {"stage": "concepts", "data": self.state.concepts}

        # WAIT FOR USER SELECTION
        if not self.state.selected_concept:
            return {"stage": "select_concept", "data": self.state.concepts}

        # STEP 4: LAYOUT
        if not self.state.layout:
            self.state.layout = self.layout_agent.run(self.state.selected_concept)
            return {"stage": "layout", "data": self.state.layout}

        # STEP 5: GRAPHICS
        if not self.state.graphics:
            self.state.graphics = self.graphics_agent.run(self.state.selected_concept)
            return {"stage": "graphics", "data": self.state.graphics}

        # STEP 6: RENDER
        if not self.state.render:
            self.state.render = self.render_agent.run(self.state.selected_concept)
            return {"stage": "render", "data": self.state.render}

        # STEP 7: PRODUCTION
        if not self.state.production:
            self.state.production = self.production_agent.run(self.state.selected_concept)
            return {"stage": "production", "data": self.state.production}

        return {"stage": "complete", "data": "Project fully generated"}


orch = Orchestrator()


# =========================
# API ROUTES
# =========================

@app.post("/run")
def run_agent(req: UserInput):
    return orch.run(req.input)


@app.post("/select-concept")
def select_concept(req: SelectConcept):
    orch.state.selected_concept = f"Concept {req.concept_index + 1}"
    return {"status": "concept selected"}
