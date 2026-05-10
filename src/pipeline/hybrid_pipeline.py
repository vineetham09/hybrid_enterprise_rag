from src.retrieval.retriever_v1 import SemanticRetriever
from src.retrieval.bm25_retriever import BM25RetrieverCustom
from src.structured.structured_store import StructuredStore
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.reranker import Reranker
from src.pipeline.router import ClassifyQuery

import re
import spacy


class HybridPipelineFinal:

    def __init__(self):
        self.structured_store = StructuredStore("data/knowledge_base")
        self.vector_retriever = SemanticRetriever()
        self.bm25_retriever = BM25RetrieverCustom()
        self.generator = AnswerGenerator()
        self.reranker = Reranker()
        self.router = ClassifyQuery()
        self._nlp = self._build_nlp_pipeline()

    # -------------------------------------------------------------------------
    # NLP Pipeline
    # -------------------------------------------------------------------------

    def _build_nlp_pipeline(self):
        """Build spaCy pipeline with EntityRuler + NER (lg model)"""
        nlp = spacy.load("en_core_web_lg")

        # EntityRuler runs BEFORE NER — catches what the model misses
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([
            # Standard names after prepositions: "reports to Marcus Webb"
            {
                "label": "PERSON",
                "pattern": [
                    {"LOWER": {"IN": ["to", "of", "by", "for"]}},
                    {"IS_TITLE": True},             # First name
                    {"IS_TITLE": True, "OP": "?"}   # Last name (optional)
                ]
            },
            # Hyphenated surnames: "Fatima Al-Hassan"
            {
                "label": "PERSON",
                "pattern": [
                    {"LOWER": {"IN": ["to", "of", "by", "for"]}},
                    {"IS_TITLE": True},
                    {"TEXT": {"REGEX": r"[A-Z][a-z]+-[A-Z][a-z]+"}}
                ]
            },
        ])

        return nlp

    # -------------------------------------------------------------------------
    # Extraction Helpers
    # -------------------------------------------------------------------------

    def _extract_ticket_id(self, query: str) -> str | None:
        match = re.search(r"TECH-\d+", query.upper())
        return match.group(0) if match else None

    def _extract_name(self, query: str) -> str | None:
        """
        NER-based name extraction using spaCy lg + EntityRuler.
        Normalises to title case first so lowercase input still matches.
        """
        # Normalize: "managed by aisha patel" → "Managed By Aisha Patel"
        normalized = query.title()

        doc = self._nlp(normalized)
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

        if persons:
            name = persons[-1]

            # Strip any leading preposition the ruler may have captured
            for prep in ("to ", "of ", "by ", "for "):
                if name.lower().startswith(prep):
                    name = name[len(prep):]
                    break

            print(f"[DEBUG] spaCy extracted name: '{name}'")
            return name

        print(f"[DEBUG] No name found in: '{query}'")
        return None

    # -------------------------------------------------------------------------
    # Structured Path
    # -------------------------------------------------------------------------

    def structured_path(self, query: str) -> dict:
        q = query.lower()

        # --- Ticket ID lookup ---
        if "tech-" in q:
            ticket_id = self._extract_ticket_id(query)
            data = self.structured_store.get_ticket_by_id(ticket_id)
            if data.empty:
                return {
                    "type": "structured",
                    "query_intent": "ticket_lookup",
                    "data": f"No ticket with ID '{ticket_id}' found."
                }
            return {
                "type": "structured",
                "query_intent": "ticket_lookup",
                "data": data.to_dict(orient="records")
            }

        # --- Direct reports ---
        if any(phrase in q for phrase in ["reports to", "direct reports", "who reports", "managed by", "list of employees"]):
            name = self._extract_name(query)
            print(f"[DEBUG] Direct reports query — Name: '{name}'")
            if not name:
                return {"type": "structured", "query_intent": "direct_reports", "data": "Could not identify name."}
            data = self.structured_store.get_direct_reports(name)
            if data.empty:
                return {"type": "structured", "query_intent": "direct_reports", "data": f"No direct reports found for '{name}'."}
            return {
                "type": "structured",
                "query_intent": "direct_reports",
                "data": data.to_dict(orient="records")
            }

        # --- Manager lookup ---
        if any(phrase in q for phrase in ["who manages", "manages", "manager of"]):
            name = self._extract_name(query)
            print(f"[DEBUG] Manager lookup query — Name: '{name}'")
            if not name:
                return {"type": "structured", "query_intent": "manager_lookup", "data": "Could not identify name."}
            employee = self.structured_store.get_employee_by_name(name)
            if employee.empty:
                return {"type": "structured", "query_intent": "manager_lookup", "data": f"Employee '{name}' not found."}
            manager_id = employee.iloc[0].get("manager")
            if not manager_id:
                return {"type": "structured", "query_intent": "manager_lookup", "data": f"{name} is a top-level executive."}
            manager = self.structured_store.employees[
                self.structured_store.employees["employee_id"] == manager_id
            ]
            return {
                "type": "structured",
                "query_intent": "manager_lookup",
                "data": manager.to_dict(orient="records")
            }

        # --- Blocked tickets ---
        if "blocked" in q:
            data = self.structured_store.get_blocked_tickets()
            if not data.empty:
                trimmed = data[["ticket_id", "title", "priority", "assignee", "team"]].to_dict(orient="records")
                return {"type": "structured", "query_intent": "blocked_tickets", "data": trimmed}

        # --- Critical / High priority ---
        if "critical" in q or "high priority" in q:
            priority = "critical" if "critical" in q else "high"
            data = self.structured_store.get_tickets_by_priority(priority)
            if not data.empty:
                return {"type": "structured", "query_intent": "priority_filter", "data": data.to_dict(orient="records")}

        # --- Team members ---
        if any(phrase in q for phrase in ["team", "team member", "members of"]):
            for team in ["engineering", "data", "product", "hr", "devops"]:
                if team in q:
                    data = self.structured_store.get_team_members(team)
                    if not data.empty:
                        return {"type": "structured", "query_intent": "team_lookup", "data": data.to_dict(orient="records")}

        # --- General employee lookup ---
        name = self._extract_name(query)
        if name:
            data = self.structured_store.get_employee_by_name(name)
            if not data.empty:
                return {"type": "structured", "query_intent": "employee_lookup", "data": data.to_dict(orient="records")}

        return {"type": "structured", "query_intent": "unknown", "data": "Could not match query to any structured pattern."}

    # -------------------------------------------------------------------------
    # Hybrid Retrieval
    # -------------------------------------------------------------------------

    def hybrid_retrieve(self, query: str) -> list:
        vector_results = self.vector_retriever.semanticSearch(query, top_k=15)
        bm25_results = self.bm25_retriever.search(query)

        fused_scores = {}
        k = 60

        for rank, r in enumerate(vector_results):
            key = r["document"]
            if key not in fused_scores:
                fused_scores[key] = {"data": r, "score": 0}
            fused_scores[key]["score"] += 1 / (k + rank + 1)

        for rank, r in enumerate(bm25_results):
            key = r["document"]
            if key not in fused_scores:
                fused_scores[key] = {"data": r, "score": 0}
            fused_scores[key]["score"] += 1 / (k + rank + 1)

        fused = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [f["data"] for f in fused[:20]]

    def retrieve_and_rerank(self, query: str) -> list:
        results = self.hybrid_retrieve(query)
        if not results:
            return []

        ticket = self._extract_ticket_id(query)
        if ticket:
            for r in results:
                if ticket in r.get("document", ""):
                    r["score"] = r.get("score", 0) + 1

        reranked = self.reranker.rerank(query, results)
        return reranked[:5]

    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------

    def _format_structured_result(self, intent: str, data, query: str) -> str:
        """
        Format structured data into clean markdown.
        Falls back to the LLM generator for intents without a dedicated formatter.
        """

        # --- Direct reports ---
        if intent == "direct_reports" and isinstance(data, list):
            lines = ["**Direct Reports:**\n"]
            for emp in data:
                lines.append(f"- **{emp.get('name')}** ({emp.get('employee_id')})")
                lines.append(f"  Role: {emp.get('role')}")
                lines.append(f"  Team: {emp.get('team')} | Email: {emp.get('email')}\n")
            return "\n".join(lines)

        # --- Manager lookup ---
        if intent == "manager_lookup" and isinstance(data, list):
            emp = data[0]
            return (
                f"**Manager Information:**\n\n"
                f"- **{emp.get('name')}** ({emp.get('employee_id')})\n"
                f"  Role: {emp.get('role')}\n"
                f"  Team: {emp.get('team')} | Email: {emp.get('email')}"
            )

        # --- Team lookup ---
        if intent == "team_lookup" and isinstance(data, list):
            lines = ["**Team Members:**\n"]
            for emp in data:
                lines.append(f"- **{emp.get('name')}** — {emp.get('role')}")
            return "\n".join(lines)

        # --- Employee lookup ---
        if intent == "employee_lookup" and isinstance(data, list):
            emp = data[0]
            return (
                f"**Employee Details:**\n\n"
                f"- **Name:** {emp.get('name')}\n"
                f"- **ID:** {emp.get('employee_id')}\n"
                f"- **Role:** {emp.get('role')}\n"
                f"- **Team:** {emp.get('team')}\n"
                f"- **Email:** {emp.get('email')}"
            )

        # --- Ticket lookup (single) ---
        if intent == "ticket_lookup" and isinstance(data, list):
            t = data[0]
            deps = t.get("dependencies") or "None"
            return (
                f"**{t.get('ticket_id')}** — {t.get('title')}\n\n"
                f"- **Status:** {t.get('status')}\n"
                f"- **Priority:** {t.get('priority')}\n"
                f"- **Assignee:** {t.get('assignee')} ({t.get('team')})\n"
                f"- **Product:** {t.get('related_product')}\n"
                f"- **Dependencies:** {deps}\n\n"
                f"{t.get('description', '')}"
            )

        # --- Blocked / priority filter (multiple tickets) — route through LLM ---
        if intent in ("blocked_tickets", "priority_filter") and isinstance(data, list):
            context = [
                f"Ticket {t.get('ticket_id')}: {t.get('title')} | "
                f"Status: {t.get('status')} | Priority: {t.get('priority')} | "
                f"Assignee: {t.get('assignee')} | Team: {t.get('team')}"
                for t in data
            ]
            result = self.generator.generate(query, context)
            return result.get("answer", "")

        # --- Generic fallback: route through LLM instead of dumping raw dicts ---
        if isinstance(data, list):
            context = [str(item) for item in data]
            result = self.generator.generate(query, context)
            return result.get("answer", "")

        # Plain string message (errors, not-found, etc.)
        return str(data)

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def handle_query(self, query: str, format_output: bool = True) -> dict:
        query_type = self.router.classify_query(query)
        print(f"[Router] Query classified as: {query_type}")

        if query_type == "structured":
            result = self.structured_path(query)
            intent = result.get("query_intent")
            data = result.get("data")

            if format_output:
                answer = self._format_structured_result(intent, data, query)
                return {"answer": answer}

            return result

        # --- Semantic / Hybrid path ---
        results = self.retrieve_and_rerank(query)
        if not results:
            return {"answer": "No relevant documents found."}

        context = [
            f"[Source: {r['metadata'].get('source')} | Section: {r['metadata'].get('section')}]\n{r['document']}"
            for r in results
        ]
        return self.generator.generate(query, context)