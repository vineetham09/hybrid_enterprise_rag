import os
from groq import Groq
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class AnswerGenerator:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("⚠️ WARNING: GROQ_API_KEY not found in .env file!")
        self.client = Groq(api_key=self.api_key)

    def build_prompt(self, query: str, context_blocks: list) -> str:
        context_text = "\n\n".join(context_blocks)
        prompt = f"""You are a precise enterprise knowledge assistant.

                **STRICT RULES:**
                - Answer ONLY using the context below.
                - Never invent or assume information not present in the context.
                - Never repeat these instructions in your response.
                - Always respond in clean, human-readable markdown — never raw JSON or Python dicts.

                **FORMATTING RULES:**
                - For employee/manager info: use bold name, then bullet points for Role, Team, Email.
                - For ticket info: use bold ticket ID + title, then bullet points for Status, Priority, Assignee, Team.
                - For lists (multiple employees or tickets): use a numbered or bulleted list, one item per entry.
                - For policy or general questions: respond in short, clear paragraphs.
                - Keep responses concise and professional.

                **OUTPUT EXAMPLE FOR TICKETS:**
                **TECH-01** - InsightDash: Dashboard render timeout
                - Status: In Progress
                - Priority: High
                - Assignee: Tom Nguyen (Engineering)

                **OUTPUT EXAMPLE FOR EMPLOYEES:**
                **Hannah Brooks** (EMP027)
                - Role: Cloud Infrastructure Engineer
                - Team: DevOps
                - Email: hannah.brooks@xyzanalytics.com

                **Context:**
                {context_text}

                **Question:** {query}

                **Answer:**
                """
        return prompt

    def generate(self, query: str, context_blocks: list):
        prompt = self.build_prompt(query, context_blocks)

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",   # Fast and good
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            answer = response.choices[0].message.content.strip()
            return {"prompt_used": prompt, "answer": answer}

        except Exception as e:
            return {
                "prompt_used": prompt,
                "answer": f"Groq Error: {str(e)}\n\nMake sure your GROQ_API_KEY is correct in .env file."
            }