import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class Problem_Phaser:
    def __init__(self, model, tokeniser):
        self.llm = model
        self.tokeniser = tokeniser
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        self.device = next(self.llm.parameters()).device

    def _sanitize_generation(self, text):
        # Added more aggressive stopping strings to prevent LLM rambling
        for marker in ["\nInput:", "\nAnswer:", "\nEXAMPLES", "\nGIVEN TEXT", "\nOutput:", "\n\n", "Output:"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def _clean_requirements(self, raw_reqs_str):
        raw_reqs_str = raw_reqs_str.strip()
        
        # Robust "None" checking: Catch variants like "None.", "None stated", "NONE"
        if "none" in raw_reqs_str.lower()[:15] or raw_reqs_str == "":
            return []
            
        lines = [line.strip() for line in raw_reqs_str.split('\n') if line.strip()]
        cleaned_items = []
        for line in lines:
            cleaned_line = re.sub(r'^([-\*\•]|\d+\.)\s*', '', line).strip()
            if cleaned_line:
                cleaned_items.append(cleaned_line)
                
        if len(cleaned_items) == 1 and "," in cleaned_items[0]:
            comma_split = [item.strip() for item in cleaned_items[0].split(',') if item.strip()]
            if len(comma_split) > 1:
                return comma_split
                
        return cleaned_items

    def _get_goal_prompt(self, raw_text):
        goal = f"""You are an expert systems architect specializing in intent distillation and task decomposition.
    ACTION: Identify the singular, core functional objective the user intends to achieve and state it as one precise, imperative declarative sentence.

    RULES:
    - Focus strictly on the primary action to be performed. Strip away all constraints, formatting demands, background context, and situational dependencies.
    - If a request is multi-faceted, distill it into the primary overarching outcome.
    - Do NOT include preambles, introductory phrases, or fluff.
    - Output must be exactly one sentence, no line breaks, no markdown, no quotes.

    EXAMPLES:
    - Input: "I have a CSV of sales data, can you write me a script to plot monthly revenue trends? Please use matplotlib and keep it under 50 lines."
      Output: Generate a script to visualize monthly revenue trends from sales data.
    - Input: "Refactor our monolithic Java backend to support async operations, but keep the existing Oracle DB connection logic intact."
      Output: Refactor a monolithic Java backend to support asynchronous operations while preserving legacy Oracle database connectivity.
    - Input: "How do I convert Celsius to Fahrenheit?"
      Output: Perform a unit conversion from Celsius to Fahrenheit.

    GIVEN TEXT:
    '{raw_text}'

    Goal sentence:"""

        with torch.no_grad():
            inputs = self.tokeniser(goal, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            # Slashed max_new_tokens to force single-sentence brevity
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens= 50, 
                temperature=0.1,
                do_sample=True, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            generated_ids = outputs[0][prompt_length:]
            goal_sentence = self.tokeniser.decode(generated_ids, skip_special_tokens=True).strip()
            goal_sentence = self._sanitize_generation(goal_sentence)
            goal_vector = self.embed_model.encode(goal_sentence, convert_to_numpy=True)

        return goal_sentence, goal_vector

    def _get_background_info(self, raw_text):
        background_info = f"""You are a senior systems researcher and context-extraction expert.
        ACTION: Extract all foundational situational facts, existing infrastructure dependencies, user environment details, or prior knowledge states explicitly mentioned in the text.
        If the text contains no such situational context, you must output exactly the word: NONE.

        RULES:
        - Anticipate complex, professional inquiries where the user's environment or existing work is critical.
        - Filter out mere pleasantries or conversational fillers. Extract only the technical or situational facts.
        - If the information is not explicitly provided in the text, you must output NONE.
        - Output NOTHING ELSE but the word NONE if there is no context.
        - Output must be a single, dense, declarative sentence (or NONE). No preamble, no markdown.

        EXAMPLES:
        - Input: "I have a CSV of sales data, can you write me a script to plot monthly revenue trends?"
          Output: The user is working with a CSV-formatted sales dataset.
        - Input: "Refactor our monolithic Java backend to support async operations, but keep the existing Oracle DB connection logic intact as we cannot migrate the database due to strict compliance requirements."
          Output: The current system is a monolithic Java application using an Oracle database that cannot be migrated due to compliance regulations.
        - Input: "How do I convert Celsius to Fahrenheit?"
          Output: NONE

        GIVEN TEXT:
        '{raw_text}'

        Context:"""

        with torch.no_grad():
            inputs = self.tokeniser(background_info, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            # Slashed token count to prevent rambling essays on context
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            generated_ids = outputs[0][prompt_length:]
            context_sentence = self.tokeniser.decode(generated_ids, skip_special_tokens=True).strip()
            context_sentence = self._sanitize_generation(context_sentence)

            # Highly robust None detection
            if "none" in context_sentence.lower()[:15] or context_sentence == "":
                return "NONE", np.zeros(self.embed_dim)

            context_vector = self.embed_model.encode(context_sentence, convert_to_numpy=True)

        return context_sentence, context_vector

    def _get_requirement(self, raw_text):
        requirement = f"""You are a strict systems analyst and constraints-extraction engine.
        ACTION: Isolate and extract all explicit rules, architectural boundaries, performance budgets, strict formatting demands, security protocols, or technical specifications from the text.
        If the text contains absolutely no explicit constraints or requirements, you must output exactly the word: NONE.

        RULES:
        - Anticipate highly complex, multi-layered, or heavily constrained enterprise-grade inquiries.
        - Break down dense, compound constraints into distinct, granular bullet points.
        - Do NOT invent, hallucinate, or assume requirements (e.g., best practices) that are not explicitly mandated in the text.
        - Output NOTHING ELSE but the word NONE if there are no explicit demands.
        - If outputting requirements, use a simple bulleted list with no introductory text.

        EXAMPLES:
        - Input: "How do I convert Celsius to Fahrenheit?"
          Output: NONE
        - Input: "Design a real-time bidding microservice. It must handle 50k RPS with sub-10ms latency. The data layer is restricted to ScyllaDB, and the cache must be Redis. All inter-service comms must use gRPC secured with mTLS. Strict budget: max 4 CPUs per Kubernetes pod."
          Output: 
          - Must be a microservice architecture
          - Must handle 50k requests per second (RPS)
          - Must maintain sub-10ms latency
          - Data layer must use ScyllaDB
          - Cache must use Redis
          - Inter-service communication must use gRPC
          - Communication must be secured with mTLS
          - CPU usage is strictly capped at 4 CPUs per Kubernetes pod

        GIVEN TEXT:
        '{raw_text}'

        Requirements:"""

        with torch.no_grad():
            inputs = self.tokeniser(requirement, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            # Slashed to prevent the LLM from inventing fake rules
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            generated_ids = outputs[0][prompt_length:]
            requirement_sentence = self.tokeniser.decode(generated_ids, skip_special_tokens=True).strip()
            requirement_sentence = self._sanitize_generation(requirement_sentence)

        requirements_list = self._clean_requirements(requirement_sentence)
        vectored_requirements_list = requirements_list.copy()
        for i in range(len(requirements_list)):
            vectored_requirements_list[i] = self.embed_model.encode(vectored_requirements_list[i], convert_to_numpy=True)

        return requirements_list, vectored_requirements_list

    def _get_domain(self, raw_text):
        domain = f"""You are an elite academic ontology engine designed to classify inquiries ranging from foundational questions to highly specialized, multi-disciplinary, and advanced post-graduate research.

    TASK: Analyze the deepest technical, theoretical, or systemic context implied by the text and output ONLY a definitive domain classification in this exact form:
    <Macro-Discipline> > <Niche Specialty> (Focus: <comma-separated granular concepts>)

    STRICT RULES:
    - Anticipate complex, ambiguous, or highly technical prompts. Extract the core academic or professional discipline required to solve it.
    - If a query heavily crosses multiple domains, classify it under its primary operational framework.
    - Output ONLY the classification line. No explanation, no preamble. 
    - One line only, no markdown, no quotes.

    EXAMPLES:
    - Input: "How do I calculate eigenvectors using NumPy?"
      Output: Computer Science > Numerical Linear Algebra (Focus: NumPy, Matrix Decomposition)
    - Input: "How do I convert Celsius to Fahrenheit?"
      Output: Theoretical Mathematics > Arithmetic (Focus: Unit Conversion, Basic Formulas)

    GIVEN TEXT:
    '{raw_text}'

    Domain Taxonomy Output:"""

        with torch.no_grad():
            inputs = self.tokeniser(domain, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=48, # Extremely short to force the format
                temperature=0.1,
                do_sample=True, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            generated_ids = outputs[0][prompt_length:]
            domain_str = self.tokeniser.decode(generated_ids, skip_special_tokens=True).strip()
            domain_str = re.sub(r'[`"\'*]', '', domain_str).strip()
            domain_str = domain_str.split('\n')[0].strip()

            has_separator = " > " in domain_str
            has_focus_clause = "(focus:" in domain_str.lower()
            is_placeholder_echo = "[" in domain_str or "<" in domain_str or "macro discipline" in domain_str.lower()
            
            if not domain_str or is_placeholder_echo or not has_separator or not has_focus_clause:
                domain_str = "General Discourse > Unstructured Inquiry (Focus: Everyday Conversational Knowledge)"

            domain_vector = self.embed_model.encode(domain_str)

        return domain_str, domain_vector

    def parse_problem(self, raw_text):
        goal_sentence, goal_vector = self._get_goal_prompt(raw_text)
        context_sentence, context_vector = self._get_background_info(raw_text)
        requirement_list, vectored_requirement_list = self._get_requirement(raw_text)
        domain_str, domain_vector = self._get_domain(raw_text)

        spec = {
            "raw_text": raw_text,
            "goal": goal_sentence,
            "goal_vector": goal_vector,
            "context": context_sentence,
            "context_vector": context_vector,
            "requirement": requirement_list,
            "requirement_vectors": vectored_requirement_list,
            "domain": domain_str,
            "domain_vector": domain_vector,
        }
        return spec

    def _estimate_by_constraints(self, spec):
        num_reqs = len(spec["requirement"])  
        # Lowered step value and hard capped at 2.0 to prevent blowout
        constraint_score = min(1.0 + (num_reqs * 0.15), 2.0)
        return constraint_score

    def _estimate_by_domain(self, spec):
        DOMAIN_MULTIPLIERS = {
            "Theoretical Mathematics": 1.5,
            "Aerospace & Automation": 1.45,
            "Electrical & Computer Engineering": 1.4,
            "Computer Science": 1.35,
            "Mechanical Engineering": 1.3,
            "Chemical Engineering & Materials": 1.3,
            "Finance & Quantitative Analysis": 1.25,
            "Software Engineering": 1.25,
            "Biomedical & Life Sciences": 1.2,
            "Data Engineering": 1.15,
            "Legal & Compliance Analysis": 1.15,
            "Professional Communications": 1.0,
            "General Discourse": 1.0 # Lifted from 0.8 so fallbacks don't break logic
        }
        domain_str = spec.get("domain", "General Discourse") 
        macro_discipline = domain_str.split(">")[0].strip()
        
        # 1. Exact match lookup
        if macro_discipline in DOMAIN_MULTIPLIERS:
            return DOMAIN_MULTIPLIERS[macro_discipline]
            
        # 2. Fuzzy match lookup (in case LLM alters format slightly)
        for key, multiplier in DOMAIN_MULTIPLIERS.items():
            if key.lower() in domain_str.lower():
                return multiplier
                
        return 1.0

    def _estimate_by_semantic_gap(self, spec):
        # FIX: Having no context is completely normal for simple queries.
        # It should NOT heavily penalize the score. Default to neutral 1.0.
        if np.all(spec["context_vector"] == 0):
            return 1.0
            
        similarity = util.cos_sim(spec["goal_vector"], spec["context_vector"]).item()
        similarity = max(0.0, similarity)
        
        # Relevant context (1.0) -> multiplier 1.0
        # Disconnected context (0.0) -> multiplier 1.5 (max penalty)
        semantic_gap_score = 1.0 + (0.5 * (1.0 - similarity))
        return semantic_gap_score

    def estimate_complexity(self, spec):
        base_multiplier = self._estimate_by_domain(spec)
        constraint_weight = self._estimate_by_constraints(spec)
        semantic_gap = self._estimate_by_semantic_gap(spec)

        score = base_multiplier * constraint_weight * semantic_gap
        
        # print("\n--- Complexity Breakdown ---")
        # print(f"Goal: {spec['goal']}")
        # print(f"Constraints ({len(spec['requirement'])}): {constraint_weight:.2f}x")
        # print(f"Domain ({spec['domain'].split(' > ')[0]}): {base_multiplier:.2f}x")
        # print(f"Context Gap: {semantic_gap:.2f}x")
        # print(f"Final Score: {score:.2f}x")

        # Recalibrated Thresholds for the newly clamped math
        if score <= 2.2:
            tier, budget = "S", 200
        elif score <= 3.0:
            tier, budget = "M", 500
        elif score <= 4.5:
            tier, budget = "L", 1000
        else:
            tier, budget = "XL", 2000

        print(f"--> Assigned Tier: {tier} (Budget: {budget})\n")

        spec["complexity_score"] = round(score, 2)
        spec["colony_tier"] = tier
        spec["colony_budget"] = budget
        return spec

    def run_phaser(self):
        prompt = input("What can we help you with today?\n")
        spec = self.parse_problem(prompt)
        spec = self.estimate_complexity(spec)
        return spec