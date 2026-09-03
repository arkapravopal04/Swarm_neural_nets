import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class Problem_Phaser:
    """
    Problem_Phaser: An expert intent-distillation and complexity-estimation engine.
    Parses unstructured user requests into distinct, actionable components (Goals, 
    Contexts, Constraints, Domains) and computes an accurate execution budget 
    based on cognitive load, technical depth, and semantic clarity.
    """
    
    DOMAIN_MULTIPLIERS = {
        "Theoretical Mathematics": 2.0,
        "Aerospace & Automation": 1.9,
        "Electrical & Computer Engineering": 1.8,
        "Computer Science": 1.7,
        "Mechanical Engineering": 1.6,
        "Chemical Engineering & Materials": 1.6,
        "Finance & Quantitative Analysis": 1.5,
        "Software Engineering": 1.5,
        "Biomedical & Life Sciences": 1.4,
        "Data Engineering": 1.3,
        "Legal & Compliance Analysis": 1.3,
        "Professional Communications": 1.0,
        "General Discourse": 1.0,
    }

    # Constraint scoring variables
    CONSTRAINT_BASE = 1.0
    CONSTRAINT_COEF = 0.5
    CONSTRAINT_CAP = 2.5

    # Semantic gap variables
    SEMANTIC_BASE = 1.0
    SEMANTIC_COEF = 0.8

    def __init__(self, model, tokeniser, embed_model=None):
        self.llm = model
        self.tokeniser = tokeniser

        # FIX: previously always constructed its own SentenceTransformer here,
        # and memory_state.py's MemoryStore did the same independently -- two
        # separate loads of the same weights, and (more importantly) any
        # embedding Judge.semantic_check compares against needed to come from
        # the SAME embedder instance as whatever it's being compared to for
        # cosine similarity to be meaningful. main.py now constructs one
        # SentenceTransformer and injects it here and into MemoryStore.
        self.embed_model = embed_model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        
        # Safely determine device; fallback to cpu if parameters are unexposed
        try:
            self.device = next(self.llm.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sanitize_generation(self, text):
        """Removes LLM rambling by aggressively stopping at known conversational markers."""
        # Removed "Output:" from stop markers so it doesn't clip our desired responses.
        stop_markers = [
            "\nInput:", "\nAnswer:", "\nEXAMPLES", "\nGIVEN TEXT", 
            "\n\n", "Here is the", "Certainly!", "Sure,"
        ]
        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def _clean_requirements(self, raw_reqs_str):
        """Safely parses bulleted/comma-separated strings into lists, bypassing false positives."""
        raw_reqs_str = raw_reqs_str.strip()
        lowered_str = raw_reqs_str.lower()
        
        # Comprehensive 'None' checking
        exact_none_matches = ["none", "none.", "n/a", "no explicit constraints", "none stated"]
        if not raw_reqs_str or lowered_str in exact_none_matches:
            return []
            
        lines = [line.strip() for line in raw_reqs_str.split('\n') if line.strip()]
        cleaned_items = []
        
        # Regex to strip multiple bullet formats and hallucinated markdown bolding
        for line in lines:
            # Skip hallucinated preambles
            if "here are" in line.lower() or "requirements:" in line.lower():
                continue
                
            # Strip dashes, asterisks, numbers
            cleaned_line = re.sub(r'^([-\*\•]\s*|\d+\.\s*)', '', line).strip()
            # Strip bold tags if model tries to bold the start of a bullet
            cleaned_line = re.sub(r'^\*\*(.*?)\*\*:\s*', r'\1: ', cleaned_line).strip()
            
            if cleaned_line and cleaned_line.lower() not in exact_none_matches:
                cleaned_items.append(cleaned_line)
                
        # If model outputs a single comma-separated line instead of bullets
        if len(cleaned_items) == 1 and "," in cleaned_items[0]:
            comma_split = [item.strip() for item in cleaned_items[0].split(',') if item.strip()]
            if len(comma_split) > 1:
                return comma_split
                
        return cleaned_items

    def _cosine_sim(self, vec_a, vec_b):
        """Safe, pure-NumPy cosine similarity calculation preventing PyTorch tensor mismatch issues."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def _get_goal_prompt(self, raw_text):
        """Extracts the singular core action statement from the user's prompt."""
        goal_prompt = f"""You are an expert systems architect specializing in intent distillation.
TASK: Extract the single core objective from the user's request.
RULES: 
1. Output EXACTLY ONE imperative sentence.
2. No preambles, conversational filler, or explanations.

EXAMPLE:
Input: "I have a CSV of sales data, can you write me a script to plot monthly revenue trends?"
Output: Generate a script to visualize monthly revenue trends from sales data.

GIVEN TEXT: 
<user_input>
{raw_text}
</user_input>

Output:"""

        with torch.no_grad():
            inputs = self.tokeniser(goal_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            outputs = self.llm.generate(
                **inputs, max_new_tokens=100, min_new_tokens=5, do_sample=False, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            goal_sentence = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            goal_sentence = self._sanitize_generation(goal_sentence)
            goal_vector = self.embed_model.encode(goal_sentence, convert_to_numpy=True)

        return goal_sentence, goal_vector

    def _get_background_info(self, raw_text):
        """Extracts situational facts and dependencies, or firmly returns NONE."""
        background_info = f"""You are a senior context-extraction expert.
TASK: Extract foundational situational facts, existing infrastructure, or current state dependencies from the user's input.
RULES: 
1. Output exactly ONE clear sentence describing the existing context.
2. If there is absolutely no background context or existing state mentioned, output exactly: NONE.

EXAMPLES:
Input: "Using my existing AWS RDS Postgres database, create a query to find duplicates."
Output: The user is operating with an existing AWS RDS PostgreSQL database.

Input: "I have a CSV file with columns 'Date' and 'Amount'. Write a script."
Output: The user currently has a dataset in CSV format with 'Date' and 'Amount' columns.

Input: "Can you write a short sci-fi story?"
Output: NONE

GIVEN TEXT: 
<user_input>
{raw_text}
</user_input>

Output:"""

        with torch.no_grad():
            inputs = self.tokeniser(background_info, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            outputs = self.llm.generate(
                **inputs, max_new_tokens=100, min_new_tokens=2, do_sample=False, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            context_sentence = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            context_sentence = self._sanitize_generation(context_sentence)

            # Tighter check to prevent false 'NONE' positives
            cleaned_context = context_sentence.lower().strip()
            exact_none_matches = ["none", "none.", "n/a", "no context"]
            
            if not cleaned_context or cleaned_context in exact_none_matches:
                return "NONE", np.zeros(self.embed_dim)

            context_vector = self.embed_model.encode(context_sentence, convert_to_numpy=True)

        return context_sentence, context_vector

    def _get_requirement(self, raw_text):
        """Extracts technical boundaries, constraints, and requirements as distinct vectors."""
        requirement_prompt = f"""You are a strict constraints-extraction engine.
TASK: Extract all explicit technical boundaries, rules, performance targets, and formatting demands.

RULES:
1. Output ONLY a Markdown bulleted list using the '-' character.
2. No introductory text. No concluding text.
3. Do not invent constraints. Stick strictly to the text.
4. If no explicit constraints exist, output exactly: NONE

EXAMPLES:
Input: "Build a web scraper in Python. It must use BeautifulSoup and run under 5 seconds. Don't use Selenium."
Output:
- Must be written in Python.
- Must use BeautifulSoup framework.
- Execution time must be under 5 seconds.
- Selenium is strictly prohibited.

Input: "Explain the theory of relativity."
Output:
NONE

GIVEN TEXT: 
<user_input>
{raw_text}
</user_input>

Output:
"""

        with torch.no_grad():
            inputs = self.tokeniser(requirement_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            outputs = self.llm.generate(
                **inputs, max_new_tokens=120, min_new_tokens=2, do_sample=False, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            req_output = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            req_output = self._sanitize_generation(req_output)

        requirements_list = self._clean_requirements(req_output)
        
        vectored_reqs = []
        if requirements_list:
            vectored_reqs = self.embed_model.encode(requirements_list, convert_to_numpy=True)

        return requirements_list, vectored_reqs

    def _get_domain(self, raw_text):
        """Classifies the prompt into an exact taxonomy tier, ensuring formatting constraints."""
        domain_prompt = f"""You are an elite academic ontology engine.
TASK: Classify the input text into a definitive domain taxonomy.
RULE: Output ONLY the classification in this EXACT format: Macro-Discipline > Niche Specialty (Focus: comma-separated concepts)

EXAMPLES:
Input: "Write a script to simulate drone rotor aerodynamics."
Output: Aerospace & Automation > Aerodynamics (Focus: drone, rotors, simulation)

Input: "Help me write a cold email for a marketing job."
Output: Professional Communications > Networking (Focus: cold email, marketing, job search)

GIVEN TEXT: 
<user_input>
{raw_text}
</user_input>

Output:"""

        with torch.no_grad():
            inputs = self.tokeniser(domain_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            outputs = self.llm.generate(
                **inputs, max_new_tokens=60, min_new_tokens=5, do_sample=False, 
                pad_token_id=self.tokeniser.eos_token_id
            )
            domain_str = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            
            domain_str = re.sub(r'[`"\'*]', '', domain_str).strip()
            domain_str = domain_str.split('\n')[0].strip()

            has_separator = " > " in domain_str
            has_focus = "(focus:" in domain_str.lower()
            
            is_placeholder = "<" in domain_str or "macro-discipline" in domain_str.lower()
            
            if not (domain_str and has_separator and has_focus) or is_placeholder:
                domain_str = self._recover_domain_from_malformed(domain_str, raw_text)
                
            domain_vector = self.embed_model.encode(domain_str, convert_to_numpy=True)

        return domain_str, domain_vector

    def _recover_domain_from_malformed(self, malformed_str, raw_text=""):
        """Recovers discipline base if the LLM breaks the strict taxonomy format or echoes placeholders."""
        lowered_malformed = malformed_str.lower() if malformed_str else ""
        
        for key in sorted(self.DOMAIN_MULTIPLIERS.keys(), key=len, reverse=True):
            if key.lower() in lowered_malformed:
                return f"{key} > Recovered (Focus: {malformed_str[:50]})"
                
        lowered_raw = raw_text.lower()
        keyword_heuristics = {
            "Aerospace & Automation": ["aerospace", "drone", "vtol", "flight", "aircraft", "uav", "kalman"],
            "Theoretical Mathematics": ["lyapunov", "theorem", "topology", "manifold", "calculus"],
            "Electrical & Computer Engineering": ["sensor fusion", "circuit", "pcb", "embedded", "microcontroller"],
            "Mechanical Engineering": ["kinematics", "thermodynamics", "structural", "cad"],
            "Computer Science": ["algorithm", "database", "api", "backend", "docker", "kubernetes"],
            "Finance & Quantitative Analysis": ["quantitative", "finance", "trading", "market", "portfolio"]
        }
        
        for domain, keywords in keyword_heuristics.items():
            if any(kw in lowered_raw for kw in keywords):
                return f"{domain} > Inferred Context (Focus: {keywords[0]} heuristic)"
                
        return "General Discourse > Unstructured Inquiry (Focus: Everyday Conversational Knowledge)"

    def _estimate_by_constraints(self, spec):
        """Scales difficulty based on constraints using a diminishing returns (sqrt) curve."""
        num_reqs = len(spec["requirement"])
        constraint_score = self.CONSTRAINT_BASE + self.CONSTRAINT_COEF * np.sqrt(num_reqs)
        return min(constraint_score, self.CONSTRAINT_CAP)

    def _estimate_by_domain(self, spec):
        """Fetches the multiplier for the identified academic/professional discipline."""
        domain_str = spec.get("domain", "General Discourse")
        macro_discipline = domain_str.split(">")[0].strip()

        if macro_discipline in self.DOMAIN_MULTIPLIERS:
            return self.DOMAIN_MULTIPLIERS[macro_discipline]

        for key, multiplier in self.DOMAIN_MULTIPLIERS.items():
            if key.lower() in domain_str.lower():
                return multiplier
        return 1.0

    def _estimate_by_semantic_gap(self, spec):
        """Penalizes score if goal and context are highly disconnected."""
        if np.all(spec["context_vector"] == 0):
            return 1.0 

        similarity = self._cosine_sim(spec["goal_vector"], spec["context_vector"])
        similarity = max(0.0, min(1.0, similarity)) 

        return self.SEMANTIC_BASE + self.SEMANTIC_COEF * (1.0 - similarity)

    def estimate_complexity(self, spec):
        """
        Computes the final tier and accurate operational budget.
        Replaces rigid mathematical formulas with piecewise interpolation to guarantee 
        budgets accurately hit their intended scale based on tier boundaries.
        """
        base_mult = self._estimate_by_domain(spec)
        constraint_wt = self._estimate_by_constraints(spec)
        semantic_gap = self._estimate_by_semantic_gap(spec)

        raw_score = base_mult * constraint_wt * semantic_gap
        
        if raw_score <= 2.0:
            tier = "S"
        elif raw_score <= 3.5:
            tier = "M"
        elif raw_score <= 6.0:
            tier = "L"
        else:
            tier = "XL"

        score_points = [1.0, 2.0, 3.5, 6.0, 9.0]
        budget_points = [100, 300, 800, 1500, 3000]
        
        clamped_score = max(1.0, min(raw_score, 9.0))
        budget = int(np.interp(clamped_score, score_points, budget_points))

        spec.update({
            "complexity_score": round(raw_score, 2),
            "colony_tier": tier,
            "colony_budget": budget,
            "complexity_breakdown": {
                "domain_multiplier": round(base_mult, 2),
                "constraint_weight": round(constraint_wt, 2),
                "semantic_gap": round(semantic_gap, 2),
                "raw_score": round(raw_score, 2),
            }
        })

        print(
            f"--> Assigned Tier: {tier} (Budget: {budget}) "
            f"[domain={base_mult:.2f}x constraints={constraint_wt:.2f}x "
            f"semantic_gap={semantic_gap:.2f}x score={raw_score:.2f}]\n"
        )
        return spec

    def parse_problem(self, raw_text):
        """Orchestrates extraction of all structural elements from unstructured text."""
        if not raw_text or not raw_text.strip():
            print("Empty input detected. Returning minimal specification.")
            return {
                "raw_text": raw_text,
                "goal": "None", "goal_vector": np.zeros(self.embed_dim),
                "context": "NONE", "context_vector": np.zeros(self.embed_dim),
                "requirement": [], "requirement_vectors": [],
                "domain": "General Discourse > Unstructured Inquiry (Focus: None)",
                "domain_vector": np.zeros(self.embed_dim),
            }

        max_chars = 3000
        if len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars] + "\n... [TRUNCATED]"

        try:
            goal_sentence, goal_vector = self._get_goal_prompt(raw_text)
            context_sentence, context_vector = self._get_background_info(raw_text)
            requirement_list, vectored_reqs = self._get_requirement(raw_text)
            domain_str, domain_vector = self._get_domain(raw_text)

            return {
                "raw_text": raw_text,
                "goal": goal_sentence,
                "goal_vector": goal_vector,
                "context": context_sentence,
                "context_vector": context_vector,
                "requirement": requirement_list,
                "requirement_vectors": vectored_reqs,
                "domain": domain_str,
                "domain_vector": domain_vector,
            }
        except Exception as e:
            print(f"Error during problem parsing: {e}")
            return {
                "raw_text": raw_text,
                "goal": raw_text,
                "requirement": [],
                "context_vector": np.zeros(self.embed_dim),
                "goal_vector": np.zeros(self.embed_dim),
                "domain": "General Discourse",
                "colony_budget": 100,
            }

    def run_phaser(self):
        """Entry point for the terminal interactive session."""
        prompt = input("What can we help you with today?\n")
        
        if not prompt.strip():
            print("Empty input detected. Aborting sequence.")
            return None
            
        spec = self.parse_problem(prompt)
        spec = self.estimate_complexity(spec)
        return spec