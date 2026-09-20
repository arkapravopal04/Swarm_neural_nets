import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from text_utils import (
    asks_for_software,
    cut_at_code_block,
    cut_instruction_tail,
    dedupe_global_and_cap,
    dedupe_list_exact,
    final_derived_constraint,
    looks_like_source_code,
    plain_register,
    reasoning_reason,
    strip_code_fences,
)

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

    # Repetition penalty for every extraction call below. Raised from 1.15:
    # at that strength these four extractions still degenerated often
    # enough that the cleanup downstream of them (dedupe_global_and_cap on
    # the goal, dedupe_list_exact on the constraint bullets) was doing real
    # work on real runs rather than sitting there as an unused backstop.
    #
    # Safe to raise HERE specifically, unlike the other two 1.15s in the
    # system. main.py's shared judge/synthesizer closure has to stay at
    # 1.15 because deep_critique structurally repeats the words "accept"/
    # "reject", and 1.3 fragmented them into subword pieces ("ac ce pt")
    # that judge.py's verdict regex could not match -- silently discarding
    # the model's actual determination. Nothing the phaser generates has
    # that shape: a goal sentence, a context sentence, a bulleted
    # constraint list and a taxonomy string are parsed structurally (by
    # bullet, by " > ", by "(Focus:") and none of them needs a specific
    # keyword emitted more than once, so there is no token here that a
    # stronger penalty can break by discouraging its repeat.
    REPETITION_PENALTY = 1.3

    def __init__(self, model, tokeniser, embed_model=None, reword_software_vocabulary=True):
        self.llm = model
        self.tokeniser = tokeniser
        # Phaser half of the orchestrator's "reword" framing lever: plain
        # wording for code-coded words in the goal and requirements of a
        # request that never asked for software. Separate switch because the
        # phaser is constructed separately; set both off for a baseline run.
        self.reword_software_vocabulary = reword_software_vocabulary

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

    @staticmethod
    def _is_dropped_reasoning(line, path):
        """True (and logged) when a candidate constraint is model self-talk.
        Every path that filters constraints goes through here, so a
        wrongly dropped constraint always shows up in the run log."""
        reason = reasoning_reason(line)
        if reason is None:
            return False
        print(f"[Problem_Phaser] dropped {path} constraint ({reason}): {line[:120]!r}")
        return True

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
            
            # FIX: keep only the constraint, not the derivation that
            # produced it. The prompt asks for bare bullets, but the inputs
            # this runs on ask for shown reasoning, and the extractor reads
            # that register back out: "The base alloy cannot survive 1400C
            # alone, therefore an internal cooling scheme is required." The
            # derivation half is then embedded into requirement_vectors,
            # counted by _estimate_by_constraints (inflating the budget) and
            # threaded into every child agent's "Constraints you must
            # satisfy" block, where prose reads as context to continue
            # rather than as a bound to meet. A bullet that turns out to be
            # pure derivation comes back empty and is dropped.
            cleaned_line = final_derived_constraint(cleaned_line)

            if cleaned_line and cleaned_line.lower() not in exact_none_matches:
                cleaned_items.append(cleaned_line)

        # Model self-talk ("Wait, I need to check... Let me fix this.") is
        # not a bound anyone can meet, and it rides into every child's
        # constraints block -- so it is filtered out below. Filtering
        # happens AFTER the comma-split decision: filtering the single line
        # first dropped a whole comma-separated list because one of its
        # items was self-talk.

        # If model outputs a single comma-separated line instead of bullets
        if len(cleaned_items) == 1 and "," in cleaned_items[0]:
            fragments = [
                constraint
                for item in cleaned_items[0].split(',')
                if item.strip()
                for constraint in [final_derived_constraint(item)]
                if constraint
            ]
            if len(fragments) > 1:
                comma_split = [
                    c for c in fragments
                    if not self._is_dropped_reasoning(c, "comma-split")
                ]
                return dedupe_list_exact(comma_split)

        cleaned_items = [
            c for c in cleaned_items if not self._is_dropped_reasoning(c, "bullet")
        ]

        # FIX: greedy decoding can repeat an entire bullet verbatim (distinct
        # from the sentence-level dedupe used for goal/context text above --
        # here the unit of repetition is a whole list item). Duplicated
        # bullets directly inflate num_reqs in _estimate_by_constraints,
        # which drives the sqrt(num_reqs) budget multiplier -- an undeduped
        # list silently overestimates task complexity.
        return dedupe_list_exact(cleaned_items)

    def _cosine_sim(self, vec_a, vec_b):
        """Safe, pure-NumPy cosine similarity calculation preventing PyTorch tensor mismatch issues."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    # Generation-time stop strings for every phaser call.
    #
    # THIS IS THE UPSTREAM FIX; cut_instruction_tail is the band-aid behind
    # it. Run 6's goal came back as 254 characters of which the last 123
    # were "---THESE THREE QUESTIONS NEED TO BE ANSWERED IN ORDER--END OF
    # OUTPUT-- -- -- --", and that string became goal_vector.
    #
    # WHY IT HAPPENS. The adapter is a ChatML LoRA on Qwen3-4B: its
    # tokenizer_config declares eos_token "<|im_end|>" and it ships the
    # stock Qwen3 chat_template.jinja. Nothing in this codebase calls
    # apply_chat_template -- every prompt here is raw text -- so
    # "<|im_end|>" never appears in the prompt and is never a natural
    # continuation. The model therefore has no in-distribution way to stop
    # and runs to max_new_tokens every time, padding the tail with the
    # boundary marker it DOES know: all 200 records in fine_tune/train3.json
    # wrap their instruction in a bare "---" line, then the prompt body,
    # then "Your next action:", then a closing "---" line immediately
    # before the completion -- so "---"
    # is the document boundary it learned. (Serving through the chat
    # template is the better fix and is deliberately NOT done here: it moves
    # the distribution every tuning decision in this file was made against,
    # and that belongs in its own run.)
    #
    # The matched stop string is still INCLUDED in the generated text --
    # generate() stops after emitting it, not before -- so the cleanup
    # downstream is what actually removes it. The win is that generation
    # ends there instead of spending the remaining budget on 60 more tokens
    # of dashes.
    STOP_STRINGS = ["---", "END OF OUTPUT"]

    # Extra goal samples drawn when a sample is still source code after
    # fence stripping, before falling back to the user's own text.
    GOAL_CODE_RESAMPLES = 2

    def _clean_goal_text(self, goal_sentence):
        # Code block cut BEFORE _sanitize_generation: sanitize stops at
        # the first blank line, so "```python\n\nGenerate X." would
        # otherwise be cut down to the bare fence and lose the goal.
        goal_sentence = cut_at_code_block(goal_sentence)
        goal_sentence = self._sanitize_generation(goal_sentence)

        # Sampling stops the model emitting a whole function body for a
        # code-shaped input (see _get_goal_prompt), but it still sometimes
        # wraps the one sentence it does produce in a code fence -- and the
        # max_new_tokens cap usually cuts the block before its closing
        # fence, so what arrives is an unbalanced marker rather than a
        # matched pair. Strip before dedupe/encode: the fence would
        # otherwise be printed as the root task description and embedded
        # into goal_vector, the tier-2 similarity target.
        goal_sentence = strip_code_fences(goal_sentence)

        # Same-line scaffold, LAST: run 6 ended the goal with
        # "...disliked selections.---THESE THREE QUESTIONS NEED TO BE
        # ANSWERED IN ORDER--END OF OUTPUT-- -- -- --", 116 characters of
        # delimiter on a 130-character goal. _sanitize_generation above
        # cannot reach it -- every stop marker it knows begins with a
        # newline or is a conversational opener, and this arrives glued to
        # the final full stop. It matters more here than anywhere else the
        # cleanup runs: this string is embedded into goal_vector, which is
        # the reference direction PATCH 12's drift gate scores every
        # subtask against, and a reference that is half delimiter separates
        # subtasks by how much delimiter they share, not by how close to
        # the project they are.
        goal_sentence, cut_reasons = cut_instruction_tail(goal_sentence)
        if cut_reasons:
            print(f"[Problem_Phaser] goal: cut same-line scaffold "
                  f"({', '.join(cut_reasons)}): {goal_sentence[:120]!r}")
        return goal_sentence

    def _pick_goal(self, sample, raw_text):
        """
        First sample whose cleaned text is prose, not code.

        Fence stripping cannot help when the model put CODE inside the
        fence ("```python def extract(text): return 1") -- what survives is
        the code body, which would become the root task description and
        the goal_vector target. Such a sample is re-drawn (sampling is
        stochastic, a retry usually yields the sentence); if every attempt
        is code, the user's own text stands in as the goal -- verbose, but
        on-topic, which a function body is not.
        """
        for attempt in range(1 + self.GOAL_CODE_RESAMPLES):
            goal_sentence = self._clean_goal_text(sample())
            if not looks_like_source_code(goal_sentence):
                return goal_sentence
            print(
                f"[Problem_Phaser] WARNING: goal sample {attempt + 1} is source "
                f"code, not a sentence -- re-sampling: {goal_sentence[:120]!r}"
            )
        # The same cut on the fallback path. raw_text is the user's own
        # input so it does not normally carry a model-emitted delimiter --
        # but this branch is reached precisely when generation has already
        # misbehaved, and a dataset question that ends in its own
        # "END OF OUTPUT" would otherwise land in goal_vector untouched.
        fallback = strip_code_fences(" ".join(raw_text.split()))
        fallback, fallback_reasons = cut_instruction_tail(fallback)
        if fallback_reasons:
            print(f"[Problem_Phaser] goal fallback: cut same-line scaffold "
                  f"({', '.join(fallback_reasons)})")
        print(
            f"[Problem_Phaser] WARNING: every goal sample was source code; "
            f"using the user's input as the goal: {fallback[:120]!r}"
        )
        return fallback

    def _plain_wording(self, text, raw_text, what):
        """text with code-coded words reworded, when the user's own request
        (raw_text) never asked for software and the lever is on. On a
        software request "implement" and "algorithm" mean what they say."""
        if not self.reword_software_vocabulary or not text or asks_for_software(raw_text):
            return text
        plain, replaced = plain_register(text)
        if replaced:
            print(f"[Problem_Phaser] {what} reworded out of software "
                  f"vocabulary {replaced}: {plain!r}")
        return plain

    def _get_goal_prompt(self, raw_text):
        """Extracts the singular core action statement from the user's prompt."""
        # Two examples, one software and one not. The only example used to
        # turn a request into "Generate a script to...", and this goal is
        # the [Project goal] line in every agent's ghost context -- one
        # code-flavoured sentence here sets the register for the whole
        # colony. "systems architect" went for the same reason.
        goal_prompt = f"""You are an expert at intent distillation.
TASK: Extract the single core objective from the user's request.
RULES: 
1. Output EXACTLY ONE imperative sentence.
2. No preambles, conversational filler, or explanations.
3. Keep the kind of result the user asked for, in the user's own plain words. Only mention a script or a program if the user asked for one.

EXAMPLES:
Input: "I have a CSV of sales data, can you write me a script to plot monthly revenue trends?"
Output: Generate a script to visualize monthly revenue trends from sales data.

Input: "Our hiking group argues every month about which trail to do. How should we decide?"
Output: Decide on a fair way for the hiking group to choose each month's trail.

GIVEN TEXT: 
<user_input>
{raw_text}
</user_input>

Output:"""

        with torch.no_grad():
            inputs = self.tokeniser(goal_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            # Sampled, not greedy. Greedy decoding on this prompt drops
            # into the highest-probability continuation of an instruction
            # block, which for a code-shaped input is the code itself --
            # this call emitted "def extract_intent(text)" as the colony's
            # GOAL, i.e. the root task description and the tier-2 similarity
            # target every child is scored against.
            def sample():
                outputs = self.llm.generate(
                    **inputs, max_new_tokens=100, min_new_tokens=5,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=self.tokeniser.eos_token_id,
                    stop_strings=self.STOP_STRINGS, tokenizer=self.tokeniser,
                    repetition_penalty=self.REPETITION_PENALTY, no_repeat_ngram_size=4,
                )
                return self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()

            goal_sentence = self._pick_goal(sample, raw_text)

            # FIX: greedy decoding here occasionally degenerates into a
            # repeated clause/sentence, which then silently pushed real
            # content past all-MiniLM-L6-v2's 256-wordpiece truncation
            # window at encode() time below. Dedupe (non-adjacent, since a
            # short goal sentence can interleave a repeat with other
            # content) and cap BEFORE encoding, so goal_sentence (the root
            # TaskNode.description the judge prints) and goal_vector (the
            # tier-2 target) are guaranteed to agree on the same string.
            cleaned_goal = dedupe_global_and_cap(goal_sentence, max_chars=400)
            if goal_sentence and len(cleaned_goal) < 0.7 * len(goal_sentence):
                print(
                    f"[Problem_Phaser] WARNING: goal generation looked degenerate -- "
                    f"dedupe removed {100 * (1 - len(cleaned_goal) / len(goal_sentence)):.0f}% "
                    f"of the text ({len(goal_sentence)} -> {len(cleaned_goal)} chars)."
                )
            goal_sentence = cleaned_goal

            # Reworded BEFORE encoding, so goal_vector and the root task
            # description still agree (see the dedupe note above). Only when
            # the user's own text never asked for software: there,
            # "implement" and "algorithm" mean exactly what they say.
            goal_sentence = self._plain_wording(goal_sentence, raw_text, "goal")
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

Input: "Our team of six shares one car for site visits. Plan next week's visits."
Output: The user's team has six people sharing a single car for site visits.

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
                pad_token_id=self.tokeniser.eos_token_id,
                stop_strings=self.STOP_STRINGS, tokenizer=self.tokeniser,
                repetition_penalty=self.REPETITION_PENALTY, no_repeat_ngram_size=4,
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
TASK: Extract all explicit boundaries, rules, limits, targets, and formatting demands.

RULES:
1. Output ONLY a Markdown bulleted list using the '-' character.
2. No introductory text. No concluding text.
3. Do not invent constraints. Stick strictly to the text.
4. Each bullet states ONLY the constraint itself. Never write the
   reasoning behind it -- no 'because', 'since', 'therefore', 'so'.
5. If no explicit constraints exist, output exactly: NONE

EXAMPLES:
Input: "Build a web scraper in Python. It must use BeautifulSoup and run under 5 seconds. Don't use Selenium."
Output:
- Must be written in Python.
- Must use BeautifulSoup framework.
- Execution time must be under 5 seconds.
- Selenium is strictly prohibited.

Input: "Plan a dinner for 8 guests. Keep it under $200, two guests are vegetarian, and no nuts."
Output:
- Must serve 8 guests.
- Total cost must be under $200.
- Must include vegetarian options for 2 guests.
- Nuts are prohibited.

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
                pad_token_id=self.tokeniser.eos_token_id,
                stop_strings=self.STOP_STRINGS, tokenizer=self.tokeniser,
                repetition_penalty=self.REPETITION_PENALTY, no_repeat_ngram_size=4,
            )
            req_output = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            req_output = self._sanitize_generation(req_output)

        requirements_list = self._clean_requirements(req_output)
        # Requirements reach every agent's "Constraints you must satisfy"
        # block, so an extractor that writes "Must implement a fair
        # resolution algorithm" would re-prime every child regardless of how
        # its own task is worded. Reworded before encoding, same as the goal.
        requirements_list = [
            self._plain_wording(req, raw_text, "constraint") for req in requirements_list
        ]

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
                pad_token_id=self.tokeniser.eos_token_id,
                stop_strings=self.STOP_STRINGS, tokenizer=self.tokeniser,
                repetition_penalty=self.REPETITION_PENALTY, no_repeat_ngram_size=4,
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