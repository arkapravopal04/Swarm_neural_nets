'''
the actual agent sits here
the agent can
think(): runs latent forward passes, updates hidden state, returns nothing
decide(): decodes once, returns one action token
execute(): reads the action, calls the right method


role_caps = {
    "decomposer": 512, # root agent thinks longer
    "executor": 256, # workers think less
    "verifier": 128, # verifier just checks
} - maybe double them
'''


from event_queue import Event , Messenger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json


class Agent:
    def __init__(self ,tokeniser , model,message : Messenger,task : str, agent_id: str , role :str ,parent_id: str = None,failure_reason: str = None, KV_Cache = None , last_hidden_state = None ):
        self.agent_id = agent_id
        self.role = role
        self.task = task
        self.parent_id = parent_id
        self.KV_Cache = KV_Cache
        self.last_hidden_state = last_hidden_state
        self.message = message
        self.fail_reason = failure_reason
        self.tokeniser = tokeniser
        self.model = model
        self.ghost_context = None
        self.thought_process = ""
        self.last_token_id = None   
        # actions
        self.action_tokens = ["THINK", "SPAWN", "TOOL", "REPORT", "DIE"]
        self.think_cycle = 0


    def _get_role_cap(self):
        """Actual cap of number of tokens....add more when ever"""
        if self.role == "decomposer":
            return 512
        if self.role == "executor":
            return 256
        if self.role == "verifier":
            return 128


    def _build_prompt(self, available_roles=["decomposer", "executor", "verifier"], available_tools=[]):
        # 1. Handle conditional strings safely outside the f-string
        ghost_str = f"Ghost Context: {self.ghost_context}\n" if self.ghost_context else ""
        fail_str = f"WARNING - Previous Action Failed: {self.fail_reason}\n" if self.fail_reason else ""
        thoughts_str = f"Your Previous Thoughts:\n{self.thought_process}\n" if self.thought_process else ""
        
        # 2. Build the comprehensive prompt
        prompt = f"""You are an AI agent in a colony of agents working together to solve problems.
Your Agent ID: {self.agent_id}
Your Role: {self.role}
Your Task: {self.task}

{ghost_str}{fail_str}{thoughts_str}
Available actions:
- THINK  — continue reasoning before acting (Use this to plan your next move).
- SPAWN  — create a sub-agent to handle a sub-task. Available roles: {available_roles}.
- TOOL   — call an external tool. Available tools: {available_tools}.
- REPORT — your task is complete, submit your final result to your parent.
- DIE    — you cannot complete this task, signal failure to your parent.

OUTPUT FORMAT INSTRUCTIONS:
You must respond with EXACTLY ONE action block in the following format:

ACTION: <One of the available actions>
PAYLOAD: <Depends on the action>
- If THINK: Provide your reasoning in plain text.
- If SPAWN: Provide a JSON object: {{"role": "chosen_role", "task": "specific task definition"}}
- If TOOL: Provide a JSON object: {{"tool_name": "name", "args": {{...}}}}
- If REPORT: Provide the final answer or result in plain text.
- If DIE: Provide the reason you cannot proceed.

Your next action:"""
        
        return prompt
    

    def think(self):
        """
        1. Determine the cap
   - look up role_caps[self.agent_role], fallback to a default

2. Build the starting input
   - if this is the first think() call, tokenize self.task into input_ids
   - if self.KV_Cache already exists (continuing from before), 
     you only need the NEW input, not the whole history —
     the cache already remembers everything before

3. Loop up to the cap, one token at a time
   for step in range(max_tokens):
   
       a. run one forward pass
          - pass input_ids (or inputs_embeds if continuing latent)
          - pass past_key_values = self.KV_Cache
          - set use_cache=True
       
       b. extract from the output
          - new KV cache  → save to self.KV_Cache
          - last hidden state → save to self.last_hidden_state
          - logits → pick the next token (greedy or sampled)
       
       c. decode that one token to text
       
       d. check if it's an action word
          - if yes → stop the loop, return the action
          - if no  → this token becomes the next input_ids, go to step a

4. If loop finishes without an action word
   - safety fallback, return "THINK" or force a decide() call
        """
        max_tokens = self._get_role_cap() 
        self.think_cycle += 1

        if self.KV_Cache == None:
            inputs = self.tokeniser(self.task, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]

        else:
            if self.last_token_id is not None:
                input_ids = torch.tensor([[self.last_token_id]], device=self.model.device)
            else:
                raise ValueError("KV_Cache is present, but missing last_token_id to continue generation.")
            
# need to incase this for loop in a torch.no
        for step in range(max_tokens):
            outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.KV_Cache,
            use_cache=True,
            output_hidden_states=True
            )

            self.last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            
            self.KV_Cache = outputs.past_key_values
            logits = outputs.logits
            
            next_token_tensor = torch.argmax(logits[:, -1, :], dim=-1)
            self.last_token_id = next_token_tensor.item()
            
            token_text = self.tokeniser.decode([self.last_token_id])
            self.thought_process += token_text  
            
            action_word = token_text.strip()
            if action_word in self.action_tokens:
                return action_word
            
            input_ids = next_token_tensor.unsqueeze(0)
        
        # now loop has finished without any action_tokens....
        if self.think_cycle <= 3:
            return "THINK"
        else:
            return self.decide()

    def decide(self , available_roles = None , available_tools = None):
        if available_roles is None:
            available_roles = ["decomposer", "executor" , "verifier"]
        if available_tools is None:
            available_tools = []
        
        prompt = self._build_prompt(available_roles , available_tools)
        inputs = self.tokeniser(prompt , return_tensors = "pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 150,
                pad_token_id = self.tokeniser.eos_token_id,
                do_sample = False
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokeniser.decode(generated_tokens, skip_special_tokens=True).strip()

        self.thought_process += f"\n{generated_text}\n"

        action = "REPORT"  # Safe default fallback
        action_match = re.search(r"ACTION:\s*([A-Z]+)", generated_text)
        
        if action_match:
            parsed_action = action_match.group(1).strip()
            if parsed_action in self.action_tokens:
                action = parsed_action

        payload = ""
        payload_match = re.search(r"PAYLOAD:\s*(.*)", generated_text, re.DOTALL)
        
        if payload_match:
            payload_raw = payload_match.group(1).strip()
            
            if payload_raw.startswith("{") and payload_raw.endswith("}"):
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    self.fail_reason = "JSONDecodeError: Payload was malformed."
                    payload = payload_raw
            else:
               
                payload = payload_raw

  
        return action, payload
        
    def execute(self, action, payload):
        '''just makes it to teh handler methods'''
        if action == "THINK":
            # Do nothing, let the main agent loop continue
            pass
        elif action == "SPAWN":
            self.request_spawn(payload)
        elif action == "TOOL":
            self.request_tool(payload)
        elif action == "REPORT":
            self.report(payload)
        elif action == "DIE":
            self.die(payload)
        else:
            self.fail_reason = f"Execution failed: Unknown action '{action}'"

# these are the handler methods

    def request_spawn(self, payload):
        """Creates a sub-agent to handle a sub-task."""
        if not isinstance(payload, dict) or "role" not in payload or "task" not in payload:
            self.fail_reason = "SPAWN Action Failed: Payload must be a JSON object with 'role' and 'task'."
            return
        
        self.fail_reason = None
        
        event = Event(type="SPAWN", from_agent=self.agent_id)
        event.payload.update({
            "parent_id": self.agent_id,
            "role": payload["role"],
            "task_id": payload["task"],  
            "subtasks": payload.get("subtasks", None)
        })
        
        self.message.push(event)

    def request_tool(self, payload):
        """Calls an external tool."""
        if not isinstance(payload, dict) or "tool_name" not in payload or "args" not in payload:
            self.fail_reason = "TOOL Action Failed: Payload must be a JSON object with 'tool_name' and 'args'."
            return
        
        self.fail_reason = None
        
        event = Event(type="TOOL", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "tool_name": payload["tool_name"],
            "args": payload["args"]
        })
        
        self.message.push(event)

    def report(self, payload):
        """Submits the final result to the parent agent."""
        # Memory Management: Free VRAM for completed agents
        self.KV_Cache = None
        self.last_hidden_state = None
        
        event = Event(type="REPORT", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "result": str(payload)
        })
        
        self.message.push(event)

    def die(self, payload):
        """Signals catastrophic failure to the parent."""
        # Memory Management: Free VRAM for dead agents! 
        self.KV_Cache = None
        self.last_hidden_state = None
        
        event = Event(type="DIE", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "result": str(payload)  # Using result for the death reason
        })
        
        self.message.push(event)


    def run(self, available_roles=None, available_tools=None):
        """The core heartbeat coordinator for a single execution cycle."""
        action = self.think()
        
    # this is that decidor factor
        if action in self.action_tokens and action != "THINK":
            action, payload = self.decide(available_roles, available_tools)
        else:
            # If it's just "THINK" or a baseline state, payload is empty text
            payload = self.thought_process

        # giving it back to the executor
        self.execute(action, payload)
        
        return action