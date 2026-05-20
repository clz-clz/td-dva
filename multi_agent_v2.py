import logging
import os
import json
import re
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
import argparse

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from rag_voting_engine import RAGVotingEngine 
from utils import enforce_iob2_syntax 

load_dotenv()

rag_engine = RAGVotingEngine(db_path="./chroma_db")

llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7,
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

DEEPSEEK_NER_TOKEN_IDS = {
    "B-PER": 42531, "I-PER": 42532, 
    "B-LOC": 18274, "I-LOC": 18275,
    "B-ORG": 39112, "I-ORG": 39113,
    "B-MISC": 6489, "I-MISC": 6490
}

# =====================================================================
# Dataset-aware entity ontologies
# Each dataset has its own valid entity type set. The Coder and Reviewer
# prompts inject this list so the LLM doesn't default to the common
# PER/LOC/ORG triad regardless of input dataset.
# =====================================================================
DATASET_ENTITY_TYPES = {
    "msra":      ["PER", "LOC", "ORG"],
    "conll2003": ["PER", "LOC", "ORG", "MISC"],
    # WNUT-17 in this project is COARSENED to the standard 4-class ontology
    # via type_remap in gen_noisy.py (person->PER, location->LOC,
    # corporation/group->ORG, product/creative-work->MISC). The pipeline must
    # therefore prompt with the same coarse types.
    "wnut17":    ["PER", "LOC", "ORG", "MISC"],
}

def _format_valid_tags(dataset_name: str) -> str:
    """Return 'O, B-X, I-X, ...' as a single comma-separated string."""
    types = DATASET_ENTITY_TYPES.get(dataset_name)
    if not types:
        # Fallback: infer from a sample dirty_tags later, or use union
        types = ["PER", "LOC", "ORG", "MISC"]
    tags = ["O"]
    for t in types:
        tags.append(f"B-{t}")
        tags.append(f"I-{t}")
    return ", ".join(tags)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    loop_count: int
    iterations: int
    tokens: List[str]  
    dirty_tags: List[str]             
    candidate_paths: List[List[str]]  
    rag_weights: List[float]          
    current_tags: List[str]           
    errors: List[str]
    lambda_weight: float              
    use_wash: bool
    use_topo: bool
    dataset_name: str  # Required for dataset-aware prompts (msra / conll2003 / wnut17)

    

def extract_nested_json_list(llm_output: str, expected_paths: int = 5) -> List[List[str]]:
    try:
        match = re.search(r'\[\s*\[.*?\]\s*\]', llm_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return parsed[:expected_paths]
    except Exception as e:
        logging.warning(f"Coder JSON ：{e}")
    return [["O"]] * expected_paths

def extract_float_weights(llm_output: str, expected_len: int = 5) -> List[float]:
    try:
        match = re.search(r'\[[\d\.\s,]+\]', llm_output)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                weights = [float(x) for x in parsed]
                if len(weights) >= expected_len:
                    return weights[:expected_len]
                else:
                    return weights + [0.0] * (expected_len - len(weights))
    except Exception as e:
        logging.error(f" Reviewer JSON 载入异常: {e}")

    logging.error(f" Warning: {llm_output[:50]}...")
    return [1.0] * expected_len 


def _format_sentence_initial_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate guidance about sentence-initial entity heads."""
    if dataset_name == "msra":
        return (
            "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A Chinese "
            "sentence often begins with a named entity (organization, "
            "person, location). Do not default the first token to O just "
            "because it lacks left context. If the first 1-4 tokens form a "
            "recognizable entity, label them B-<TYPE> I-<TYPE> .... Examples: "
            "'中国政府...' -> B-ORG I-ORG I-ORG; '北京市...' -> B-LOC I-LOC I-LOC."
        )
    # English-style (CoNLL-2003 and coarsened WNUT-17)
    return (
        "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A capitalized "
        "token at position 0 that names a person, organization, location, or "
        "other entity should be labeled B-<TYPE>, not O. Do not default the "
        "first token to O just because it lacks left context. Examples: "
        "'Obama said ...' -> the first token is B-PER, not O. "
        "'Apple announced ...' -> the first token is B-ORG, not O. "
        "'German troops ...' -> the first token is B-MISC, not O."
    )


def _format_misc_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate MISC type guidance (only for ontologies that have MISC)."""
    if dataset_name == "msra":
        # MSRA uses only PER/LOC/ORG, no MISC. Skip the MISC paragraph entirely.
        return ""
    return (
        "- DO NOT systematically avoid rare tag types. The MISC type (for "
        "nationalities, languages, events, works, products, and other "
        "proper-noun entities that are not PER/LOC/ORG) is valid and should "
        "be predicted whenever the token clearly refers to such an entity, "
        "even if MISC appears less frequently than PER/LOC/ORG. Examples of "
        "MISC entities: \"German\" (nationality), \"Olympic\" (event), "
        "\"iPhone\" (product), \"Bible\" (work)."
    )


def coder_node(state: State):
    print(f"\n [Coder](Self-Consistency)")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    dataset_name = state.get("dataset_name", "conll2003")
    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)
    
    active_llm = llm

    prompt = f"""
    You are an Elite AI Data Engineer performing IOB2 label denoising.
    Tokens: {tokens}
    Dirty IOB2 Tags: {dirty_tags}

    VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
    {valid_tags_str}

    DENOISING POLICY:
    - The dirty tags contain noise but most positions are already correct.
    - Preserve non-O tags from the dirty input unless they clearly conflict with the token semantics or IOB2 grammar.
    - When a dirty tag is non-O, prefer to keep it rather than collapsing to O. Only convert non-O tags to O when the token is clearly not part of any entity (e.g., a common verb, preposition, or punctuation).
    {misc_guidance}
    {sentence_initial_guidance}

    Generate EXACTLY 15 different plausible IOB2 repair paths.
    1. Every path must have exactly {len(tokens)} tags.
    2. Must follow IOB2 syntax strictly (I-X must be preceded by B-X or I-X of the same type).
    3. Use ONLY tags from the VALID TAG VOCABULARY above. Do NOT invent new tag names or change case.
    4. The 15 paths should explore different plausible interpretations, including paths that preserve the dirty tag, paths that promote rare types where appropriate, and paths that recover sentence-initial B- tags where they may have been dropped.
    Output ONLY a JSON list of 15 lists. No markdown.
    Example format: [["O", "B-PER", "I-PER"], ["B-PER", "I-PER", "I-PER"], ...]
    """
    
    response = active_llm.invoke(prompt)
    candidate_paths = extract_nested_json_list(response.content, expected_paths=15)
    
    candidate_paths = [p[:len(tokens)] + ["O"] * max(0, len(tokens) - len(p)) for p in candidate_paths]

    return {"candidate_paths": candidate_paths, "iterations": state.get("iterations", 0) + 1}

def reviewer_node(state: State):
    print(f"\n [Reviewer]")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    candidate_paths = state.get("candidate_paths", [])
    
    if state.get("use_topo", True):
        retrieved_cleans = rag_engine.retrieve_hard_examples(dirty_tags, top_k=3)
    else:
        print(" [PI DEBUG]")
        retrieved_cleans = [] 

    reviewer_prompt = f"""
    You are a RAG-Driven Adjudicator.
    Blind Tokens: {tokens}
    Dirty Tags: {dirty_tags}

    VALID TAG VOCABULARY for this dataset (case-sensitive):
    {_format_valid_tags(state.get("dataset_name", "conll2003"))}

    {len(candidate_paths)} candidate paths:
    {json.dumps(candidate_paths, indent=2)}
    
    Historical Evidence (RAG):
    {json.dumps(retrieved_cleans, indent=2)}
    
    Evaluate each candidate path against the Historical Evidence and the VALID TAG VOCABULARY.
    Candidates using tags outside the VALID TAG VOCABULARY should receive low scores.
    Output ONLY a JSON list of {len(candidate_paths)} float numbers between 0.0 and 1.0 representing the Confidence Score (\omega) for each path.
    Example: [0.1, 0.9, 0.45, 0.0, 0.8]
    """
    
    response = llm.invoke(reviewer_prompt)
    rag_weights = extract_float_weights(response.content, expected_len=len(candidate_paths))
    return {"rag_weights": rag_weights}

def voting_node(state: State):
    lam = state.get("lambda_weight", 1.35) 
    
    candidate_paths = state.get("candidate_paths", [])
    rag_weights = state.get("rag_weights", [])
    
    raw_voted_tags = rag_engine.majority_voting_with_rag(
        candidate_paths, 
        rag_weights, 
        entity_boost=lam 
    )
    
    return {"current_tags": raw_voted_tags}


def physical_wash_node(state: State):
   
    print(f"\n [Physical Wash] ...")
    tokens = state.get("tokens", [])
    raw_tags = state.get("current_tags", [])
    dirty_tags = state.get("dirty_tags", []) 

    try:
        final_safe_tags = enforce_iob2_syntax(raw_tags)

        if not final_safe_tags or len(final_safe_tags) == 0:
            print(f" [DFA Alert]")
            final_safe_tags = dirty_tags if dirty_tags else ["O"] * len(tokens)
            

        if len(final_safe_tags) != len(tokens):
            print(f"[Length Correction]: {len(final_safe_tags)} -> {len(tokens)}")
            if len(final_safe_tags) > len(tokens):
                final_safe_tags = final_safe_tags[:len(tokens)]
            else:
                final_safe_tags.extend(["O"] * (len(tokens) - len(final_safe_tags)))
                
        print(f"sample clean finish: {len(final_safe_tags)}")
        return {"current_tags": final_safe_tags}
        
    except Exception as e:

        return {"current_tags": ["O"] * len(tokens)}


workflow = StateGraph(State)

workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("voting", voting_node)  
workflow.add_node("physical_wash", physical_wash_node)

workflow.add_edge(START, "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", "voting")


def route_wash(state: State):
    if state.get("use_wash", True) is False:
        return END
    return "physical_wash"

workflow.add_conditional_edges("voting", route_wash)
workflow.add_edge("physical_wash", END)

multi_agent_graph = workflow.compile()


#if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="TD-DVA Agent Runner")
    #parser.add_argument("--input", required=True, help="Path to input noisy jsonl")
    #parser.add_argument("--output", required=True, help="Path to save predictions")
    #args = parser.parse_args()
    
    # run_agent_pipeline(args.input, args.output)


async def run_agent_pipeline(tokens: List[str], dirty_tags: List[str],
                              config: dict = None,
                              dataset_name: str = None) -> List[str]:
    """
    `dataset_name` controls which entity-type ontology is injected into
    the Coder / Reviewer prompts. Allowed values: 'msra', 'conll2003', 'wnut17'.
    If omitted, falls back to config['__dataset__'], then to 'conll2003'.
    """
    if config is None:
        # Full System
        config = {"lambda_bias": 1.35, "use_dfa": True, "use_topology_rag": True}

    # Dataset name resolution: explicit arg → config["__dataset__"] → default
    if dataset_name is None:
        dataset_name = config.get("__dataset__", "conll2003")
    if dataset_name not in DATASET_ENTITY_TYPES:
        logging.warning(f"[!] Unknown dataset_name={dataset_name!r}; "
                       f"defaulting to conll2003 ontology. "
                       f"Known: {list(DATASET_ENTITY_TYPES)}")
        dataset_name = "conll2003"

    initial_state = {
        "messages": [],
        "tokens": tokens,
        "dirty_tags": dirty_tags,
        "loop_count": 0,
        "iterations": 0,
        "errors": [],
        "candidate_paths": [],
        "rag_weights": [],
        "current_tags": [],
        "lambda_weight": config.get("lambda_bias", 1.35), 
        "use_wash": config.get("use_dfa", True),
        "use_topo": config.get("use_topology_rag", True),
        "dataset_name": dataset_name,
    }
    
    try:
        final_state = await multi_agent_graph.ainvoke(initial_state)
        predicted_tags = final_state.get("current_tags", [])
        
        if not predicted_tags or len(predicted_tags) != len(tokens):
            return dirty_tags 
            
        return predicted_tags
    except Exception as e:
        logging.error(f"[-] Pipeline : {e}")
        return dirty_tags