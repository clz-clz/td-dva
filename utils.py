import re
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


def enforce_iob2_syntax(tags: List[str]) -> List[str]:

    VALID_ENTITIES = {"PER", "LOC", "ORG"}
    fixed = []
    
    for i, tag in enumerate(tags):
        
        clean_tag = tag
        if clean_tag != "O":
            try:
                prefix, ent_type = clean_tag.split("-", 1)
                if prefix not in {"B", "I"} or ent_type not in VALID_ENTITIES:
                    clean_tag = "O" 
            except ValueError:
                clean_tag = "O"
                
        
        if clean_tag.startswith("I-"):
            if i == 0:
                fixed.append("B-" + clean_tag[2:])
                continue
            prev_tag = fixed[i-1] 
            if prev_tag == "O":
                fixed.append("B-" + clean_tag[2:])
                continue
            prev_type = prev_tag.split("-", 1)[1]
            curr_type = clean_tag.split("-", 1)[1]
            if prev_type != curr_type:
                fixed.append("B-" + clean_tag[2:])
                continue
                
        fixed.append(clean_tag)
        
    return fixed

def extract_json_list(llm_output: str, fallback_length: int) -> list:
    if isinstance(llm_output, list):
        return llm_output
    try:
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_list = json.loads(json_str)
            if isinstance(parsed_list, list):
                return parsed_list
        parsed_list = json.loads(llm_output)
        if isinstance(parsed_list, list):
            return parsed_list
    except Exception as e:
        logger.warning(f"JSON parsing failed: {llm_output[:50]}...")
    return ["O"] * fallback_length