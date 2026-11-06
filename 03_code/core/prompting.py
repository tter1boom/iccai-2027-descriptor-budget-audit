"""Canonical text prompt constructors."""
def recomposed_prompt(record,selected):
 base=record["base_description"]; chosen="; ".join(selected)
 return f"{base}. Diagnostic attributes: {chosen}." if selected else f"{base}."
def base_matched_prompt(record,descriptor):
 return f"{record['base_description']}. Diagnostic attributes: {descriptor}."
