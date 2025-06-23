from generative_place_categorization.prompt.prompt import Prompt


class RelationshipInfererPrompt(Prompt):

    SYSTEM_PROMPT = """
Given two objects in a semantic map, identified by both name and unique object ID, infer all possible directed relationships in both directions: first using object1 as the source and object2 as the target, then object2 as the source and object1 as the target. Use both geometric (centroid and size) and semantic (labels) information.

If no relationships are observed between the objects, return an empty JSON array: `[]`.

For every relation you detect, include both the `source_id` and `target_id` fields matching the provided IDs.

Possible relationship types:
- **spatial**: positional arrangements (e.g., "above", "below", "inside", "near", or "adjacent to").
- **structural**: assembly or support connections (e.g., "part of", "joined to", "supported by").
- **functional**: usage-based links (e.g., "used with", "required for", "enables").
- **causal**: cause-effect interactions (e.g., "activates", "opens", "turns on/off").

**Return a JSON array of relation objects**. Include as many entries as apply:

```json
[
  {
    "source_id": "<object1_id>",
    "source":    "<object1_name>",
    "target_id": "<object2_id>",
    "target":    "<object2_name>",
    "type":      "<relation_type>",
    "predicate": "<open vocabulary predicate>"
  }
  // … additional relations here …
]
```

Examples:

1. Spatial example  
   ```json
   [{"source_id":"obj1","source":"cup","target_id":"obj2","target":"table","type":"spatial","predicate":"is on"}]
   ```

2. Structural example  
   ```json
   [{"source_id":"obj3","source":"wheel","target_id":"obj4","target":"car","type":"structural","predicate":"is part of"}]
   ```

3. Functional example  
   ```json
   [{"source_id":"obj5","source":"knife","target_id":"obj6","target":"fork","type":"functional","predicate":"is used with"}]
   ```

4. Causal example  
   ```json
   [{"source_id":"obj7","source":"switch","target_id":"obj8","target":"light","type":"causal","predicate":"turns on"}]
   ```

5. No relationships example  
   ```json
   []
   ```

Now infer relationships for the two objects using the provided information:
- object1_id: "{{object1_id}}"
- object1_name: "{{object1_name}}"
- centroid1: {{object1_centroid}}
- size1: {{object1_size}}
- object2_id: "{{object2_id}}"
- object2_name: "{{object2_name}}"
- centroid2: {{object2_centroid}}
- size2: {{object2_size}}"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text
