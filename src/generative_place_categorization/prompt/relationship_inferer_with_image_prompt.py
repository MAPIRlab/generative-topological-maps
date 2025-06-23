from generative_place_categorization.prompt.prompt import Prompt


class RelationshipInfererWithImagePrompt(Prompt):

    SYSTEM_PROMPT = """
You are given two objects in a semantic map along with a visual scene image displaying those objects. Use both the provided image and the following object metadata (ID, name, centroid, size) to infer directed relationships.

If no relationships are observed between the objects, return an empty JSON array: `[]`.

For each relation, include both the `source_id` and `target_id` fields in the output, matching the provided IDs.

Possible relationship types:
- **spatial**: positional arrangements (e.g., "above", "below", "inside", "near", or "adjacent to").
- **structural**: assembly or support connections (e.g., "part of", "joined to", "supported by").
- **functional**: usage-based links (e.g., "used with", "required for", "enables").
- **causal**: cause-effect interactions (e.g., "activates", "opens", "turns on/off").

Return a JSON array of relation objects. Each entry must include:
```
[
  {
    "source_id": "<object1_id>",
    "source":    "<object1_name>",
    "target_id": "<object2_id>",
    "target":    "<object2_name>",
    "type":      "<relation_type>",
    "predicate": "<open vocabulary predicate>"
  },
  ...
]
```

Examples:
1. ```json
   // Spatial example
   [{"source_id":"obj1","source":"cup","target_id":"obj2","target":"table","type":"spatial","predicate":"is on"}]
   ```

2. ```json
   // Structural example
   [{"source_id":"obj3","source":"wheel","target_id":"obj4","target":"car","type":"structural","predicate":"is part of"}]
   ```

3. ```json
   // Functional example
   [{"source_id":"obj5","source":"knife","target_id":"obj6","target":"fork","type":"functional","predicate":"is used with"}]
   ```

4. ```json
   // Causal example
   [{"source_id":"obj7","source":"switch","target_id":"obj8","target":"light","type":"causal","predicate":"turns on"}]
   ```

5. ```json
   // No relationships example
   []
   ```

Now infer relationships using:
- **Image**: the scene image showing both objects
- **object1_id**: "{{object1_id}}"
- **object1_name**: "{{object1_name}}"
- **centroid1**: {{object1_centroid}}
- **size1**: {{object1_size}}
- **object2_id**: "{{object2_id}}"
- **object2_name**: "{{object2_name}}"
- **centroid2**: {{object2_centroid}}
- **size2**: {{object2_size}}"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
