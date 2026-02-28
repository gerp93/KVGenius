# World Building System Implementation Guide

## Overview
Create a world building / lore system where users can define world contexts with subtopics that are context-aware and automatically injected into chat conversations when relevant. This provides rich world context without cluttering the main system prompt.

## Feature Requirements

### 1. Data Model

#### Database Schema (SQLite - add to `src/database/chat_history.py`)
```sql
-- Lore/World Building Contexts
CREATE TABLE IF NOT EXISTS lore_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subtopics within a context
CREATE TABLE IF NOT EXISTS lore_subtopics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    tags TEXT,  -- comma-separated keywords for matching
    content TEXT,  -- the actual lore/context text
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_id) REFERENCES lore_contexts(id) ON DELETE CASCADE
);

-- Junction table: Characters can have multiple lore contexts
CREATE TABLE IF NOT EXISTS character_lore_contexts (
    character_id TEXT NOT NULL,
    context_id INTEGER NOT NULL,
    PRIMARY KEY (character_id, context_id),
    FOREIGN KEY (context_id) REFERENCES lore_contexts(id) ON DELETE CASCADE
);
```

### 2. Core Functions (in `ChatHistoryDB` class)

#### Context Management
- `create_lore_context(name: str, description: str = "") -> int` — Returns context ID
- `list_lore_contexts() -> List[Dict]` — Returns all contexts with their subtopics count
- `get_lore_context(context_id: int) -> Dict` — Full context with all subtopics
- `update_lore_context(context_id: int, name: str, description: str) -> bool`
- `delete_lore_context(context_id: int) -> bool`

#### Subtopic Management
- `create_lore_subtopic(context_id: int, name: str, tags: str, content: str) -> int`
- `get_lore_subtopic(subtopic_id: int) -> Dict`
- `update_lore_subtopic(subtopic_id: int, name: str, tags: str, content: str) -> bool`
- `delete_lore_subtopic(subtopic_id: int) -> bool`

#### Character-Context Association
- `add_context_to_character(character_id: str, context_id: int) -> bool`
- `remove_context_from_character(character_id: str, context_id: int) -> bool`
- `get_character_lore_contexts(character_id: str) -> List[Dict]` — All contexts for a character
- `get_contexts_by_keywords(keywords: List[str]) -> List[Dict]` — Find subtopics matching any keyword

### 3. Context Injection Logic

#### Keyword Matching Function
Create a function in `core/chat_gen.py` that:
1. Takes the latest user message
2. Extracts all "tags" from all subtopics in the character's lore contexts
3. Finds overlapping words (case-insensitive, word-boundary matching)
4. Returns list of matching subtopics with their content

```python
def extract_relevant_lore(message: str, character_id: str) -> List[Dict]:
    """
    Extract lore subtopics relevant to the current message.
    
    Returns:
        List of dicts with keys: {name, tags, content, context_name}
    """
```

#### Integration with Chat Generation
Modify `generate_chat_response()` in `core/chat_gen.py`:
- After building the base system prompt
- Before sending to model
- Call `extract_relevant_lore(user_message, character_id)`
- If matches found, append them to system prompt with a header like:
  ```
  [RELEVANT WORLD CONTEXT]
  Context: World_Name > Subtopic_Name
  Tags: keyword1, keyword2
  
  <content>
  ```

### 4. Export/Import

#### Export to JSON
Function `export_lore_as_json(context_id: int) -> str`:
```json
{
  "context": {
    "name": "Star Wars Universe",
    "description": "The galaxy far far away",
    "subtopics": [
      {
        "name": "Tatooine",
        "tags": "tatooine, desert, lars farm, binary sunset",
        "content": "Twin-mooned desert planet where Luke Skywalker grew up..."
      }
    ]
  }
}
```

Function `export_character_lore_as_json(character_id: str) -> str`:
- Export all contexts associated with a character

#### Import from JSON
- `import_lore_from_json(json_str: str) -> bool` — Creates/updates contexts

### 5. UI Implementation (in `desktop_app.py`)

#### New Tab: "📚 Lore Book" under Chat Section
- Similar to Model Manager in layout
- Two sub-sections:

##### Left Panel: Context List
- ListView of all lore contexts
- Search bar to filter
- Buttons: [+ New Context] [Edit] [Delete] [Export]
- When clicked, show details on right

##### Right Panel: Context Details & Subtopics
- Display selected context name, description
- Table/list of subtopics with:
  - Name, Tags preview, Content preview (truncated)
  - Buttons per row: [Edit] [Delete] [↑ ↓ reorder]
- Button: [+ Add Subtopic]

#### Dialogs
- **New/Edit Context Dialog**:
  - Text field: Context Name
  - Text field: Description
  - Button: [Create/Save]
  
- **New/Edit Subtopic Dialog**:
  - Text field: Subtopic Name
  - Text area: Tags (comma-separated)
  - Text area: Content (large, multi-line)
  - Button: [Create/Save]

- **Character-Context Assignment Dialog**:
  - Show when editing a character
  - List of all contexts with checkboxes
  - Allow user to select which contexts apply to this character

#### Integration with Character Tab
- When editing/creating a character, add button/section:
  - "📚 Manage Lore Contexts" 
  - Opens dialog showing available contexts
  - User selects which ones to link to this character

### 6. Chat Integration

When viewing chat conversation:
- If lore was used in generating a response, show an indicator (e.g., small 📚 icon next to AI message)
- On hover/click, show which lore subtopic was used:
  ```
  📚 Used: Star Wars > Tatooine
  ```

### 7. Export with Conversations

Modify conversation export in `core/chat_gen.py`:
- When exporting a conversation as JSON/text
- Include a "LORE_CONTEXTS_USED" section listing all subtopics mentioned in the chat
- Include full lore context at the end of export for reference

## Implementation Steps

1. **Database layer** (chat_history.py)
   - Add table definitions
   - Implement CRUD functions for contexts, subtopics, associations

2. **Core logic** (core/chat_gen.py)
   - `extract_relevant_lore()` function with keyword matching
   - Modify `generate_chat_response()` to inject lore into system prompt
   - Track which lore was used (for display in UI)

3. **UI layer** (desktop_app.py)
   - New "Lore Book" tab with context/subtopic management
   - Character dialog updates for context assignment
   - Chat view indicators for used lore

4. **Export/Import**
   - JSON export/import functions
   - Integration with existing conversation export

5. **Testing**
   - Keyword matching edge cases
   - Circular dependencies
   - Export/import round-trip validation

## Notes

- Tags should use common words (place names, topic keywords, character names)
- Content should be substantial enough to be useful (50+ chars minimum)
- Limit tags to ~5-10 per subtopic for performance
- Consider context injection order (most specific/recent first)
- Watch for token budget - lore injection should not exceed 500 tokens per message

## Success Criteria

- ✅ User can create/edit/delete contexts and subtopics
- ✅ Contexts can be linked to characters
- ✅ Relevant lore automatically injected when keywords match
- ✅ Export/import works round-trip without data loss
- ✅ UI is intuitive and doesn't clutter chat tab
- ✅ No performance degradation with large lore databases
