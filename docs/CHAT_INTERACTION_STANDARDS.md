# Chat Interaction Standards Implementation Guide

## Overview
Implement a more expressive chat interaction system with support for:
1. **Actions/Narration** — Text in `*asterisks*` represents internal thoughts or actions, not dialog
2. **Directions Box** — An optional expandable field where users give instructions to the character without sending those instructions as dialog

## Feature Requirements

### 1. UI/UX Changes

#### Chat Input Area Redesign

Current layout (approximate):
```
[Chat input text area]
[Send button]
```

New layout:
```
[Chat input text area - for dialog/messages only]
[⋮ Expand directions / ↥ Collapse directions]  (toggle)

[Directions text area - HIDDEN by default]  ⬅️ Only visible when expanded
(user can write "Mary should be confused" or "respond with an action")

[Send] [Clear]
```

##### Dialog Text Area
- Placeholder: "Type your character's response or dialog..."
- Normal chat area (unchanged)
- Input is assumed to be what the character hears or experiences
- Supports `*action text*` formatting

##### Directions Text Area (Expandable, Hidden by Default)
- Placeholder: "Optional: Give directions on how the character should respond (not sent to AI)..."
- Acts as a "prompt engineering" layer
- Larger font / distinct visual styling
- Used to control:
  - Tone ("respond angrily")
  - Action direction ("have Mary stand up and look around")
  - Emotional state ("be confused but trying to hide it")
  - Scene direction ("this is a tense moment")

#### Toggle Button
- Location: Right side of dialog input, above the text area
- Icon: `⋮` (three dots) or `▼/▲` (chevron)
- Tooltip: "Show directions (optional instructions for character)"
- Clicking toggles directions area visibility
- State persists across messages (if user expands it, it stays expanded)

### 2. Message Parsing & Formatting

#### Action/Narration Detection
Create a parser function in `core/chat_gen.py`:

```python
def parse_message_with_actions(message: str) -> Dict:
    """
    Parse a message for *action* markers and dialog.
    
    Returns:
        {
            "has_actions": bool,
            "has_dialog": bool,
            "action_parts": List[str],  # extracted *action* text
            "dialog_parts": List[str],  # non-action text
            "formatted": str,  # HTML/markdown formatted version
            "plain": str,  # plain text with actions indicated
            "display_text": str  # what to show in UI
        }
    
    Examples:
        "*looks around confused*" 
        -> action_parts: ["looks around confused"]
        
        "I don't understand. *nervous laugh*"
        -> dialog_parts: ["I don't understand."]
        -> action_parts: ["nervous laugh"]
        
        "*picks up the sword and examines it* That's a fine weapon."
        -> action_parts: ["picks up the sword and examines it"]
        -> dialog_parts: ["That's a fine weapon."]
    """
```

#### UI Display
In the chat message view:
- Dialog text: Normal display
- Action text (was in `*...*`): 
  - Italic formatting
  - Different color (gray or muted)
  - Prefix with `*` and suffix with `*` to make it clear it's narration
  - Example: `*nervously adjusts glasses*`

#### System Prompt Integration
When sending to model, format like:
```
User: *opens the door and steps inside nervously* What's happening here?
```
The model is already trained on this format from creative writing datasets.

### 3. Directions System

#### Client-Side Handling
In `core/chat_gen.py`, create:

```python
def build_message_with_directions(dialog: str, directions: str) -> str:
    """
    Build the final message to send to the model, incorporating directions
    as a system-level instruction without making them part of dialog.
    
    Args:
        dialog: What the user/character is saying (supports *actions*)
        directions: Optional meta-instructions for how to respond
    
    Returns:
        Formatted message string for model input
    """
```

#### Approach 1: Directions as System Context (Recommended)
If user provides directions, append them to the **system prompt momentarily**:

```
[System Prompt + Lore]
...

[CURRENT INSTRUCTIONS]
Remember to: {directions}
[/CURRENT INSTRUCTIONS]

User: {dialog}
```

This way:
- Directions don't appear in conversation history
- Model sees them as meta-context
- They influence the response without being visible to user

#### Approach 2: Directions as Parenthetical (Alternative)
Append to user message in parentheses:
```
User: I'm here. *(Respond with confusion and fear)*
```
Then parser removes the parenthetical before storing in history.
- Simpler to implement
- Might interfere with model if it's confused by meta-text
- **Not recommended** unless Approach 1 doesn't work well

**Use Approach 1 (temp system context).**

#### Storage
- Directions are **NOT** stored in conversation history
- They are **NOT** shown in conversation export
- They are purely functional for the current turn
- After model responds, directions box can be cleared or kept for next turn (user choice)

### 4. Implementation Details

#### Desktop App Changes (desktop_app.py)

##### Chat Input UI Refactor
Around the chat input area (find existing chat message input box):

```python
# Existing
chat_input = TextField(label="Your response", multiline=True, expand=True)
send_btn = ElevatedButton("Send", on_click=self._send_message)

# New
chat_input = TextField(label="Your response (use *actions* for narration)", multiline=True, expand=True)
directions_input = TextField(
    label="Optional: Give directions on how character should respond",
    multiline=True,
    expand=True,
    visible=False,  # Start hidden
)
directions_toggle = IconButton(
    icon=Icons.EXPAND_MORE,
    tooltip="Show directions",
    on_click=lambda e: self._toggle_directions(),
)
send_btn = ElevatedButton("Send", on_click=self._send_message)

# Layout
Column([
    Row([chat_input, directions_toggle]),
    directions_input,  # Hidden until toggled
    Row([send_btn, clear_btn]),
])
```

Add instance variable to track state:
```python
self.directions_visible = False
```

##### Methods to Add
```python
def _toggle_directions(self):
    """Toggle visibility of directions input."""
    self.directions_visible = not self.directions_visible
    self.directions_input.visible = self.directions_visible
    # Update button icon
    if self.directions_visible:
        self.directions_toggle.icon = Icons.EXPAND_LESS
    else:
        self.directions_toggle.icon = Icons.EXPAND_MORE
    self.page.update()

def _send_message(self):
    """Handle sending a message with optional directions."""
    dialog = self.chat_input.value.strip()
    directions = self.directions_input.value.strip()
    
    if not dialog:
        return  # Don't send empty messages
    
    # Process message (parse actions, build prompt with directions)
    # ... existing send logic but now pass directions
    
    # Clear dialog input (good UX)
    self.chat_input.value = ""
    # Optionally clear directions too, or leave for next turn
    # self.directions_input.value = ""
    
    self.page.update()
```

##### Chat Display Formatting
When displaying received messages in the conversation view, apply action formatting:

```python
def _render_message(self, role: str, message: str):
    """
    Render a message in the chat view, formatting actions.
    """
    parsed = parse_message_with_actions(message)
    
    # Use parsed["display_text"] or build RichText with formatting
    # Actions should show in italic/muted color
    
    # Example using Flet's Text with style
    display = RichText(spans=[
        TextSpan(
            text=parsed["display_text"],
            style=TextStyle(font_family=...)
        )
    ])
```

#### Core Chat Generation Changes (core/chat_gen.py)

```python
def generate_chat_response(
    character_id: str,
    persona_id: str = None,
    user_message: str = "",
    directions: str = "",  # NEW
) -> GenerationResult:
    """
    Generate a chat response with optional directions.
    
    Args:
        user_message: The dialog/action the user is providing
        directions: Optional meta-instructions for character behavior
    """
    
    # ... existing setup ...
    
    # Parse actions in user message
    parsed = parse_message_with_actions(user_message)
    
    # Build system prompt with directions if provided
    if directions:
        system_prompt += f"\n[CURRENT SCENE INSTRUCTIONS]\n{directions}\n[/CURRENT SCENE INSTRUCTIONS]"
    
    # Send to model (using parsed message)
    response = model.generate(
        system_prompt=system_prompt,
        user_message=parsed["plain"] or user_message,  # Use parsed if available
        # ... other params
    )
    
    # Store in history (only dialog, without directions)
    store_message(
        role="user",
        content=parsed["plain"] or user_message,
        character_id=character_id,
        # directions are NOT stored
    )
    
    return response
```

### 5. Database Considerations

- No new tables needed (directions are transient)
- Conversation history stores only the final user message without directions
- If you want audit trail of directions used, add optional `directions_text` column to `conversation_messages` but mark as internal/not exported

### 6. Export Behavior

When exporting conversations:
- Include action markers: `*actions* appear in exported text`
- Do **NOT** include directions (they never appear in export)
- Example export format:
  ```
  User: *takes a deep breath* I'm scared but I'm ready.
  
  Character: *nods reassuringly* You can do this. I'm with you.
  ```

### 7. Testing & Examples

#### Test Cases
1. **Pure action**: `*character faints*` 
   - Should display as italic narration
   - Model should treat as narrative description

2. **Pure dialog**: `"Hello there"`
   - Should display normally
   - Works as before

3. **Mixed**: `*stands up dramatically* This is unacceptable!`
   - Action part formatted as narration
   - Dialog part as normal text
   - Model sees both parts

4. **Directions only**: Dialog: `"What should I do?"` | Directions: `"Be indecisive and confused"`
   - Directions injected into system prompt
   - User never sees directions in history
   - Response reflects the emotional direction

5. **Complex**: 
   - Dialog: `*nervously laughs* I don't know what you mean.`
   - Directions: `You're actually lying and scared. Let that show in your body language.`
   - Result: Character's response reflects both the stated uncertainty and the hidden fear

### 8. Implementation Steps

1. **Parser function** (`parse_message_with_actions()`)
   - Test with various action formats
   - Edge cases: nested asterisks, escaped asterisks, etc.

2. **UI elements** (desktop_app.py)
   - Add directions input box (hidden)
   - Add toggle button
   - Hook up toggle event

3. **Message sending** (desktop_app.py)
   - Extract dialog and directions from UI
   - Pass both to generation function

4. **Chat generation** (core/chat_gen.py)
   - Parse user message for actions
   - Inject directions into system prompt
   - Store only dialog (not directions) in history

5. **Message display** (desktop_app.py)
   - Format actions with italic/muted styling
   - Display normally for dialogs

6. **Testing**
   - Unit test parser function
   - Integration test with actual model
   - Manual test: various action/dialog/direction combinations

## Success Criteria

- ✅ Users can write `*actions*` for narration
- ✅ Actions display distinctly (italic/muted) in chat history
- ✅ Directions input box expands/collapses on demand
- ✅ Directions affect character response without appearing in history
- ✅ Export does NOT include directions
- ✅ Export DOES format actions properly
- ✅ Parser handles edge cases (empty actions, long descriptions, etc.)
- ✅ No performance impact from parsing

## Notes

- The asterisk format `*...*` is already common in creative writing communities
- Markdown uses `*` for italics, so it's intuitive
- Consider warning user if they use asterisks in normal dialog (parsing ambiguity)
- Directions should be brief (< 50 words) to fit in system context
- Test with various models to ensure they respect action formatting

## Optional Future Enhancements

- Save "canned directions" as templates (e.g., "Angry", "Confused", "Sarcastic")
- Direction templates per character/persona
- Highlighting of actions in the UI with special formatting
- Analysis of character consistency (warnings if directions contradict character)
