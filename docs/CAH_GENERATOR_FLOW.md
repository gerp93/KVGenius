leuwyz3680  -\'`. Cards Against Humanity Generator - Process Flow+

This document explains how the Card Generator works, distinguishing between programmatic logic and AI-generated content.

---

## Overview

The Card Generator has two modes:
1. **Topic-Based Generation** - You provide a topic, AI creates cards
2. **Article Extraction** - You paste an article, AI extracts jokes as cards

Both modes use the same underlying flow:
```
User Input → Prompt Assembly (Programmatic) → AI Generation → Parsing (Programmatic) → Save
```

---

## 🔧 What's Programmatic (No AI)

### 1. **Prompt Assembly**
The system builds prompts by combining:
- User's visible input (topic or article text)
- Hidden system prompts (style-specific instructions)
- Hidden formatting instructions (JSON output format)

### 2. **Response Parsing**
After AI generates text, we programmatically:
- Extract JSON from the response (handles markdown code blocks)
- Parse each card's text and type
- **Force card type** if user selected "White Only" or "Black Only" (ignores AI's type labels)
- Assign unique IDs and timestamps
- Save to `data/cah/generated/cards.json`

### 3. **Card Management**
- Saving/loading cards from disk
- Exporting to JSON/TXT/CSV
- Deleting cards
- Tracking statistics

---

## 🤖 What Uses AI

The AI (your loaded chat model) only does **one thing**: Generate the card text based on the assembled prompt.

The AI receives:
1. A **system prompt** (hidden, style-specific)
2. A **user prompt** (combines your input with formatting instructions)

The AI outputs: A list of card texts (ideally in JSON format)

---

## Hidden Prompts (What You Don't See)

### YES - There are hardcoded prompts that augment your input!

---

## Topic-Based Generation Flow

### Step 1: User Input (Visible)
```
Topic: "Office life"
Card Type: White
Quantity: 10
Style: Classic
```

### Step 2: System Prompt Assembly (Hidden)

Based on your **Style** selection, one of these system prompts is used:

#### CLASSIC Style:
```
You are a Cards Against Humanity card writer. Your job is to create 
hilarious, irreverent, and often crude cards that push boundaries 
while being clever.

Rules for BLACK cards (prompts):
- Use "___" to indicate blanks where white cards go
- Can have 1-3 blanks
- Should set up a joke or absurd situation
- Often reference pop culture, politics, or taboo topics

Rules for WHITE cards (answers):
- Short phrases or single concepts (2-8 words typically)
- Can be nouns, actions, concepts, or famous people/things
- The funnier and more unexpected, the better
- Should work as punchlines for various black cards

Be creative, provocative, and funny. Don't be boring or generic.
```

#### Other Styles Available:
- **ABSURD** - Surreal, nonsensical, dreamlike humor
- **DARK** - Gallows humor, morbid jokes
- **WHOLESOME** - Family-friendly, clean humor
- **NERDY** - Gaming, tech, sci-fi references
- **CUSTOM** - Uses your custom instructions instead

### Step 3: User Prompt Assembly (Hidden Formatting Added)

Your topic gets wrapped with formatting instructions:

```
Generate 10 WHITE cards about: Office life

Style: classic

Format your response as a JSON array like this:
[
  {"text": "Card text here", "type": "white", "blanks": 0},
  {"text": "Another card", "type": "white", "blanks": 0}
]

For BLACK cards, use ___ for blanks and set blanks count accordingly.
For WHITE cards, blanks should be 0.

Be creative and match the requested style!
```

### Step 4: AI Generation
The model receives both prompts and generates a response.

### Step 5: Parsing (Programmatic)

```python
# Try to extract JSON
if "```json" in response:
    # Extract from markdown code block
    json_str = response[start:end]

# Parse JSON
data = json.loads(json_str)

# Force card type if user selected specific type
if force_type == CardType.WHITE:
    card_type = "white"
    blanks = 0  # Override any blanks
elif force_type == CardType.BLACK:
    card_type = "black"
```

---

## Article Extraction Flow

### Step 1: User Input (Visible)
```
Input Mode: "Enter URL" or "Paste Text"
URL: https://clickhole.com/funny-article  (if URL mode)
Article Text: [Pasted article from Clickhole, etc.]  (if Text mode)
Card Type: White
Quantity: 8
```

### Step 1b: URL Fetching (Programmatic - if URL mode)
If you enter a URL instead of pasting text:
1. Fetch HTML from the URL
2. Parse with BeautifulSoup
3. Extract article content (looks for `<article>`, `.article-content`, `<main>`, etc.)
4. Strip navigation, scripts, ads
5. Return clean article text

### Step 2: System Prompt (Hidden)

This is the **Article Extraction System Prompt**:

```
You are an expert at extracting humor from articles and converting 
them into Cards Against Humanity style cards.

Your task is to:
1. Read the provided article text
2. Identify the funniest lines, jokes, absurd statements, or memorable phrases
3. Convert them into CAH cards by:
   - Adjusting verb tenses to work as standalone cards
   - Making them punchy and concise (white cards: 2-30 words, black cards: short sentences with ___ blanks)
   - Preserving the humor and absurdity of the original
   - Creating black cards (prompts) from questions or setups
   - Creating white cards (answers) from punchlines, nouns, or funny phrases

Rules for conversion:
- White cards should be noun phrases, gerunds (-ing), or short concepts
- Black cards should have ___ where answers would go
- Convert to present tense or gerunds: "He was eating the cake" → "Eating the cake"
- Remove article-specific context that wouldn't make sense standalone
- Keep the core joke/absurdity intact

Output format: JSON array with "text", "type", and "blanks" fields.
```

### Step 3: User Prompt Assembly (Hidden Formatting Added)

```
Extract and create 8 WHITE cards from this article.

ARTICLE TEXT:
---
[Your pasted article appears here]
---

Find the funniest jokes, phrases, and absurd statements. Convert them 
to CAH card format, adjusting verb tenses as needed.

Format as JSON array:
[
  {"text": "Card text", "type": "white", "blanks": 0}
]
```

### Step 4: AI Generation
Model reads article and extracts humor.

### Step 5: Parsing (Same as Topic-Based)
- Extract JSON
- Force card type if specified
- Save cards

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │ Topic: "Gaming"  │   OR    │ Article: [pasted text...]    │  │
│  │ Type: White      │         │ Type: Both                   │  │
│  │ Qty: 10          │         │ Qty: 8                       │  │
│  │ Style: Nerdy     │         │                              │  │
│  └──────────────────┘         └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PROMPT ASSEMBLY (Programmatic)                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SYSTEM PROMPT (Hidden - based on style/mode)            │    │
│  │ "You are a Cards Against Humanity card writer..."       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              +                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ USER PROMPT (Your input + hidden formatting rules)      │    │
│  │ "Generate 10 WHITE cards about: Gaming                  │    │
│  │  Format as JSON array: [{"text":..., "type":...}]"      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI GENERATION                               │
│                                                                  │
│  Your loaded chat model (e.g., Mistral, Dolphin) processes      │
│  the combined prompts and generates card text.                  │
│                                                                  │
│  Output: JSON array of cards                                    │
│  [{"text": "Rage-quitting during the tutorial", "type": "white"}]│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PARSING (Programmatic)                         │
│                                                                  │
│  1. Extract JSON from AI response                               │
│  2. Parse each card object                                      │
│  3. FORCE card type if user selected White/Black only           │
│     (Ignores what AI labeled them as!)                          │
│  4. Assign unique ID and timestamp                              │
│  5. Save to data/cah/generated/cards.json                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DISPLAY & STORAGE                           │
│                                                                  │
│  ⬜ Rage-quitting during the tutorial                           │
│  ⬜ "It's not a bug, it's a feature"                            │
│  ⬜ Microtransactions for basic emotions                        │
│                                                                  │
│  Saved to: data/cah/generated/cards.json                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Component | Programmatic | AI-Powered | Hidden from User |
|-----------|:------------:|:----------:|:----------------:|
| Style selection dropdown | ✅ | | |
| System prompt (style instructions) | ✅ | | ✅ |
| User prompt formatting | ✅ | | ✅ |
| JSON format instructions | ✅ | | ✅ |
| Actual card text generation | | ✅ | |
| Response parsing | ✅ | | |
| Card type forcing | ✅ | | |
| Card saving/loading | ✅ | | |
| Delete/export | ✅ | | |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/cards/cah_generator.py` | Core logic, prompts, parsing |
| `web_app_multi.py` | UI and generation functions |
| `config/cah_presets.yaml` | Configuration settings |
| `data/cah/generated/cards.json` | Saved cards |

---

## Why Hidden Prompts?

The hidden system prompts serve to:
1. **Ensure consistent output format** - AI returns parseable JSON
2. **Set the creative tone** - Each style has specific instructions
3. **Guide card structure** - Rules for blanks, length, etc.
4. **Improve quality** - "Don't be boring or generic"

Without these, the AI might return cards in random formats, inconsistent styles, or ignore CAH conventions entirely.
