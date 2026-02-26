"""
Cards Against Humanity style card generator.

This module handles the generation of CAH-style cards using LLMs.
Supports both black cards (prompts with blanks) and white cards (answers).
"""

import os
import json
import random
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


class CardType(Enum):
    """Type of card to generate."""
    BLACK = "black"  # Prompt cards with blanks (___)
    WHITE = "white"  # Answer cards
    BOTH = "both"    # Generate both types


class CardStyle(Enum):
    """Style/tone of the cards."""
    CLASSIC = "classic"      # Classic CAH crude humor
    ABSURD = "absurd"        # Surreal, weird humor
    DARK = "dark"            # Dark/morbid humor
    WHOLESOME = "wholesome"  # Family-friendly version
    NERDY = "nerdy"          # Pop culture, tech, gaming references
    CUSTOM = "custom"        # User-defined style


@dataclass
class Card:
    """Represents a single CAH-style card."""
    id: str
    text: str
    card_type: str  # "black" or "white"
    style: str
    topic: str
    blanks: int = 0  # Number of blanks for black cards
    created_at: str = ""
    favorited: bool = False
    exported: bool = False  # True if card has been exported to external deck
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = f"{self.card_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"


# System prompts for different card generation modes
SYSTEM_PROMPTS = {
    CardStyle.CLASSIC: """You are a Cards Against Humanity card writer. Your job is to create hilarious, irreverent, and often crude cards that push boundaries while being clever.

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

Be creative, provocative, and funny. Don't be boring or generic.""",

    CardStyle.ABSURD: """You are a surrealist comedy writer creating Cards Against Humanity style cards. Your specialty is bizarre, nonsensical, and dreamlike humor.

Rules for BLACK cards:
- Use "___" for blanks
- Create weird, surreal scenarios
- Embrace non-sequiturs and absurdity
- Think Salvador Dali meets stand-up comedy

Rules for WHITE cards:
- Embrace the bizarre and unexpected
- Mix mundane with fantastical
- The weirder the better

Make it strange, funny, and memorable.""",

    CardStyle.DARK: """You are writing Cards Against Humanity cards with a dark comedy edge. Think gallows humor, morbid jokes, and uncomfortable truths delivered with wit.

Rules for BLACK cards:
- Use "___" for blanks
- Explore dark themes with clever wordplay
- Find humor in uncomfortable topics
- Be shocking but smart

Rules for WHITE cards:
- Dark concepts and references
- Morbid but clever
- Unexpected dark twists

Dark doesn't mean mean-spirited. It means finding light in darkness through comedy.""",

    CardStyle.WHOLESOME: """You are creating a family-friendly version of Cards Against Humanity. Keep it clean but still funny and clever.

Rules for BLACK cards:
- Use "___" for blanks
- Family-friendly scenarios
- Wordplay and puns welcome
- Safe for all ages

Rules for WHITE cards:
- Clean, clever answers
- Pop culture references (family-friendly)
- Silly and fun

Be creative and funny without any adult content.""",

    CardStyle.NERDY: """You are creating Cards Against Humanity cards for geeks, gamers, and tech enthusiasts. Reference games, sci-fi, fantasy, programming, and internet culture.

Rules for BLACK cards:
- Use "___" for blanks
- Reference video games, D&D, sci-fi, fantasy, coding, etc.
- Inside jokes for nerds
- Mix highbrow and lowbrow geek humor

Rules for WHITE cards:
- Gaming references
- Tech terminology used humorously
- Sci-fi and fantasy concepts
- Internet culture and memes

Embrace the nerd. Be clever. Know your references.""",
}

# System prompt for article extraction mode
ARTICLE_EXTRACTION_PROMPT = """You are an expert at extracting humor from articles and converting them into Cards Against Humanity style cards.

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
- Keep the original joke's spirit but make it work in CAH format
- Remove article-specific context that wouldn't make sense as a card
- Convert to present tense or gerunds: "He was eating the cake" → "Eating the cake" or "The man ran" → "Running"
- Make it timeless when possible (remove dated references unless they're the joke)

Be creative in your extraction - find the hidden card potential in every joke!"""


def get_article_extraction_prompt(article_text: str, card_type: CardType, quantity: int) -> str:
    """Build the user prompt for extracting cards from an article."""
    
    type_instruction = ""
    if card_type == CardType.BLACK:
        type_instruction = f"Extract and create {quantity} BLACK cards (prompts with ___ blanks)"
    elif card_type == CardType.WHITE:
        type_instruction = f"Extract and create {quantity} WHITE cards (answer cards)"
    else:
        half = quantity // 2
        type_instruction = f"Extract and create {half} BLACK cards and {quantity - half} WHITE cards"
    
    prompt = f"""{type_instruction} from the following article:

---ARTICLE START---
{article_text}
---ARTICLE END---

Instructions:
1. Find the funniest, most absurd, or most memorable lines
2. Convert them into CAH-style cards
3. Adjust grammar/tense to work as standalone cards
4. Keep the humor but make it context-independent

Format your response as a JSON array with objects containing:
- "text": the card text (converted for CAH format)
- "type": "black" or "white"
- "blanks": number of ___ in the card (0 for white cards)
- "original": brief note about what inspired this card (optional)

Example conversions:
- Article: "The man spent 47 years perfecting his signature" → White card: "Spending 47 years perfecting your signature"
- Article: "What could possibly go wrong?" → Black card: "What could possibly go wrong with ___?"
- Article: "an unsettling amount of mayonnaise" → White card: "An unsettling amount of mayonnaise"

Generate exactly {quantity} cards. Be creative and preserve the original humor!"""

    return prompt


def get_generation_prompt(topic: str, card_type: CardType, quantity: int, style: CardStyle, custom_instructions: str = "") -> str:
    """Build the user prompt for card generation."""
    
    type_instruction = ""
    if card_type == CardType.BLACK:
        type_instruction = f"Generate {quantity} BLACK cards (prompt cards)"
        type_rules = """BLACK CARD RULES:
- BLACK cards are PROMPTS that need answers
- MUST contain "___" (three underscores) as a blank for answers
- Questions ending with ? should ALSO have a blank, e.g. "What does ___ fear most?"
- Can be statements with blanks like "___ is the reason I drink."
- 1-3 blanks per card"""
    elif card_type == CardType.WHITE:
        type_instruction = f"Generate {quantity} WHITE cards (answer cards) ONLY"
        type_rules = """WHITE CARD RULES - IMPORTANT:
- WHITE cards are SHORT ANSWER PHRASES ONLY (2-8 words)
- NEVER generate questions - NO question marks allowed
- NO blanks (no ___)
- Just nouns, phrases, concepts, or funny things
- Examples: "A disappointing handjob", "Grandma's dentures", "Elon Musk's ego"
- DO NOT include any setup questions, ONLY the answer phrases"""
    else:
        half = quantity // 2
        type_instruction = f"Generate {half} BLACK cards and {quantity - half} WHITE cards"
        type_rules = """BLACK CARD RULES: Prompts with ___ blanks (questions should also have blanks)
WHITE CARD RULES: Short answers (2-8 words), NO blanks, NO questions"""
    
    prompt = f"""{type_instruction} about: {topic}

{type_rules}

OUTPUT FORMAT - JSON array only, no extra text:
[
  {{"text": "card text here", "type": "{card_type.value if card_type != CardType.BOTH else 'black'}", "blanks": 0}}
]

Generate exactly {quantity} cards. Output ONLY the JSON array."""

    if custom_instructions:
        prompt += f"\n\nAdditional style: {custom_instructions}"
    
    return prompt


class CAHGenerator:
    """Cards Against Humanity style card generator using LLMs."""
    
    def __init__(self, data_dir: str = "data/cah"):
        self.data_dir = data_dir
        self.generated_dir = os.path.join(data_dir, "generated")
        self.cards_file = os.path.join(self.generated_dir, "cards.json")
        self.cards: List[Card] = []
        
        # Ensure directories exist
        os.makedirs(self.generated_dir, exist_ok=True)
        
        # Load existing cards
        self._load_cards()
    
    def _load_cards(self):
        """Load saved cards from disk."""
        if os.path.exists(self.cards_file):
            try:
                with open(self.cards_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cards = [Card(**card) for card in data]
                logger.info(f"Loaded {len(self.cards)} saved cards")
            except Exception as e:
                logger.error(f"Failed to load cards: {e}")
                self.cards = []
        else:
            self.cards = []
    
    def _save_cards(self):
        """Save all cards to disk."""
        try:
            with open(self.cards_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(c) for c in self.cards], f, indent=2)
            logger.info(f"Saved {len(self.cards)} cards")
        except Exception as e:
            logger.error(f"Failed to save cards: {e}")
    
    def get_system_prompt(self, style: CardStyle, custom_style: str = "") -> str:
        """Get the system prompt for a given style."""
        if style == CardStyle.CUSTOM and custom_style:
            return f"""You are a Cards Against Humanity card writer with a custom style.

Style instructions: {custom_style}

Rules for BLACK cards (prompts):
- Use "___" to indicate blanks where white cards go
- Can have 1-3 blanks

Rules for WHITE cards (answers):
- Short phrases (2-8 words typically)
- Should work as punchlines for black cards

Format output as JSON array."""
        
        return SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS[CardStyle.CLASSIC])
    
    def parse_generated_cards(self, response: str, topic: str, style: CardStyle, force_type: Optional[CardType] = None) -> List[Card]:
        """Parse LLM response into Card objects.
        
        Args:
            response: The LLM response text
            topic: Topic/source for the cards
            style: Card style
            force_type: If set to BLACK or WHITE, force all cards to that type
        """
        cards = []
        
        try:
            # Try to extract JSON from response
            # Handle cases where the model wraps JSON in markdown code blocks
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            if isinstance(data, list):
                for item in data:
                    # Determine card type - respect force_type if set
                    if force_type and force_type != CardType.BOTH:
                        card_type = force_type.value
                        blanks = 0  # No blanks for forced white cards
                    else:
                        card_type = item.get("type", "white")
                        blanks = item.get("blanks", 0)
                    
                    card = Card(
                        id="",
                        text=item.get("text", ""),
                        card_type=card_type,
                        style=style.value,
                        topic=topic,
                        blanks=blanks
                    )
                    if card.text:
                        cards.append(card)
            
            logger.info(f"Parsed {len(cards)} cards from response")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to salvage cards from non-JSON response
            cards = self._parse_freeform_response(response, topic, style, force_type)
        
        return cards
    
    def _parse_freeform_response(self, response: str, topic: str, style: CardStyle, force_type: Optional[CardType] = None) -> List[Card]:
        """Attempt to parse cards from non-JSON response."""
        cards = []
        lines = response.strip().split('\n')
        
        # Phrases that indicate model commentary, not actual cards
        skip_phrases = [
            'i have generated', 'here are', 'here they are', 'as requested',
            'json format', 'following cards', 'cards about', 'hope you',
            'let me know', 'enjoy', 'here\'s', 'note:', 'disclaimer'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Skip model commentary
            line_lower = line.lower()
            if any(phrase in line_lower for phrase in skip_phrases):
                continue
            
            # Try to parse line as JSON object (model may output one JSON per line)
            if line.startswith('{') and '"text"' in line:
                try:
                    # Clean up trailing comma if present
                    clean_line = line.rstrip(',')
                    item = json.loads(clean_line)
                    text = item.get('text', '')
                    if text:
                        # Skip questions if we're generating WHITE cards only
                        if force_type == CardType.WHITE and '?' in text:
                            logger.debug(f"Skipping question in WHITE mode: {text}")
                            continue
                        
                        # Determine card type
                        if force_type and force_type != CardType.BOTH:
                            card_type = force_type.value
                            blanks = 0 if force_type == CardType.WHITE else text.count('___')
                        else:
                            card_type = item.get('type', 'white')
                            blanks = item.get('blanks', text.count('___'))
                        
                        card = Card(
                            id="",
                            text=text,
                            card_type=card_type,
                            style=style.value,
                            topic=topic,
                            blanks=blanks
                        )
                        cards.append(card)
                        continue
                except json.JSONDecodeError:
                    pass  # Fall through to regular parsing
            
            # Remove common prefixes
            for prefix in ['- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ', 
                          '6. ', '7. ', '8. ', '9. ', '10. ']:
                if line.startswith(prefix):
                    line = line[len(prefix):]
            
            # Skip JSON-like fragments that didn't parse
            if line.startswith('{') or line.startswith('[') or line.startswith(']'):
                continue
            
            # Skip questions if we're generating WHITE cards only
            if force_type == CardType.WHITE and '?' in line:
                logger.debug(f"Skipping question in WHITE mode: {line}")
                continue
            
            # Determine card type - respect force_type if set
            if force_type and force_type != CardType.BOTH:
                card_type = force_type.value
                blanks = 0 if force_type == CardType.WHITE else line.count('___')
            else:
                # Auto-detect by presence of blanks
                blanks = line.count('___')
                card_type = "black" if blanks > 0 else "white"
            
            if len(line) > 2:  # Skip very short lines
                card = Card(
                    id="",
                    text=line,
                    card_type=card_type,
                    style=style.value,
                    topic=topic,
                    blanks=blanks
                )
                cards.append(card)
        
        logger.info(f"Parsed {len(cards)} cards from freeform response")
        return cards
    
    def add_cards(self, cards: List[Card]) -> int:
        """Add generated cards to the collection and save."""
        self.cards.extend(cards)
        self._save_cards()
        return len(cards)
    
    def get_cards(self, 
                  card_type: Optional[str] = None, 
                  style: Optional[str] = None,
                  topic: Optional[str] = None,
                  favorited_only: bool = False) -> List[Card]:
        """Get cards with optional filtering."""
        filtered = self.cards
        
        if card_type:
            filtered = [c for c in filtered if c.card_type == card_type]
        if style:
            filtered = [c for c in filtered if c.style == style]
        if topic:
            filtered = [c for c in filtered if topic.lower() in c.topic.lower()]
        if favorited_only:
            filtered = [c for c in filtered if c.favorited]
        
        return filtered
    
    def toggle_favorite(self, card_id: str) -> bool:
        """Toggle favorite status of a card."""
        for card in self.cards:
            if card.id == card_id:
                card.favorited = not card.favorited
                self._save_cards()
                return card.favorited
        return False
    
    def delete_card(self, card_id: str) -> bool:
        """Delete a card by ID."""
        for i, card in enumerate(self.cards):
            if card.id == card_id:
                del self.cards[i]
                self._save_cards()
                return True
        return False
    
    def update_card_text(self, card_id: str, new_text: str) -> bool:
        """Update the text of a card by ID."""
        for card in self.cards:
            if card.id == card_id:
                card.text = new_text
                # Update blanks count for black cards
                if card.card_type == "black":
                    card.blanks = new_text.count("___")
                self._save_cards()
                return True
        return False
    
    def toggle_exported(self, card_id: str) -> bool:
        """Toggle exported status of a card."""
        for card in self.cards:
            if card.id == card_id:
                card.exported = not card.exported
                self._save_cards()
                return card.exported
        return False
    
    def set_exported(self, card_id: str, exported: bool) -> bool:
        """Set exported status of a card."""
        for card in self.cards:
            if card.id == card_id:
                card.exported = exported
                self._save_cards()
                return True
        return False
    
    def export_cards(self, card_ids: Optional[List[str]] = None, format: str = "json") -> str:
        """Export cards to various formats."""
        cards_to_export = self.cards
        if card_ids:
            cards_to_export = [c for c in self.cards if c.id in card_ids]
        
        if format == "json":
            return json.dumps([asdict(c) for c in cards_to_export], indent=2)
        
        elif format == "txt":
            lines = []
            lines.append("=== BLACK CARDS (Prompts) ===\n")
            for c in cards_to_export:
                if c.card_type == "black":
                    lines.append(f"• {c.text}")
            lines.append("\n=== WHITE CARDS (Answers) ===\n")
            for c in cards_to_export:
                if c.card_type == "white":
                    lines.append(f"• {c.text}")
            return "\n".join(lines)
        
        elif format == "csv":
            import csv
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["Type", "Text", "Blanks", "Style", "Topic", "Created"])
            for c in cards_to_export:
                writer.writerow([c.card_type, c.text, c.blanks, c.style, c.topic, c.created_at])
            return output.getvalue()
        
        return ""
    
    def get_stats(self) -> Dict:
        """Get statistics about the card collection."""
        black_cards = [c for c in self.cards if c.card_type == "black"]
        white_cards = [c for c in self.cards if c.card_type == "white"]
        
        styles = {}
        topics = {}
        for c in self.cards:
            styles[c.style] = styles.get(c.style, 0) + 1
            topics[c.topic] = topics.get(c.topic, 0) + 1
        
        return {
            "total": len(self.cards),
            "black_cards": len(black_cards),
            "white_cards": len(white_cards),
            "favorited": len([c for c in self.cards if c.favorited]),
            "styles": styles,
            "topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10])
        }


# Example cards for seeding/testing
EXAMPLE_CARDS = [
    Card(id="ex1", text="What's that smell?", card_type="black", style="classic", topic="example", blanks=0),
    Card(id="ex2", text="I drink to forget ___.", card_type="black", style="classic", topic="example", blanks=1),
    Card(id="ex3", text="___ + ___ = profit.", card_type="black", style="classic", topic="example", blanks=2),
    Card(id="ex4", text="Aggressive hand gestures", card_type="white", style="classic", topic="example", blanks=0),
    Card(id="ex5", text="A disappointing birthday party", card_type="white", style="classic", topic="example", blanks=0),
    Card(id="ex6", text="Pretending to know what you're doing", card_type="white", style="classic", topic="example", blanks=0),
]
