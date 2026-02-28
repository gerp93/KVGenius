"""
Unit tests for src/cards/cah_generator.py
"""
import json
import os
import pytest

from src.cards.cah_generator import (
    Card,
    CardType,
    CardStyle,
    CAHGenerator,
    get_generation_prompt,
    get_article_extraction_prompt,
    EXAMPLE_CARDS,
    SYSTEM_PROMPTS,
)


# ---------------------------------------------------------------------------
# Card dataclass
# ---------------------------------------------------------------------------

class TestCard:
    def test_auto_id_generated(self):
        card = Card(id="", text="Hello", card_type="white", style="classic", topic="test")
        assert card.id != ""

    def test_explicit_id_preserved(self):
        card = Card(id="myid", text="Hi", card_type="black", style="classic", topic="test")
        assert card.id == "myid"

    def test_created_at_auto_set(self):
        card = Card(id="", text="x", card_type="white", style="classic", topic="t")
        assert card.created_at != ""

    def test_blanks_default(self):
        card = Card(id="", text="text", card_type="white", style="classic", topic="t")
        assert card.blanks == 0

    def test_favorited_default_false(self):
        card = Card(id="", text="text", card_type="white", style="classic", topic="t")
        assert card.favorited is False

    def test_exported_default_false(self):
        card = Card(id="", text="text", card_type="white", style="classic", topic="t")
        assert card.exported is False


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------

class TestEnums:
    def test_card_type_values(self):
        assert CardType.BLACK.value == "black"
        assert CardType.WHITE.value == "white"
        assert CardType.BOTH.value == "both"

    def test_card_style_values(self):
        assert CardStyle.CLASSIC.value == "classic"
        assert CardStyle.ABSURD.value == "absurd"
        assert CardStyle.DARK.value == "dark"
        assert CardStyle.WHOLESOME.value == "wholesome"
        assert CardStyle.NERDY.value == "nerdy"
        assert CardStyle.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# CAHGenerator – basic CRUD
# ---------------------------------------------------------------------------

@pytest.fixture()
def generator(tmp_path):
    return CAHGenerator(data_dir=str(tmp_path / "cah"))


class TestCAHGeneratorCRUD:
    def test_initial_cards_empty(self, generator):
        assert generator.get_cards() == []

    def test_add_cards(self, generator):
        cards = [
            Card(id="", text="Test card", card_type="white", style="classic", topic="test")
        ]
        added = generator.add_cards(cards)
        assert added == 1
        assert len(generator.get_cards()) == 1

    def test_get_cards_filter_by_type(self, generator):
        generator.add_cards([
            Card(id="", text="Black card ___", card_type="black", style="classic", topic="t", blanks=1),
            Card(id="", text="White card", card_type="white", style="classic", topic="t"),
        ])
        blacks = generator.get_cards(card_type="black")
        whites = generator.get_cards(card_type="white")
        assert len(blacks) == 1
        assert len(whites) == 1

    def test_get_cards_filter_by_topic(self, generator):
        generator.add_cards([
            Card(id="", text="A", card_type="white", style="classic", topic="cats"),
            Card(id="", text="B", card_type="white", style="classic", topic="dogs"),
        ])
        cats = generator.get_cards(topic="cats")
        assert len(cats) == 1
        assert cats[0].topic == "cats"

    def test_get_cards_favorited_only(self, generator):
        card = Card(id="fav1", text="Fav", card_type="white", style="classic", topic="t")
        generator.add_cards([card])
        assert generator.get_cards(favorited_only=True) == []
        generator.toggle_favorite("fav1")
        favs = generator.get_cards(favorited_only=True)
        assert len(favs) == 1

    def test_toggle_favorite(self, generator):
        card = Card(id="c1", text="X", card_type="white", style="classic", topic="t")
        generator.add_cards([card])
        result = generator.toggle_favorite("c1")
        assert result is True
        result2 = generator.toggle_favorite("c1")
        assert result2 is False

    def test_toggle_favorite_nonexistent(self, generator):
        assert generator.toggle_favorite("nope") is False

    def test_delete_card(self, generator):
        card = Card(id="del1", text="Y", card_type="white", style="classic", topic="t")
        generator.add_cards([card])
        result = generator.delete_card("del1")
        assert result is True
        assert generator.get_cards() == []

    def test_delete_nonexistent_card(self, generator):
        assert generator.delete_card("ghost") is False

    def test_update_card_text(self, generator):
        card = Card(id="upd1", text="Old text ___", card_type="black", style="classic", topic="t", blanks=1)
        generator.add_cards([card])
        result = generator.update_card_text("upd1", "New text ___ and ___")
        assert result is True
        cards = generator.get_cards()
        assert cards[0].text == "New text ___ and ___"
        assert cards[0].blanks == 2

    def test_update_nonexistent_card_text(self, generator):
        assert generator.update_card_text("no_card", "text") is False

    def test_toggle_exported(self, generator):
        card = Card(id="exp1", text="Z", card_type="white", style="classic", topic="t")
        generator.add_cards([card])
        assert generator.toggle_exported("exp1") is True
        assert generator.toggle_exported("exp1") is False

    def test_set_exported(self, generator):
        card = Card(id="se1", text="Q", card_type="white", style="classic", topic="t")
        generator.add_cards([card])
        assert generator.set_exported("se1", True) is True
        cards = generator.get_cards()
        assert cards[0].exported is True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestCAHGeneratorPersistence:
    def test_cards_saved_and_reloaded(self, tmp_path):
        gen1 = CAHGenerator(data_dir=str(tmp_path / "cah"))
        gen1.add_cards([Card(id="p1", text="Saved", card_type="white", style="classic", topic="t")])
        gen2 = CAHGenerator(data_dir=str(tmp_path / "cah"))
        cards = gen2.get_cards()
        assert len(cards) == 1
        assert cards[0].text == "Saved"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_empty(self, generator):
        stats = generator.get_stats()
        assert stats["total"] == 0
        assert stats["black_cards"] == 0
        assert stats["white_cards"] == 0
        assert stats["favorited"] == 0

    def test_stats_populated(self, generator):
        generator.add_cards([
            Card(id="b1", text="B ___", card_type="black", style="classic", topic="t", blanks=1),
            Card(id="w1", text="W", card_type="white", style="classic", topic="t"),
            Card(id="w2", text="W2", card_type="white", style="nerdy", topic="t"),
        ])
        generator.toggle_favorite("w1")
        stats = generator.get_stats()
        assert stats["total"] == 3
        assert stats["black_cards"] == 1
        assert stats["white_cards"] == 2
        assert stats["favorited"] == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportCards:
    def test_export_json(self, generator):
        generator.add_cards([Card(id="e1", text="A", card_type="white", style="classic", topic="t")])
        output = generator.export_cards(format="json")
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["text"] == "A"

    def test_export_txt(self, generator):
        generator.add_cards([
            Card(id="b1", text="Prompt ___", card_type="black", style="classic", topic="t", blanks=1),
            Card(id="w1", text="Answer", card_type="white", style="classic", topic="t"),
        ])
        output = generator.export_cards(format="txt")
        assert "Prompt ___" in output
        assert "Answer" in output

    def test_export_csv(self, generator):
        generator.add_cards([Card(id="c1", text="CSV card", card_type="white", style="classic", topic="t")])
        output = generator.export_cards(format="csv")
        assert "CSV card" in output
        assert "Type" in output  # header row

    def test_export_specific_ids(self, generator):
        generator.add_cards([
            Card(id="x1", text="One", card_type="white", style="classic", topic="t"),
            Card(id="x2", text="Two", card_type="white", style="classic", topic="t"),
        ])
        output = generator.export_cards(card_ids=["x1"], format="json")
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["id"] == "x1"

    def test_export_unknown_format_returns_empty(self, generator):
        output = generator.export_cards(format="xml")
        assert output == ""


# ---------------------------------------------------------------------------
# parse_generated_cards
# ---------------------------------------------------------------------------

class TestParseGeneratedCards:
    def test_parse_valid_json(self, generator):
        response = json.dumps([
            {"text": "Card text", "type": "white", "blanks": 0},
        ])
        cards = generator.parse_generated_cards(response, topic="test", style=CardStyle.CLASSIC)
        assert len(cards) == 1
        assert cards[0].text == "Card text"

    def test_parse_markdown_wrapped_json(self, generator):
        response = "```json\n[{\"text\": \"Wrapped\", \"type\": \"white\", \"blanks\": 0}]\n```"
        cards = generator.parse_generated_cards(response, topic="t", style=CardStyle.CLASSIC)
        assert len(cards) == 1
        assert cards[0].text == "Wrapped"

    def test_force_type_white(self, generator):
        response = json.dumps([
            {"text": "Should be white", "type": "black", "blanks": 1},
        ])
        cards = generator.parse_generated_cards(
            response, topic="t", style=CardStyle.CLASSIC, force_type=CardType.WHITE
        )
        assert cards[0].card_type == "white"

    def test_parse_freeform_response(self, generator):
        response = "- Funny white card\n- Another white card"
        cards = generator.parse_generated_cards(response, topic="t", style=CardStyle.CLASSIC)
        assert len(cards) >= 1

    def test_parse_empty_text_skipped(self, generator):
        response = json.dumps([{"text": "", "type": "white", "blanks": 0}])
        cards = generator.parse_generated_cards(response, topic="t", style=CardStyle.CLASSIC)
        assert cards == []


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

class TestSystemPrompts:
    def test_get_system_prompt_classic(self, generator):
        prompt = generator.get_system_prompt(CardStyle.CLASSIC)
        assert "BLACK cards" in prompt or "black" in prompt.lower()

    def test_get_system_prompt_custom(self, generator):
        prompt = generator.get_system_prompt(CardStyle.CUSTOM, custom_style="Pirate style")
        assert "Pirate style" in prompt

    def test_get_system_prompt_custom_no_instructions_falls_back(self, generator):
        prompt = generator.get_system_prompt(CardStyle.CUSTOM, custom_style="")
        # Falls back to CLASSIC
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestGetGenerationPrompt:
    def test_black_card_prompt(self):
        p = get_generation_prompt("cats", CardType.BLACK, 5, CardStyle.CLASSIC)
        assert "BLACK" in p
        assert "cats" in p

    def test_white_card_prompt(self):
        p = get_generation_prompt("dogs", CardType.WHITE, 3, CardStyle.CLASSIC)
        assert "WHITE" in p
        assert "dogs" in p

    def test_both_card_prompt(self):
        p = get_generation_prompt("tech", CardType.BOTH, 4, CardStyle.CLASSIC)
        assert "BLACK" in p
        assert "WHITE" in p

    def test_custom_instructions_appended(self):
        p = get_generation_prompt("x", CardType.WHITE, 1, CardStyle.CLASSIC, custom_instructions="be funny")
        assert "be funny" in p


class TestGetArticleExtractionPrompt:
    def test_contains_article_text(self):
        p = get_article_extraction_prompt("The funny article.", CardType.WHITE, 3)
        assert "The funny article." in p

    def test_white_instruction(self):
        p = get_article_extraction_prompt("article", CardType.WHITE, 2)
        assert "WHITE" in p

    def test_black_instruction(self):
        p = get_article_extraction_prompt("article", CardType.BLACK, 2)
        assert "BLACK" in p


# ---------------------------------------------------------------------------
# Example cards sanity
# ---------------------------------------------------------------------------

class TestExampleCards:
    def test_example_cards_count(self):
        assert len(EXAMPLE_CARDS) == 6

    def test_example_cards_have_ids(self):
        for card in EXAMPLE_CARDS:
            assert card.id != ""
