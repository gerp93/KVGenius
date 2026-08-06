"""Applies VisualAssault (https://github.com/gerp93/VisualAssault) color
themes to the KVGenius Flet UI. Thin wrapper matching the old
flet_kvg_themes API shape (get_theme/get_theme_list/theme_exists) so
desktop_app.py and ui/state.py didn't need to change beyond the import.
"""
import flet as ft
from visual_assault_flet import THEMES

DEFAULT_THEME_ID = "hacker"


def theme_exists(theme_id: str) -> bool:
    return theme_id in THEMES


def get_theme_background(theme_id: str) -> str:
    """The theme's page-background hex, for page.bgcolor (see get_theme's
    docstring for why this isn't folded into the ft.ColorScheme itself)."""
    return THEMES[theme_id]["background"]


def get_theme_list() -> list[tuple[str, str]]:
    """Return [(theme_id, display_name), ...] for populating a dropdown."""
    return [(theme_id, data["name"]) for theme_id, data in THEMES.items()]


def get_theme(theme_id: str):
    """Build a flet Theme from a VisualAssault theme id, or None if unknown.

    ft.ColorScheme has no "background" field on the installed flet version
    (0.86.x uses Material 3's surface-based scheme, not the older
    background/on_background pair VisualAssault's README example assumes —
    see that package's README for the caveat that fields vary by flet
    version). Callers that want the page background to follow the theme
    too should additionally set `page.bgcolor = THEMES[theme_id]["background"]`.
    """
    theme = THEMES.get(theme_id)
    if theme is None:
        return None
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=theme["primary"],
            surface=theme["surface"],
            on_surface=theme["foreground"],
            outline=theme["border"],
            error=theme["accentRed"],
        )
    )
