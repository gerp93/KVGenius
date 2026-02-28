"""
Chat Tab - chat interface with character/persona management
"""
from typing import Optional, Callable
from pathlib import Path
import logging
import threading

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, TextField, ElevatedButton,
    IconButton, ListView, Colors, Icons, MainAxisAlignment, ScrollMode,
    FontWeight, padding, border_radius, SnackBar, ProgressBar,
    Slider, Dropdown, dropdown, Tabs, Tab, TextButton, CrossAxisAlignment,
    ButtonStyle, ControlState, alignment,
)

from ..state import app_state

logger = logging.getLogger(__name__)


class ChatTab:
    """Chat interface with character/persona management - uses the model loaded from header."""
    
    def __init__(self, page: Page):
        self.page = page
        self._init_db()
        self.selected_character = None  # Current selected character dict
        self.selected_persona = None  # Current selected persona dict
        self.editing_character_id = None  # ID of character being edited (None = new)
        self.editing_persona_id = None  # ID of persona being edited (None = new)
        self._build_ui()
    
    def _init_db(self):
        """Initialize the database connection."""
        try:
            from src.database.chat_history import ChatHistoryDB
            self.db = ChatHistoryDB(db_path="./data/chat_history.db")
        except ImportError:
            logger.warning("ChatHistoryDB not available, using mock")
            self.db = MockChatDB()
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _build_ui(self):
        """Build all UI components for the chat tab with sub-tabs."""
        # =========== CONVERSATION TAB COMPONENTS ===========
        
        # Model status (shows what's loaded from header)
        self.model_status = Text("💡 Load a 💬 Chat model from the header dropdown", size=12, color=Colors.GREY_500)
        
        # Character dropdown
        self.char_dropdown = Dropdown(
            label="AI Character",
            width=240,
            options=self._get_character_options(),
            value="none",
            on_change=self._on_character_change,
        )
        
        # Persona dropdown
        self.persona_dropdown = Dropdown(
            label="User Persona",
            width=240,
            options=self._get_persona_options(),
            value="none",
            on_change=self._on_persona_change,
        )
        
        # System prompt (hidden when character is selected)
        self.system_prompt = TextField(
            label="System Prompt",
            value="You are a helpful AI assistant.",
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        
        # Generation parameters with labels
        self.temp_value_text = Text("", size=10, width=55)
        self.temp_slider = Slider(
            min=0.1, max=1.5, value=0.7,
            divisions=14,
            width=130,
            on_change=lambda e: self._update_slider_label(self.temp_value_text, "Temp {:.1f}", self.temp_slider.value),
        )
        
        self.max_tokens_value_text = Text("", size=10, width=65)
        self.max_tokens_slider = Slider(
            min=64, max=512, value=256,
            divisions=7,
            width=130,
            on_change=lambda e: self._update_slider_label(self.max_tokens_value_text, "{} tokens", int(self.max_tokens_slider.value)),
        )
        
        # Initialize Chat tab slider labels
        self._update_slider_label(self.temp_value_text, "Temp {:.1f}", self.temp_slider.value)
        self._update_slider_label(self.max_tokens_value_text, "{} tokens", int(self.max_tokens_slider.value))
        
        # Chat messages list
        self.chat_messages = ListView(
            spacing=10,
            padding=padding.all(10),
            expand=True,
            auto_scroll=True,
        )
        
        # Message input with Enter to send (shift_enter makes Enter submit)
        self.message_input = TextField(
            label="Type your message... (Enter to send, Shift+Enter for newline)",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
            shift_enter=True,
            on_submit=self._send_message,
        )
        
        self.send_btn = ElevatedButton(
            "Send",
            icon=Icons.SEND,
            on_click=self._send_message,
            style=ButtonStyle(
                bgcolor={
                    ControlState.DEFAULT: Colors.BLUE_700,
                    ControlState.DISABLED: Colors.GREY_700,
                },
                color={
                    ControlState.DEFAULT: Colors.WHITE,
                    ControlState.DISABLED: Colors.GREY_500,
                },
                overlay_color=Colors.TRANSPARENT,
            ),
        )
        
        self.clear_btn = IconButton(
            icon=Icons.DELETE_SWEEP,
            tooltip="Clear conversation",
            on_click=self._clear_conversation,
        )
        
        # Typing indicator (positioned above message input)
        self.typing_indicator = Row([
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=Colors.GREEN_400),
            Text("AI is typing...", size=11, color=Colors.GREEN_400, italic=True),
        ], spacing=8, visible=False)
        
        # No model banner
        self.no_model_banner = Container(
            content=Row([
                ft.Icon(Icons.WARNING_AMBER_ROUNDED, size=24, color=Colors.AMBER_400),
                Text("💬 No chat model loaded! Load a model from the header dropdown to chat.", 
                     size=14, color=Colors.AMBER_300),
            ], spacing=10, alignment=MainAxisAlignment.CENTER),
            bgcolor=Colors.with_opacity(0.3, Colors.AMBER_900),
            padding=padding.all(12),
            border_radius=border_radius.all(8),
            visible=True,  # Show by default until model loads
        )
        
        # Progress indicator
        self.progress = ProgressBar(visible=False)
        self.generating = False
        
        # =========== CHARACTERS TAB COMPONENTS ===========
        
        # Character list for browsing
        self.char_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Character edit form
        self.char_name_input = TextField(label="Character Name", hint_text="e.g., Friendly Tutor")
        self.char_desc_input = TextField(label="Short Description", hint_text="Brief dropdown description")
        self.char_system_input = TextField(
            label="System Prompt (Personality)",
            hint_text="You are a helpful tutor who explains things clearly...",
            multiline=True,
            min_lines=5,
            max_lines=10,
        )
        self.char_avatar_input = TextField(label="Avatar Emoji", value="🤖", width=100)
        
        # Character editor sliders with labels
        self.char_temp_value_text = Text("", size=11)
        self.char_temp_slider = Slider(min=0.1, max=1.5, value=0.7, divisions=14,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_temp_value_text, "Temp {:.1f}", self.char_temp_slider.value))
        
        self.char_topp_value_text = Text("", size=11)
        self.char_topp_slider = Slider(min=0.1, max=1.0, value=0.95, divisions=18,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_topp_value_text, "Top-P {:.2f}", self.char_topp_slider.value))
        
        self.char_topk_value_text = Text("", size=11)
        self.char_topk_slider = Slider(min=1, max=100, value=50, divisions=99,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.char_topk_value_text, "Top-K {}", int(self.char_topk_slider.value)))
        
        # Initialize character slider labels
        self._update_slider_label(self.char_temp_value_text, "Temp {:.1f}", self.char_temp_slider.value)
        self._update_slider_label(self.char_topp_value_text, "Top-P {:.2f}", self.char_topp_slider.value)
        self._update_slider_label(self.char_topk_value_text, "Top-K {}", int(self.char_topk_slider.value))
        
        self.char_save_btn = ElevatedButton("💾 Save Character", icon=Icons.SAVE, on_click=self._save_character)
        self.char_delete_btn = ElevatedButton("🗑️ Delete", icon=Icons.DELETE, on_click=self._delete_character, color=Colors.RED_400)
        self.char_back_btn = TextButton("← Back to Gallery", on_click=self._show_char_browse)
        
        # Browse vs Edit views
        self.char_browse_view = Column([], expand=True)
        self.char_edit_view = Column([], expand=True, visible=False)
        
        self.char_status = Text("", size=12, color=Colors.GREEN_400)
        
        # =========== PERSONAS TAB COMPONENTS ===========
        
        # Persona list for browsing
        self.persona_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        # Persona edit form
        self.persona_name_input = TextField(label="Persona Name", hint_text="e.g., Space Explorer")
        self.persona_desc_input = TextField(label="Short Description", hint_text="Brief dropdown description")
        self.persona_bg_input = TextField(
            label="Background/Context",
            hint_text="Detailed roleplay background: You are Captain Sarah Chen...",
            multiline=True,
            min_lines=5,
            max_lines=10,
        )
        self.persona_avatar_input = TextField(label="Avatar Emoji", value="👤", width=100)
        
        self.persona_save_btn = ElevatedButton("💾 Save Persona", icon=Icons.SAVE, on_click=self._save_persona)
        self.persona_delete_btn = ElevatedButton("🗑️ Delete", icon=Icons.DELETE, on_click=self._delete_persona, color=Colors.RED_400)
        self.persona_back_btn = TextButton("← Back to Gallery", on_click=self._show_persona_browse)
        
        # Browse vs Edit views
        self.persona_browse_view = Column([], expand=True)
        self.persona_edit_view = Column([], expand=True, visible=False)
        
        self.persona_status = Text("", size=12, color=Colors.GREEN_400)
    
    def _get_character_options(self):
        """Get character options for dropdown."""
        options = [dropdown.Option("none", "None (Use System Prompt)")]
        for char in self.db.get_all_ai_characters():
            options.append(dropdown.Option(
                str(char['id']),
                f"{char['avatar']} {char['name']}"
            ))
        return options
    
    def _get_persona_options(self):
        """Get persona options for dropdown."""
        options = [dropdown.Option("none", "None")]
        for persona in self.db.get_all_user_personas():
            options.append(dropdown.Option(
                str(persona['id']),
                f"{persona['avatar']} {persona['name']}"
            ))
        return options
    
    def _refresh_dropdowns(self):
        """Refresh character and persona dropdowns."""
        self.char_dropdown.options = self._get_character_options()
        self.persona_dropdown.options = self._get_persona_options()
    
    def _on_character_change(self, e):
        """Handle character selection."""
        if e.control.value == "none":
            self.selected_character = None
            self.system_prompt.visible = True
            self.temp_slider.value = 0.7
        else:
            char_id = int(e.control.value)
            self.selected_character = self.db.get_ai_character(char_id)
            if self.selected_character:
                self.system_prompt.visible = False
                self.temp_slider.value = self.selected_character.get('temperature', 0.7)
        self.page.update()
    
    def _on_persona_change(self, e):
        """Handle persona selection."""
        if e.control.value == "none":
            self.selected_persona = None
        else:
            persona_id = int(e.control.value)
            self.selected_persona = self.db.get_user_persona(persona_id)
        self.page.update()
    
    def update_model_state(self):
        """Update UI based on current model state."""
        has_model = app_state.loaded_model_type == "chat"
        self.send_btn.disabled = not has_model
        self.no_model_banner.visible = not has_model
        try:
            self.page.update()
        except:
            pass
    
    def _add_message(self, role: str, content: str):
        """Add a message bubble to the chat."""
        is_user = role == "user"
        
        # Get display name from character/persona
        if is_user:
            display_name = self.selected_persona['name'] if self.selected_persona else "You"
        else:
            display_name = self.selected_character['name'] if self.selected_character else "AI"
        
        # Theme-aware colors: use primary for user, surface variant for AI
        if is_user:
            bubble_bg = Colors.with_opacity(0.3, Colors.PRIMARY)
            name_color = Colors.PRIMARY
        else:
            bubble_bg = Colors.with_opacity(0.15, Colors.ON_SURFACE)
            name_color = Colors.SECONDARY
        
        bubble = Container(
            content=Column([
                Text(
                    display_name,
                    size=10,
                    weight=FontWeight.BOLD,
                    color=name_color,
                ),
                Text(content, selectable=True),
            ], spacing=3),
            bgcolor=bubble_bg,
            padding=padding.all(12),
            border_radius=border_radius.all(12),
            margin=ft.margin.only(
                left=50 if is_user else 0,
                right=0 if is_user else 50,
            ),
        )
        
        self.chat_messages.controls.append(bubble)
    
    def _send_message(self, e):
        """Send a message and get AI response."""
        if self.generating:
            return
        
        message = self.message_input.value.strip()
        if not message:
            return
        
        # Check if a chat model is loaded
        if app_state.loaded_model_type != "chat":
            self._add_message("assistant", "⚠️ Please load a 💬 Chat model from the header dropdown first.")
            self.page.update()
            return
        
        # Clear input and add user message
        self.message_input.value = ""
        self._add_message("user", message)
        self.generating = True
        self.typing_indicator.visible = True
        self.progress.visible = True
        self.send_btn.disabled = True
        self.page.update()
        
        def do_generate():
            try:
                from core.chat_gen import generate_chat_response, clear_conversation
                from core.prompt_builder import PromptBuilder
                
                # Build system prompt from character or manual input
                if self.selected_character:
                    base_system = self.selected_character.get('system_prompt', '')
                    temp = self.selected_character.get('temperature', 0.7)
                else:
                    base_system = self.system_prompt.value or None
                    temp = self.temp_slider.value
                
                # Prepare PromptBuilder with character overrides
                builder = PromptBuilder()
                builder.load_character_overrides(self.selected_character)
                
                # Gather persona info
                persona_name = None
                persona_bg = None
                if self.selected_persona and self.selected_persona.get('background'):
                    persona_name = self.selected_persona['name']
                    persona_bg = self.selected_persona['background']
                
                # Assemble system prompt
                system = builder.build_system_prompt(
                    base_system_prompt=base_system,
                    persona_name=persona_name,
                    persona_background=persona_bg,
                )
                
                # Get stop phrases from builder
                char_name = self.selected_character.get('name') if self.selected_character else None
                p_name = self.selected_persona.get('name') if self.selected_persona else None
                extra_stops = builder.get_stop_phrases(character_name=char_name, persona_name=p_name)
                
                success, response = generate_chat_response(
                    user_message=message,
                    system_prompt=system,
                    max_new_tokens=int(self.max_tokens_slider.value),
                    temperature=temp,
                    extra_stop_phrases=extra_stops,
                )
                
                self._add_message("assistant", response)
            except ImportError:
                self._add_message("assistant", "❌ Chat generation module not available")
            except Exception as ex:
                self._add_message("assistant", f"❌ Error: {ex}")
                logger.error(f"Chat generation error: {ex}")
            finally:
                self.generating = False
                self.typing_indicator.visible = False
                self.progress.visible = False
                self.send_btn.disabled = False
                
                try:
                    self.page.update()
                except:
                    pass
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _clear_conversation(self, e):
        """Clear the conversation history."""
        try:
            from core.chat_gen import clear_conversation
            clear_conversation()
        except ImportError:
            pass
        self.chat_messages.controls.clear()
        self.page.update()
    
    # =========== CHARACTER MANAGEMENT ===========
    
    def _build_char_list(self):
        """Build the character browse list."""
        self.char_list.controls.clear()
        
        # Add "Create New" button
        self.char_list.controls.append(
            Container(
                content=Row([
                    ft.Icon(Icons.ADD_CIRCLE_OUTLINE, size=24, color=Colors.BLUE_400),
                    Text("➕ Create New Character", size=14, weight=FontWeight.BOLD),
                ], spacing=10),
                bgcolor=Colors.BLUE_900,
                padding=padding.all(12),
                border_radius=border_radius.all(8),
                on_click=lambda e: self._edit_character(None),
                ink=True,
            )
        )
        
        # Add existing characters
        for char in self.db.get_all_ai_characters():
            self.char_list.controls.append(
                Container(
                    content=Row([
                        Text(char['avatar'], size=24),
                        Column([
                            Text(char['name'], weight=FontWeight.BOLD),
                            Text(char.get('description', '')[:50] or "No description", size=11, color=Colors.GREY_500),
                        ], spacing=2, expand=True),
                        ft.Icon(Icons.EDIT, size=18, color=Colors.GREY_400),
                    ], spacing=10),
                    bgcolor=Colors.GREY_900,
                    padding=padding.all(12),
                    border_radius=border_radius.all(8),
                    on_click=lambda e, c=char: self._edit_character(c),
                    ink=True,
                )
            )
    
    def _show_char_browse(self, e=None):
        """Show character browse view."""
        self._build_char_list()
        self.char_browse_view.visible = True
        self.char_edit_view.visible = False
        self.page.update()
    
    def _edit_character(self, char):
        """Show character edit view."""
        self.editing_character_id = char['id'] if char else None
        
        # Populate form
        self.char_name_input.value = char['name'] if char else ""
        self.char_desc_input.value = char.get('description', '') if char else ""
        self.char_system_input.value = char.get('system_prompt', '') if char else ""
        self.char_avatar_input.value = char.get('avatar', '🤖') if char else "🤖"
        self.char_temp_slider.value = char.get('temperature', 0.7) if char else 0.7
        self.char_topp_slider.value = char.get('top_p', 0.95) if char else 0.95
        self.char_topk_slider.value = char.get('top_k', 50) if char else 50
        
        self.char_delete_btn.visible = char is not None
        self.char_status.value = ""
        
        self.char_browse_view.visible = False
        self.char_edit_view.visible = True
        self.page.update()
    
    def _save_character(self, e):
        """Save character to database."""
        name = self.char_name_input.value.strip()
        if not name:
            self.char_status.value = "❌ Name is required"
            self.char_status.color = Colors.RED_400
            self.page.update()
            return
        
        data = {
            'name': name,
            'description': self.char_desc_input.value.strip(),
            'system_prompt': self.char_system_input.value.strip(),
            'avatar': self.char_avatar_input.value.strip() or "🤖",
            'temperature': self.char_temp_slider.value,
            'top_p': self.char_topp_slider.value,
            'top_k': int(self.char_topk_slider.value),
        }
        
        try:
            if self.editing_character_id:
                self.db.update_ai_character(self.editing_character_id, **data)
                self.char_status.value = "✅ Character updated!"
            else:
                self.db.create_ai_character(**data)
                self.char_status.value = "✅ Character created!"
            
            self.char_status.color = Colors.GREEN_400
            self._refresh_dropdowns()
            self.page.update()
            
            # Go back to browse after a short delay
            import time
            time.sleep(0.5)
            self._show_char_browse()
        except Exception as ex:
            self.char_status.value = f"❌ Error: {ex}"
            self.char_status.color = Colors.RED_400
            self.page.update()
    
    def _delete_character(self, e):
        """Delete the current character."""
        if self.editing_character_id:
            try:
                self.db.delete_ai_character(self.editing_character_id)
                self._refresh_dropdowns()
                self._show_char_browse()
            except Exception as ex:
                self.char_status.value = f"❌ Error: {ex}"
                self.char_status.color = Colors.RED_400
                self.page.update()
    
    # =========== PERSONA MANAGEMENT ===========
    
    def _build_persona_list(self):
        """Build the persona browse list."""
        self.persona_list.controls.clear()
        
        # Add "Create New" button
        self.persona_list.controls.append(
            Container(
                content=Row([
                    ft.Icon(Icons.ADD_CIRCLE_OUTLINE, size=24, color=Colors.PURPLE_400),
                    Text("➕ Create New Persona", size=14, weight=FontWeight.BOLD),
                ], spacing=10),
                bgcolor=Colors.PURPLE_900,
                padding=padding.all(12),
                border_radius=border_radius.all(8),
                on_click=lambda e: self._edit_persona(None),
                ink=True,
            )
        )
        
        # Add existing personas
        for persona in self.db.get_all_user_personas():
            self.persona_list.controls.append(
                Container(
                    content=Row([
                        Text(persona['avatar'], size=24),
                        Column([
                            Text(persona['name'], weight=FontWeight.BOLD),
                            Text(persona.get('description', '')[:50] or "No description", size=11, color=Colors.GREY_500),
                        ], spacing=2, expand=True),
                        ft.Icon(Icons.EDIT, size=18, color=Colors.GREY_400),
                    ], spacing=10),
                    bgcolor=Colors.GREY_900,
                    padding=padding.all(12),
                    border_radius=border_radius.all(8),
                    on_click=lambda e, p=persona: self._edit_persona(p),
                    ink=True,
                )
            )
    
    def _show_persona_browse(self, e=None):
        """Show persona browse view."""
        self._build_persona_list()
        self.persona_browse_view.visible = True
        self.persona_edit_view.visible = False
        self.page.update()
    
    def _edit_persona(self, persona):
        """Show persona edit view."""
        self.editing_persona_id = persona['id'] if persona else None
        
        # Populate form
        self.persona_name_input.value = persona['name'] if persona else ""
        self.persona_desc_input.value = persona.get('description', '') if persona else ""
        self.persona_bg_input.value = persona.get('background', '') if persona else ""
        self.persona_avatar_input.value = persona.get('avatar', '👤') if persona else "👤"
        
        self.persona_delete_btn.visible = persona is not None
        self.persona_status.value = ""
        
        self.persona_browse_view.visible = False
        self.persona_edit_view.visible = True
        self.page.update()
    
    def _save_persona(self, e):
        """Save persona to database."""
        name = self.persona_name_input.value.strip()
        if not name:
            self.persona_status.value = "❌ Name is required"
            self.persona_status.color = Colors.RED_400
            self.page.update()
            return
        
        data = {
            'name': name,
            'description': self.persona_desc_input.value.strip(),
            'background': self.persona_bg_input.value.strip(),
            'avatar': self.persona_avatar_input.value.strip() or "👤",
        }
        
        try:
            if self.editing_persona_id:
                self.db.update_user_persona(self.editing_persona_id, **data)
                self.persona_status.value = "✅ Persona updated!"
            else:
                self.db.create_user_persona(**data)
                self.persona_status.value = "✅ Persona created!"
            
            self.persona_status.color = Colors.GREEN_400
            self._refresh_dropdowns()
            self.page.update()
            
            # Go back to browse after a short delay
            import time
            time.sleep(0.5)
            self._show_persona_browse()
        except Exception as ex:
            self.persona_status.value = f"❌ Error: {ex}"
            self.persona_status.color = Colors.RED_400
            self.page.update()
    
    def _delete_persona(self, e):
        """Delete the current persona."""
        if self.editing_persona_id:
            try:
                self.db.delete_user_persona(self.editing_persona_id)
                self._refresh_dropdowns()
                self._show_persona_browse()
            except Exception as ex:
                self.persona_status.value = f"❌ Error: {ex}"
                self.persona_status.color = Colors.RED_400
                self.page.update()
    
    def build(self) -> Container:
        """Build the chat tab with sub-tabs."""
        # Build character browse/edit views
        self._build_char_list()
        self.char_browse_view.controls = [
            Text("🎴 Your AI Characters", size=18, weight=FontWeight.BOLD),
            Text("Click any character to edit it.", size=12, color=Colors.GREY_500),
            Container(height=10),
            self.char_list,
        ]
        
        self.char_edit_view.controls = [
            self.char_back_btn,
            Container(height=10),
            Text("Edit AI Character" if self.editing_character_id else "Create AI Character", size=18, weight=FontWeight.BOLD),
            Container(height=10),
            self.char_name_input,
            self.char_desc_input,
            self.char_system_input,
            Container(height=10),
            Text("Generation Parameters", weight=FontWeight.BOLD),
            Text("Temperature:", size=11),
            self.char_temp_slider,
            self.char_temp_value_text,
            Container(height=8),
            Text("Top P:", size=11),
            self.char_topp_slider,
            self.char_topp_value_text,
            Container(height=8),
            Text("Top K:", size=11),
            self.char_topk_slider,
            self.char_topk_value_text,
            Container(height=8),
            self.char_avatar_input,
            Container(height=10),
            Row([self.char_save_btn, self.char_delete_btn], spacing=10),
            self.char_status,
        ]
        
        # Build persona browse/edit views
        self._build_persona_list()
        self.persona_browse_view.controls = [
            Text("🎴 Your User Personas", size=18, weight=FontWeight.BOLD),
            Text("Click any persona to edit it.", size=12, color=Colors.GREY_500),
            Container(height=10),
            self.persona_list,
        ]
        
        self.persona_edit_view.controls = [
            self.persona_back_btn,
            Container(height=10),
            Text("Edit User Persona" if self.editing_persona_id else "Create User Persona", size=18, weight=FontWeight.BOLD),
            Container(height=10),
            self.persona_name_input,
            self.persona_desc_input,
            self.persona_bg_input,
            self.persona_avatar_input,
            Container(height=10),
            Row([self.persona_save_btn, self.persona_delete_btn], spacing=10),
            self.persona_status,
        ]
        
        # Build conversation tab content
        conversation_content = Row([
            # Sidebar - Character/Persona & Settings
            Container(
                content=Column([
                    self.model_status,
                    Container(height=10),
                    Text("🎭 Character & Persona", weight=FontWeight.BOLD, size=12),
                    self.char_dropdown,
                    self.persona_dropdown,
                    Container(height=8),
                    self.system_prompt,
                    Container(height=10),
                    Text("⚙️ Parameters", weight=FontWeight.BOLD, size=12),
                    Row([self.temp_slider, self.temp_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                    Row([self.max_tokens_slider, self.max_tokens_value_text], spacing=5, vertical_alignment=CrossAxisAlignment.CENTER),
                ], spacing=8, scroll=ScrollMode.AUTO),
                width=280,
                padding=padding.all(15),
                alignment=alignment.top_left,
            ),
            # Main chat area
            Container(
                content=Column([
                    self.no_model_banner,
                    Row([
                        Text("💬 Conversation", size=18, weight=FontWeight.BOLD),
                        self.clear_btn,
                    ], alignment=MainAxisAlignment.SPACE_BETWEEN),
                    self.progress,
                    self.chat_messages,
                    # Typing indicator above message input
                    self.typing_indicator,
                    Row([
                        self.message_input,
                        self.send_btn,
                    ], spacing=10),
                ], expand=True),
                expand=True,
                padding=padding.all(15),
            ),
        ], expand=True)
        
        # Update initial state
        has_model = app_state.loaded_model_type == "chat"
        self.send_btn.disabled = not has_model
        self.no_model_banner.visible = not has_model
        
        # Characters tab content
        characters_content = Container(
            content=Column([
                self.char_browse_view,
                self.char_edit_view,
            ], expand=True),
            expand=True,
            padding=padding.all(15),
        )
        
        # Personas tab content
        personas_content = Container(
            content=Column([
                self.persona_browse_view,
                self.persona_edit_view,
            ], expand=True),
            expand=True,
            padding=padding.all(15),
        )
        
        return Container(
            content=Tabs(
                tabs=[
                    Tab(
                        text="💬 Conversation",
                        content=conversation_content,
                    ),
                    Tab(
                        text="👤 Characters",
                        content=characters_content,
                    ),
                    Tab(
                        text="🎭 Personas",
                        content=personas_content,
                    ),
                ],
                expand=True,
            ),
            expand=True,
        )


class MockChatDB:
    """Mock database for when the real one isn't available."""
    
    def __init__(self):
        self.characters = []
        self.personas = []
    
    def get_all_ai_characters(self):
        return self.characters
    
    def get_all_user_personas(self):
        return self.personas
    
    def get_ai_character(self, char_id):
        for c in self.characters:
            if c['id'] == char_id:
                return c
        return None
    
    def get_user_persona(self, persona_id):
        for p in self.personas:
            if p['id'] == persona_id:
                return p
        return None
    
    def create_ai_character(self, **data):
        data['id'] = len(self.characters) + 1
        self.characters.append(data)
        return data['id']
    
    def create_user_persona(self, **data):
        data['id'] = len(self.personas) + 1
        self.personas.append(data)
        return data['id']
    
    def update_ai_character(self, char_id, **data):
        for i, c in enumerate(self.characters):
            if c['id'] == char_id:
                self.characters[i].update(data)
                return
    
    def update_user_persona(self, persona_id, **data):
        for i, p in enumerate(self.personas):
            if p['id'] == persona_id:
                self.personas[i].update(data)
                return
    
    def delete_ai_character(self, char_id):
        self.characters = [c for c in self.characters if c['id'] != char_id]
    
    def delete_user_persona(self, persona_id):
        self.personas = [p for p in self.personas if p['id'] != persona_id]
