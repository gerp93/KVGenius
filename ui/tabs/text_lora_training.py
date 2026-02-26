"""
Text LoRA Training Tab for KVGenius Flet App
Provides UI for training text/chat model LoRAs (personas, chat styles, etc.)
Extends the base training UI with persona-specific data input.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField,
    ElevatedButton, IconButton, Card, Dropdown, FilePicker,
    Icons, Colors, FontWeight, ScrollMode,
    padding, border_radius, alignment,
    CrossAxisAlignment, MainAxisAlignment,
    dropdown, FilePickerResultEvent
)

from src.training.text_lora_trainer import (
    TextLoRAType, TextTrainingExample, create_persona_examples
)
from ui.tabs.base_text_training import BaseTextTrainingTab

logger = logging.getLogger(__name__)


class TextLoRATrainingTab(BaseTextTrainingTab):
    """
    Text LoRA Training Tab - Train LoRAs for text/chat models.
    Allows training custom personas, chat styles, and knowledge domains.
    
    Extends BaseTextTrainingTab with persona-specific data input UI.
    """
    
    def get_lora_type(self) -> TextLoRAType:
        return TextLoRAType.PERSONA
    
    def get_tab_title(self) -> str:
        return "Train Text LoRA"
    
    def get_tab_icon(self) -> str:
        return "💬"
    
    def _build_common_ui(self):
        """Build UI components - extends parent with persona-specific fields."""
        super()._build_common_ui()
        
        # === Persona Configuration ===
        self.persona_name_field = TextField(
            label="Persona/Character Name",
            hint_text="e.g., Captain Sarah, Friendly Tutor",
            width=300,
            on_change=self._validate_form,
        )
        
        self.persona_description_field = TextField(
            label="Persona Description",
            hint_text="Brief description of the persona's personality and style",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
        )
        
        # === Training Data Input ===
        self.training_data_type = Dropdown(
            label="Data Input Method",
            width=200,
            options=[
                dropdown.Option("manual", "Manual Entry"),
                dropdown.Option("file", "Upload File"),
                dropdown.Option("paste", "Paste Conversations"),
            ],
            value="manual",
            on_change=self._on_data_type_change,
        )
        
        # Manual conversation entries
        self.conversation_list = ft.ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        self.add_conversation_btn = ElevatedButton(
            "➕ Add Conversation",
            icon=Icons.ADD,
            on_click=self._add_conversation,
        )
        
        # Paste area for bulk input
        self.paste_area = TextField(
            label="Paste Conversations (JSON or formatted text)",
            hint_text='[{"user": "Hello!", "assistant": "Hi there!"}, ...]',
            multiline=True,
            min_lines=8,
            max_lines=15,
            visible=False,
            expand=True,
            on_change=self._on_paste_change,
        )
        
        # File picker
        self.file_picker = FilePicker(
            on_result=self._on_file_picked,
        )
        
        self.file_picker_btn = ElevatedButton(
            "📁 Choose File",
            icon=Icons.UPLOAD_FILE,
            on_click=lambda e: self.file_picker.pick_files(
                allowed_extensions=["json", "txt", "jsonl"]
            ),
            visible=False,
        )
        
        self.file_status = Text("", size=11, color=Colors.GREY_400, visible=False)
        
        # Store parsed examples
        self._parsed_examples: List[dict] = []
        self._manual_conversations: List[dict] = []
        
        # Example count display
        self.example_count_text = Text(
            "0 training examples",
            size=12,
            color=Colors.GREY_500,
        )
    
    def _on_data_type_change(self, e):
        """Handle data input type change."""
        data_type = self.training_data_type.value
        
        # Show/hide appropriate input areas
        self.conversation_list.visible = (data_type == "manual")
        self.add_conversation_btn.visible = (data_type == "manual")
        self.paste_area.visible = (data_type == "paste")
        self.file_picker_btn.visible = (data_type == "file")
        self.file_status.visible = (data_type == "file")
        
        self.page.update()
    
    def _add_conversation(self, e=None):
        """Add a new conversation entry."""
        conv_idx = len(self._manual_conversations)
        
        user_field = TextField(
            label=f"User Message #{conv_idx + 1}",
            hint_text="What the user says...",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
            on_change=self._on_conversation_change,
        )
        
        assistant_field = TextField(
            label=f"Assistant Response #{conv_idx + 1}",
            hint_text="How the persona responds...",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True,
            on_change=self._on_conversation_change,
        )
        
        remove_btn = IconButton(
            icon=Icons.DELETE_OUTLINE,
            icon_color=Colors.RED_400,
            tooltip="Remove this conversation",
            on_click=lambda e, idx=conv_idx: self._remove_conversation(idx),
        )
        
        conv_card = Container(
            content=Column([
                Row([
                    Text(f"Conversation {conv_idx + 1}", weight=FontWeight.BOLD, size=12),
                    Container(expand=True),
                    remove_btn,
                ]),
                user_field,
                assistant_field,
            ], spacing=8),
            padding=padding.all(10),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(8),
            data={"idx": conv_idx, "user": user_field, "assistant": assistant_field},
        )
        
        self._manual_conversations.append({
            "user": "",
            "assistant": "",
            "card": conv_card,
            "user_field": user_field,
            "assistant_field": assistant_field,
        })
        
        self.conversation_list.controls.append(conv_card)
        self._update_example_count()
        self.page.update()
    
    def _remove_conversation(self, idx: int):
        """Remove a conversation entry."""
        if 0 <= idx < len(self._manual_conversations):
            conv = self._manual_conversations[idx]
            if conv["card"] in self.conversation_list.controls:
                self.conversation_list.controls.remove(conv["card"])
            self._manual_conversations.pop(idx)
            
            # Re-number remaining conversations
            for i, conv in enumerate(self._manual_conversations):
                conv["card"].data["idx"] = i
            
            self._update_example_count()
            self._validate_form()
            self.page.update()
    
    def _on_conversation_change(self, e):
        """Handle conversation field changes."""
        # Update stored values
        for conv in self._manual_conversations:
            conv["user"] = conv["user_field"].value or ""
            conv["assistant"] = conv["assistant_field"].value or ""
        
        self._update_example_count()
        self._validate_form()
    
    def _on_paste_change(self, e):
        """Handle paste area changes - try to parse as JSON."""
        text = self.paste_area.value or ""
        self._parsed_examples = []
        
        if not text.strip():
            self._update_example_count()
            self._validate_form()
            return
        
        try:
            # Try to parse as JSON array
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "user" in item and "assistant" in item:
                        self._parsed_examples.append(item)
        except json.JSONDecodeError:
            # Try to parse as formatted text (User: ... Assistant: ...)
            lines = text.strip().split("\n")
            current_user = ""
            current_assistant = ""
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith("user:"):
                    if current_user and current_assistant:
                        self._parsed_examples.append({
                            "user": current_user,
                            "assistant": current_assistant
                        })
                    current_user = line[5:].strip()
                    current_assistant = ""
                elif line.lower().startswith("assistant:"):
                    current_assistant = line[10:].strip()
            
            if current_user and current_assistant:
                self._parsed_examples.append({
                    "user": current_user,
                    "assistant": current_assistant
                })
        
        self._update_example_count()
        self._validate_form()
    
    def _on_file_picked(self, e: FilePickerResultEvent):
        """Handle file picker result."""
        if not e.files:
            return
        
        file_path = e.files[0].path
        self.file_status.value = f"Loading: {Path(file_path).name}..."
        self.file_status.visible = True
        self.page.update()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._parsed_examples = []
            
            # Try to parse as JSON
            if file_path.endswith('.json'):
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "user" in item and "assistant" in item:
                            self._parsed_examples.append(item)
            elif file_path.endswith('.jsonl'):
                for line in content.strip().split('\n'):
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict) and "user" in item and "assistant" in item:
                            self._parsed_examples.append(item)
            else:
                # Try to parse as formatted text
                self.paste_area.value = content
                self._on_paste_change(None)
            
            self.file_status.value = f"✅ Loaded {len(self._parsed_examples)} conversations from {Path(file_path).name}"
            self.file_status.color = Colors.GREEN_400
            
        except Exception as ex:
            self.file_status.value = f"❌ Error loading file: {ex}"
            self.file_status.color = Colors.RED_400
        
        self._update_example_count()
        self._validate_form()
        self.page.update()
    
    def _update_example_count(self):
        """Update the example count display."""
        count = len(self.get_training_examples())
        self.example_count_text.value = f"{count} training examples"
        
        if count >= 3:
            self.example_count_text.color = Colors.GREEN_400
        elif count > 0:
            self.example_count_text.color = Colors.YELLOW_400
        else:
            self.example_count_text.color = Colors.GREY_500
    
    def get_training_examples(self) -> List[TextTrainingExample]:
        """Extract training examples from the UI inputs."""
        data_type = self.training_data_type.value
        persona_name = self.persona_name_field.value or "Assistant"
        persona_desc = self.persona_description_field.value or ""
        
        conversations = []
        
        if data_type == "manual":
            for conv in self._manual_conversations:
                user_text = conv.get("user", "").strip()
                assistant_text = conv.get("assistant", "").strip()
                if user_text and assistant_text:
                    conversations.append({
                        "user": user_text,
                        "assistant": assistant_text
                    })
        else:
            conversations = self._parsed_examples
        
        return create_persona_examples(
            persona_name=persona_name,
            conversations=conversations,
            persona_description=persona_desc
        )
    
    def build_data_input_section(self) -> Container:
        """Build the data input section for persona training."""
        # Add file picker to page overlay
        self.page.overlay.append(self.file_picker)
        
        # Add a few empty conversations to start
        if not self._manual_conversations:
            for _ in range(3):
                self._add_conversation()
        
        return Card(
            content=Container(
                content=Column([
                    Text("📝 Training Data", size=14, weight=FontWeight.BOLD),
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    # Persona info
                    Row([
                        self.persona_name_field,
                        self.persona_description_field,
                    ], spacing=15),
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # Data input method
                    Row([
                        self.training_data_type,
                        Container(expand=True),
                        self.example_count_text,
                    ]),
                    
                    Text(
                        "Add example conversations that demonstrate how the persona speaks and responds.",
                        size=11, color=Colors.GREY_500,
                    ),
                    
                    # Manual entry area
                    Container(
                        content=Column([
                            Container(
                                content=self.conversation_list,
                                height=300,
                                border=ft.border.all(1, Colors.GREY_800),
                                border_radius=border_radius.all(8),
                            ),
                            self.add_conversation_btn,
                        ], spacing=10),
                        visible=True,
                    ),
                    
                    # Paste area
                    self.paste_area,
                    
                    # File picker
                    Row([
                        self.file_picker_btn,
                        self.file_status,
                    ], spacing=10),
                    
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    # Tips
                    Container(
                        content=Column([
                            Text("💡 Tips for better training:", size=11, weight=FontWeight.W_500, color=Colors.BLUE_400),
                            Text("• Include 10-50+ diverse conversation examples", size=10, color=Colors.GREY_400),
                            Text("• Show the persona's unique speaking style, vocabulary, and personality", size=10, color=Colors.GREY_400),
                            Text("• Include both short and longer responses", size=10, color=Colors.GREY_400),
                            Text("• The more consistent the examples, the better the LoRA will learn the style", size=10, color=Colors.GREY_400),
                        ], spacing=3),
                        padding=padding.all(10),
                        bgcolor=Colors.with_opacity(0.1, Colors.BLUE_900),
                        border_radius=border_radius.all(8),
                    ),
                    
                ], spacing=10),
                padding=padding.all(15),
            ),
        )
