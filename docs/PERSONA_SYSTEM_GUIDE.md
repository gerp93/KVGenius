# Persona System Guide

## Overview

The KVGenius chatbot now has a dual persona system that separates:

1. **AI Characters** - What the AI roleplays as
2. **User Personas** - Who you (the user) are roleplaying as

## Concept

### AI Characters 🎭
These define **how the AI behaves and responds**:
- System prompts that control AI personality and behavior
- Generation parameters (temperature, top_p, top_k)
- Examples: Helpful Assistant, Creative Writer, Code Helper, Friendly Companion

### User Personas 👤
These define **who you are in the conversation**:
- Your character's name and description
- Background/context for roleplay scenarios
- Examples: Fantasy Adventurer, Space Explorer, Detective, or just "Myself"

## How It Works

When you chat:
1. Select an **AI Character** from the "AI Character" dropdown
   - This sets the AI's personality and system prompt
   - Controls generation parameters

2. Select a **User Persona** from the "User Persona" dropdown
   - This adds context about who you're roleplaying as
   - The AI will respond appropriately to your character

3. The combined context is sent to the model:
   ```
   System: [AI Character's system prompt]
   User is roleplaying as: [User Persona name and description]
   Background: [User Persona background]
   
   [Conversation history]
   
   User: [Your message]
   Assistant:
   ```

## Database Schema

### ai_characters table
- `name` - Character name
- `system_prompt` - Instructions for how AI should act
- `temperature` - Creativity level (0.1-1.5)
- `top_p` - Nucleus sampling parameter
- `top_k` - Top-K sampling parameter
- `description` - Brief description
- `avatar` - Emoji icon

### user_personas table
- `name` - Your character's name
- `description` - Brief description of who you are
- `background` - Detailed roleplay context
- `avatar` - Emoji icon

### conversations table
- `ai_character_id` - Foreign key to ai_characters
- `user_persona_id` - Foreign key to user_personas
- Both can be NULL (None selected)

## Default AI Characters

1. **🤖 Default Assistant** (temp: 0.7)
   - Helpful, smart, efficient AI assistant

2. **✍️ Creative Writer** (temp: 0.9)
   - Fiction, storytelling, vivid prose specialist

3. **💻 Code Helper** (temp: 0.3)
   - Programming and technical assistance

4. **😊 Friendly Companion** (temp: 0.85)
   - Casual conversation and emotional support

## Default User Personas

1. **👤 Myself (Default)**
   - Just being yourself, no roleplay

2. **⚔️ Fantasy Adventurer**
   - Brave warrior/mage on a quest

3. **🚀 Space Explorer**
   - Starship captain exploring the cosmos

4. **🕵️ Detective**
   - Private investigator solving mysteries

## Creating Custom Characters/Personas

### Creating an AI Character

1. Go to the "🎭 Manage AI Characters" tab
2. Fill in:
   - **Name**: e.g., "Pirate Captain"
   - **System Prompt**: "You are a gruff but good-hearted pirate captain. Speak with nautical slang and pirate dialect."
   - **Description**: "Swashbuckling pirate personality"
   - **Temperature**: 0.85 (creative but not too wild)
   - **Top-P**: 0.95
   - **Top-K**: 50
   - **Avatar**: 🏴‍☠️
3. Click "✨ Create AI Character"
4. Select from dropdown in main chat

### Creating a User Persona

1. Go to the "👤 Manage User Personas" tab
2. Fill in:
   - **Name**: e.g., "Royal Knight"
   - **Description**: "A noble knight in service to the kingdom"
   - **Background**: "You are Sir Roland, a decorated knight who has served the kingdom for 20 years. You value honor, duty, and protecting the innocent."
   - **Avatar**: 🛡️
3. Click "✨ Create User Persona"
4. Select from dropdown in main chat

## Use Cases

### Professional Work
- **AI**: Code Helper
- **User**: Myself (Default)
- Result: Technical programming assistance

### Creative Writing
- **AI**: Creative Writer
- **User**: Fantasy Adventurer
- Result: Interactive fantasy roleplay story

### Learning/Tutoring
- **AI**: Default Assistant
- **User**: Myself
- Result: Patient, helpful learning assistant

### Entertainment/Roleplay
- **AI**: Custom character (e.g., "Medieval Blacksmith")
- **User**: Custom persona (e.g., "Young Apprentice")
- Result: Immersive roleplay scenario

## Benefits

1. **Separation of Concerns**: AI behavior vs. user identity
2. **Mix and Match**: Any AI character can interact with any user persona
3. **Persistent**: Characters and personas saved in database
4. **Reusable**: Create once, use in many conversations
5. **Contextual**: AI responds appropriately to your character

## Migration Notes

The old "personas" have been renamed to "AI Characters" to clarify that they define the AI's behavior, not yours. The new "User Personas" system adds the ability to define who you are in the roleplay.

### Breaking Changes
- `personas` table → `ai_characters` table
- `persona_id` column → `ai_character_id` column
- Added new `user_persona_id` column
- Database schema updated (old database files will need recreation)

### API Changes
- `create_persona()` → `create_ai_character()`
- `get_persona()` → `get_ai_character()`
- `get_all_personas()` → `get_all_ai_characters()`
- Added `create_user_persona()`, `get_user_persona()`, `get_all_user_personas()`
- `init_default_personas()` → `init_defaults()` (creates both types)

## Tips

1. **Temperature Guide**:
   - 0.1-0.3: Focused, factual (code, analysis)
   - 0.5-0.7: Balanced (general chat)
   - 0.8-1.2: Creative (fiction, roleplay)
   - 1.3-1.5: Wild, experimental

2. **System Prompts**: Be specific about personality, speaking style, knowledge domains

3. **User Backgrounds**: Include motivations, skills, personality traits for better roleplay

4. **None Selection**: You can use "None" for either/both if you want vanilla chat

5. **Experimentation**: Try different AI-User combinations to find what works best
