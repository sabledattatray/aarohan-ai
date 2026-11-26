# AAROHAN AI — UI/UX MASTER PRODUCT REQUIREMENTS DOCUMENT (PRD)

**Project:** AAROHAN AI  
**Document:** ChatGPT-Inspired Desktop UI/UX Specification  
**Platform:** Windows Desktop Application + Responsive Web Interface  
**Design Direction:** Premium, Minimal, Modern AI Workspace  
**Primary Goal:** Build a polished, professional AI interface inspired by the usability and interaction patterns of ChatGPT.

---

# 1. PRODUCT VISION

Transform AAROHAN AI into a premium AI workspace with a clean, intuitive, distraction-free interface.

The application should provide a familiar conversational experience while maintaining AAROHAN's own identity.

The UI must support:

- AI conversations
- Conversation history
- Persistent memory
- Knowledge management
- File uploads and analysis
- Code generation and code presentation
- Website and application development workflows
- Model management
- Local AI inference
- Settings and diagnostics
- Future AI capabilities

The interface must look like a professionally developed AI product, not a basic chatbot or an ordinary Python application.

# 2. DESIGN PRINCIPLES

Follow these principles throughout the application:

1. Minimalism: Avoid unnecessary visual elements.
2. Consistency: Use the same spacing, typography, colours, and component behaviour throughout.
3. Readability: Prioritise comfortable reading during long conversations.
4. Responsiveness: Support different window sizes.
5. Performance: Avoid unnecessary animations and heavy UI dependencies.
6. Accessibility: Support keyboard navigation, clear focus indicators, and appropriate contrast.
7. Functionality: Every visible button and interactive component must work.
8. Identity: Maintain AAROHAN branding without copying ChatGPT logos or proprietary assets.

# 3. TYPOGRAPHY SYSTEM

Use the following typography.

## Primary font

Inter

Use Inter for:
- Application navigation
- Chat messages
- Headings
- Buttons
- Labels
- Input fields
- Settings
- Conversation history
- Modal windows

Preferred font weights:
- Regular: 400
- Medium: 500
- Semibold: 600
- Bold: 700

## Monospace font

JetBrains Mono

Use JetBrains Mono for:
- Programming code
- SQL queries
- Terminal output
- Technical logs
- JSON
- Stack traces

## Font sizing

| Element | Font size |
|---|---:|
| Main page heading | 28–32px |
| Section heading | 20–24px |
| Chat message | 15–16px |
| Sidebar text | 13–14px |
| Navigation labels | 14px |
| Secondary information | 12–13px |
| Code | 13–14px |
| Input text | 15–16px |

Use comfortable line-height:
- Chat messages: 1.65
- Headings: 1.25
- Sidebar: 1.4
- Code: 1.6

Load fonts locally where possible to ensure offline compatibility.

# 4. COLOUR SYSTEM

Create two complete themes.

## Dark theme

The dark theme should be the default.

| Component | Colour |
|---|---|
| Main application background | #212121 |
| Sidebar background | #171717 |
| Chat background | #212121 |
| Input background | #303030 |
| Secondary surface | #2A2A2A |
| Primary text | #ECECEC |
| Secondary text | #A1A1AA |
| Muted text | #71717A |
| Border | #383838 |
| Hover background | #2F2F2F |
| Primary accent | #10A37F |
| Code background | #171717 |

## Light theme

| Component | Colour |
|---|---|
| Main application background | #FFFFFF |
| Sidebar background | #F9F9F9 |
| Chat background | #FFFFFF |
| Input background | #FFFFFF |
| Secondary surface | #F4F4F5 |
| Primary text | #18181B |
| Secondary text | #52525B |
| Muted text | #71717A |
| Border | #E4E4E7 |
| Hover background | #F1F1F1 |
| Primary accent | #0D9373 |
| Code background | #F4F4F5 |

Requirements:
- Implement a functional dark/light theme switch.
- Persist the user's theme preference.
- Avoid excessive gradients.
- Avoid bright colours across large surfaces.
- Use the accent colour sparingly.
- Maintain sufficient text contrast.

# 5. MAIN APPLICATION LAYOUT

Use a three-part layout:

1. Left sidebar
2. Main conversation area
3. Optional contextual panel

The contextual panel should remain hidden by default and open only when needed.

## Desktop layout

Recommended dimensions:

- Sidebar: 260px
- Main conversation: flexible width
- Chat content maximum width: 800px
- Optional contextual panel: 300px
- Header height: 56px
- Input container maximum width: 800px

Use a flexible layout rather than fixed positioning wherever possible.

# 6. LEFT SIDEBAR

The sidebar must remain visually simple and functional.

## Sidebar header

Display:
- AAROHAN logo or symbol
- AAROHAN product name
- Sidebar collapse/expand control
- New chat button

## Search

Add a conversation search field.

Requirements:
- Search conversation titles.
- Search conversation content where supported.
- Show matching conversations.
- Support empty search results.
- Provide a clear-search action.

## Main navigation

Include:

- New Chat
- Search
- Library
- Projects
- Memory
- Knowledge Base

Only display features that are actually implemented. Do not create fake navigation destinations.

## Conversation history

Organise conversations into:

- Today
- Yesterday
- Previous 7 Days
- Previous 30 Days
- Older

Each conversation entry must support:
- Open conversation
- Rename conversation
- Delete conversation
- Context menu
- Active conversation highlighting

Conversation titles should be generated from the initial conversation where possible.

## Sidebar footer

Display:
- User profile or local user identity
- Settings
- Model status
- Application version

The sidebar must support collapsing into an icon-only navigation mode.

# 7. MAIN CHAT INTERFACE

The conversation area is the most important part of AAROHAN.

## Initial welcome screen

When the user starts a new conversation, display:

AAROHAN

**How can I help you today?**

Below the heading, show optional prompt suggestions:

- Explain a complex topic
- Write and debug code
- Analyse a file
- Help with SQL and databases
- Build a website
- Research a topic

These suggestions must populate the chat input when clicked.

Do not automatically submit a message unless the user explicitly selects an action designed to submit it.

## Chat message layout

User messages:
- Right-aligned or presented in a visually distinct user-message container.
- Comfortable padding.
- Clear separation from assistant responses.

Assistant messages:
- Left-aligned.
- Readable full-width content within the central conversation column.
- No unnecessary card borders around ordinary paragraphs.
- Support Markdown rendering.

## Supported message content

The renderer must support:

- Paragraphs
- Headings
- Bold and italic text
- Numbered lists
- Bullet lists
- Tables
- Blockquotes
- Inline code
- Fenced code blocks
- Mathematical expressions
- Links
- Images
- File references
- Error messages
- Tool execution status

Use secure Markdown rendering and sanitise untrusted HTML.

## Assistant response actions

Below assistant responses, provide:

- Copy
- Good response
- Bad response
- Regenerate
- Additional actions

Only show actions that have real implementations.

## Streaming responses

Implement token-by-token or chunk-by-chunk response streaming if supported by the existing backend.

Requirements:
- Show responses progressively.
- Provide a stop-generation button.
- Maintain message order.
- Prevent duplicate messages.
- Handle interrupted responses.
- Display meaningful error messages.

# 8. CHAT INPUT COMPOSER

The input composer should be one of the most polished UI components.

## Appearance

- Rounded container.
- Subtle border.
- Dark/light theme support.
- Comfortable internal padding.
- Multiline text support.
- Expand naturally as the user types.
- Maximum height before internal scrolling.

## Controls

Include:

- Attachment button
- Message input
- Send button
- Stop-generation button when generating
- Optional tools selector

## Behaviour

- Enter: Send message.
- Shift + Enter: New line.
- Empty input: Disable send.
- Long messages: Support multiline scrolling.
- Pasted text: Preserve readable formatting.
- File upload: Display selected files before sending.
- Sending: Show loading state.
- Generation: Replace send control with stop control.

Never lose unsent user text when switching conversations.

# 9. FILE UPLOAD AND ANALYSIS

Provide a functional attachment workflow.

Support file types according to actual backend capabilities.

Potential formats:
- TXT
- PDF
- DOCX
- XLSX
- CSV
- JSON
- PY
- SQL
- JS
- HTML
- CSS

Requirements:
- Show filename.
- Show file type.
- Show upload progress where applicable.
- Allow removal before submission.
- Validate file size and supported formats.
- Display processing status.
- Show meaningful errors.

Do not claim that a file has been analysed if the backend did not successfully process it.

# 10. CODE PRESENTATION

Code generation must have a professional presentation.

Every code block should include:

- Programming language label
- Copy button
- Optional download action
- Syntax highlighting
- Horizontal scrolling when required

For longer code outputs:
- Maintain readable indentation.
- Preserve line breaks.
- Avoid truncating generated code.
- Do not silently omit sections.

Use JetBrains Mono.

If code editing is implemented, provide a dedicated editor with:
- Line numbers
- Syntax highlighting
- Search
- Copy
- Save
- Download

# 11. MEMORY INTERFACE

Create a dedicated Memory section.

It should display only memory records actually stored by the backend.

Possible categories:
- User preferences
- Important facts
- Project information
- Instructions
- Conversation-derived information

Each record should support:
- View
- Edit
- Delete
- Search
- Category filtering

Clearly distinguish:
- Conversation history
- Persistent memory
- Retrieved knowledge
- Model training

Never represent saved memory as model training.

# 12. MODEL MANAGEMENT

Create a Model Management page.

Display:
- Active model
- Model name
- Model version
- Model location
- Model size
- Runtime status
- CPU/GPU inference status
- Available memory information when supported

Provide:
- Select model
- Download model
- Cancel download
- Remove model
- Set default model
- Test inference

Show accurate model status from the backend.

Never display a model as loaded unless the inference runtime confirms it.

# 13. SETTINGS

Organise settings into clear sections.

## General
- Theme
- Language
- Startup behaviour
- Default landing page

## Personalisation
- Display preferences
- Chat density
- Message behaviour

## Models
- Active model
- Model storage location
- Inference settings

## Memory
- Memory enable/disable
- Memory storage information
- Memory management

## Data and privacy
- Local data location
- Export conversations
- Delete conversation history
- Clear application data with confirmation

## Performance
- CPU/GPU selection
- Runtime status
- Resource usage when available

## About
- AAROHAN version
- Developer information
- Licence information
- Documentation
- Diagnostics

All settings must connect to actual application functionality.

# 14. NOTIFICATIONS AND STATUS INDICATORS

Provide subtle status indicators for:

- Model loading
- Model ready
- Generating response
- File processing
- Downloading model
- Backend unavailable
- GPU unavailable
- Error
- Successful operation

Use:
- Small status labels
- Toast notifications
- Inline error messages
- Loading indicators

Avoid intrusive popups for ordinary operations.

# 15. ANIMATIONS AND MICRO-INTERACTIONS

Use subtle animations only.

Recommended:
- Sidebar collapse transition.
- Message appearance.
- Button hover.
- Input focus.
- Dropdown opening.
- Toast appearance.
- Loading indicator.
- Modal transitions.

Suggested animation duration:
- Hover: 120–160ms
- Dropdown: 150–200ms
- Modal: 180–220ms
- Sidebar: 200–250ms

Use reduced-motion preferences.

Do not add unnecessary animated backgrounds, glowing effects, or excessive transitions.

# 16. ICONOGRAPHY

Use Lucide icons.

Recommended icons:
- MessageSquare
- Search
- Plus
- PanelLeft
- Settings
- Brain
- Library
- Folder
- Paperclip
- Send
- Square
- Copy
- ThumbsUp
- ThumbsDown
- MoreHorizontal
- Trash2
- Pencil
- Download
- Upload
- FileText
- Code
- Database
- Cpu
- HardDrive
- Moon
- Sun

Use consistent icon sizing:
- Main navigation: 18px
- Header actions: 18px
- Input controls: 20px
- Small actions: 16px

Avoid mixing multiple icon libraries.

# 17. RESPONSIVE DESIGN

Support:

- Large desktop screens
- Standard laptop screens
- Small desktop windows
- Narrow windows

At narrower widths:
- Collapse the sidebar.
- Keep the conversation readable.
- Make the composer full-width.
- Prevent horizontal overflow.
- Move secondary actions into menus.
- Hide the optional contextual panel.

The application must remain usable when resized to a small Windows window.

# 18. WINDOWS DESKTOP EXPERIENCE

The desktop application must feel like a native Windows product.

Requirements:
- Proper application title.
- Application icon.
- Desktop shortcut.
- Start menu shortcut.
- Minimise, maximise, restore, and close.
- Native installer.
- Proper application version.
- Startup splash screen if needed.
- Graceful shutdown.
- Backend startup and shutdown management.
- Local application data storage.
- Proper error handling.

The user must not need to open PowerShell, manually run Python, or start Uvicorn to use the installed application.

The desktop shell must use secure IPC and restricted application permissions.

# 19. TECHNICAL IMPLEMENTATION

Before implementation, inspect the existing AAROHAN repository.

Do not assume the existing technology stack.

Recommended architecture, subject to compatibility:

- Desktop: Tauri 2
- Frontend: Existing React application or React + TypeScript
- Styling: Tailwind CSS
- Components: shadcn/ui where appropriate
- Icons: Lucide
- Backend: Existing Python backend
- Backend packaging: PyInstaller or a suitable alternative
- Database: Existing SQLite database
- Local inference: Existing model runtime

Important:

Do not replace the existing AI engine, custom Transformer, Qwen integration, memory system, knowledge graph, RAG, or backend architecture merely to implement the UI.

The frontend must communicate with actual backend APIs.

Do not create mock AI responses.

Do not hardcode fake model statuses, memory records, conversation counts, or processing results.

# 20. DESIGN SYSTEM IMPLEMENTATION

Create reusable design tokens for:

- Colours
- Typography
- Spacing
- Border radius
- Shadows
- Component heights
- Animation timings
- Breakpoints

Use a consistent spacing scale based on 4px increments.

Recommended border radius:
- Small controls: 8px
- Buttons: 10px
- Input container: 24px
- Cards: 12px
- Dialogs: 16px

Use subtle borders instead of heavy shadows.

Build reusable components:
- AppSidebar
- SidebarNavigation
- ConversationList
- ChatHeader
- WelcomeScreen
- ChatMessage
- MessageActions
- MarkdownRenderer
- CodeBlock
- ChatComposer
- AttachmentPreview
- ModelStatus
- SettingsDialog
- ToastNotification
- ThemeProvider

# 21. ACCESSIBILITY

Implement:
- Keyboard navigation.
- Visible focus indicators.
- Proper accessible labels.
- Appropriate contrast.
- Screen-reader-friendly controls.
- Semantic HTML.
- Reduced-motion support.
- Tooltips for icon-only controls.

# 22. PERFORMANCE REQUIREMENTS

The interface should:
- Load quickly.
- Render long conversations efficiently.
- Avoid unnecessary component re-renders.
- Support large code blocks.
- Handle long messages.
- Avoid blocking the UI during model inference.
- Keep file processing and inference operations asynchronous.

Use virtualisation for large conversation histories if required.

# 23. IMPLEMENTATION PHASES

## Phase 1 — UI Foundation
- Design tokens.
- Typography.
- Theme system.
- Application shell.
- Sidebar.
- Responsive layout.

## Phase 2 — Chat Experience
- Welcome screen.
- Chat messages.
- Markdown renderer.
- Code blocks.
- Composer.
- Streaming.
- Message actions.

## Phase 3 — Workspace
- Conversation history.
- Search.
- Memory interface.
- Library.
- Projects.

## Phase 4 — AI Integration
- Connect existing backend.
- Model status.
- Model management.
- File processing.
- Error handling.

## Phase 5 — Desktop Integration
- Tauri integration.
- Backend sidecar.
- Local storage.
- Windows installer.
- Startup and shutdown.

## Phase 6 — Quality Assurance
- UI testing.
- Backend integration testing.
- Theme testing.
- Keyboard testing.
- Window resizing.
- Performance testing.
- Installer testing.

# 24. ACCEPTANCE CRITERIA

The project is considered complete only when:

1. The application has a consistent ChatGPT-inspired visual experience.
2. Dark and light themes work.
3. Sidebar navigation works.
4. Conversation history persists.
5. New conversations work.
6. Messages render correctly.
7. Markdown works.
8. Code blocks support copying.
9. Message streaming works if supported by the backend.
10. Stop generation works.
11. File uploads connect to real backend functionality.
12. Memory records come from the actual memory system.
13. Model status reflects the real inference runtime.
14. Settings persist.
15. No visible button is decorative or non-functional.
16. No fake AI responses are used.
17. The application works after installation on Windows.
18. The application can be resized without breaking its layout.
19. The application handles backend failures gracefully.
20. Existing AAROHAN data and AI capabilities remain intact.

# 25. STRICT DEVELOPMENT RULES

- First inspect the entire existing repository.
- Prepare an implementation plan before changing files.
- Preserve existing working functionality.
- Do not rewrite working backend modules unnecessarily.
- Do not remove existing AI capabilities.
- Do not replace the custom Transformer with another model.
- Do not remove Qwen integration.
- Do not fake memory or autonomous learning.
- Do not introduce paid services or mandatory subscriptions.
- Prefer free and open-source dependencies.
- Keep core inference and data storage local.
- Do not claim a feature is complete without testing it.
- Do not leave unfinished placeholder components.
- Do not introduce unnecessary dependencies.
- Maintain documentation for all significant changes.

# FINAL OBJECTIVE

Deliver AAROHAN AI as a polished, professional, locally running AI desktop application.

The visual quality should be comparable to modern AI products such as ChatGPT, while AAROHAN retains its own branding, architecture, AI engine, memory, and independent capabilities.

**The objective is not to create a ChatGPT clone. The objective is to create a world-class user experience for AAROHAN AI.**