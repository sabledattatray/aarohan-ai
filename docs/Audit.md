# AAROHAN — REAL WEB SEARCH FORENSIC AUDIT

Investigate why AAROHAN cannot perform a genuine web search when the user explicitly requests one.

This is a READ-ONLY investigation initially.

Do not install packages, modify code, change configuration, or introduce API keys.

### 1. Inspect the existing architecture

Inspect:

- aarohan/api.py
- Intent classifier and intent routing
- Task router
- Model inference and tool-calling implementation
- Existing web search modules
- Environment configuration
- Dependencies and requirements files
- Frontend chat request and response handling

### 2. Establish whether real web search exists

Identify:

- Search provider name
- Python module and function
- Provider configuration
- Required API key, if applicable
- HTTP request implementation
- Actual search endpoint
- Result parsing implementation
- Error handling
- Whether live internet requests are possible

Return VERIFIED, NOT_AVAILABLE, UNKNOWN or ERROR for each finding.

### 3. Trace the chat pipeline

Trace this exact request:

"Search the web for Datta Sable and tell me about him. Include sources."

Follow it through:

User message → Intent detection → Task routing → Search invocation → Provider response → Result validation → Model context → Final answer → UI.

Identify the exact function and source-code lines at each stage.

### 4. Verify actual search execution

If an existing provider is configured, perform one controlled test query.

Record:

- Search query
- Provider
- HTTP status
- Number of results
- Result titles
- Result URLs
- Result snippets
- Execution time
- Errors, if any

Do not fabricate results or claim a search succeeded without receiving actual provider results.

### 5. Diagnose the failure

Determine whether the problem is:

- No search provider
- Missing API key
- Missing dependency
- Internet connectivity failure
- Intent classification failure
- Task router failure
- Search tool not registered
- Tool invocation failure
- Search results not injected into model context
- Frontend not displaying returned sources

### 6. Final report

Provide the exact root cause, evidence, affected files, missing components, and recommended repair.

Do not implement the repair yet.

The objective is to establish why real web search is not working before making architectural changes.