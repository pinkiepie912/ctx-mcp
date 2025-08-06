# MCP: Model Context Protocol Server

> Context-aware dependency graph server for LLM-based coding agents

---

## 🎯 Goal

MCP helps eliminate redundant code generation and misplaced edits by providing a **dependency graph-based context server** for LLM-powered code tools such as GeminiCLI, ClaudeCode, and Cursor.

---

## ❗ The Problem

### 1. Context loss and code duplication
LLMs often forget earlier conversation context in long sessions, resulting in:
- Regenerating already existing features
- Creating new files unnecessarily
- Writing code in unrelated modules

### 2. Inconsistent transactional edits
- Signature changes without updating call sites
- Partial patches leading to broken runtime behavior

### 3. Missing test/type validations
- LLM-generated changes may not be reflected in test cases or type hints
- Static checks (mypy, linters) often fail after edits

### 4. Misaligned configuration updates
- Critical files like `.env`, `Dockerfile`, or `Helm charts` are often ignored
- LLMs don’t recognize their dependency on source code changes

---

## ✅ Our Solution

### 1. Dependency Graph Extraction with Tree-sitter
MCP uses [`tree-sitter`](https://tree-sitter.github.io/tree-sitter/) to build an AST-based dependency graph that includes:
- Class and function relationships (e.g., constructor injection)
- Module import/export dependencies
- Symbol usage and definitions across the codebase
- Config and environment file links to code modules

### 2. Context Pack for LLMs
MCP provides a structured "context pack" to help LLMs:
- Understand the full dependency landscape
- Edit consistently across all related files
- Include tests, types, and configuration changes

### 3. API-driven architecture
MCP exposes a FastAPI-based interface:
- Graph generation and querying
- Transaction-scoped context extraction
- Exporting JSON payloads for LLM injection

---

## 🧪 Example Usage

```bash
# Analyze the project directory
$ mcp analyze ./my_project

# Extract all relevant context for refactoring a symbol
$ mcp context extract --symbol MyService.update_user

# Export LLM-friendly context data
$ mcp context export --format=json > context_for_llm.json

