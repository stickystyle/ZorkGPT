  Your Assignment

  Implement Task #$ARGUMENTS from the implementation plan located at:
  .kiro/specs/mcp-integration/tasks.md

  Specification Files

  Before starting, read the following specification documents to understand the full context:

1. Requirements Document:  .kiro/specs/mcp-integration/requirements.md
  - Contains user stories and acceptance criteria
  - Defines what success looks like for each requirement
  - Specifies exact error messages and behavior
2. Design Document:  .kiro/specs/mcp-integration/design.md
  - Contains architecture overview and component design
  - Defines interfaces, data models, and implementation patterns
  - Includes security considerations and performance requirements
3. Implementation Plan: .kiro/specs/mcp-integration/tasks.md
  - Contains ordered list of implementation tasks
  - Maps tasks to specific requirements
  - Your assigned task includes the requirements it satisfies

How to Approach This Task

1. Understand the Task

- Read your assigned task from tasks.md
- Note which requirements it references (e.g., "Requirements: 1.1, 1.2, 1.3")
- Read those specific requirements in requirements.md
- Review the related design sections in design.md

2. Follow TDD (Test-Driven Development)

- Write tests FIRST based on acceptance criteria
- Implement minimal code to make tests pass
- Refactor while keeping tests green
- Ensure all referenced requirements have test coverage

3. Match the Specification Exactly

- Use exact error messages from requirements
- Follow interfaces and patterns from design document
- Implement only what the task specifies (no scope creep)
- Ask clarifying questions if specifications are ambiguous

4. Verify Completion

- All tests passing for your task's requirements
- Code follows patterns established in design document
- Logging matches structured format from design
- No security vulnerabilities introduced
- Update the tasks.md to mark the task box complete, e.g. [x]

  General Principles

- Fail-fast: Invalid states should prevent startup, not fail silently
- Fail-secure: Authentication failures default to rejection
- Explicit over implicit: Clear, readable code over clever solutions
- Security first: Never trust data before validation
- Test coverage: Every acceptance criterion needs a test

  Before You Start

1. Ask me which task number I want you to implement
2. Read all three specification files
3. Confirm you understand the task scope
4. Begin with writing tests

## Implementation

**CRITICAL**: All application code changes must be implemented using specialized agents.

- **Implementation**: Use Task tool with `subagent_type='python-engineer'`
- **Code Review**: Follow implementation with Task tool `subagent_type='code-reviewer'`
- **Fixes**: If issues identified in review, use `python-engineer` to fix before proceeding

**Exception**: Simple fixes (typos, comments, single-line changes) don't require agent delegation.

When you have fin

Questions?

If anything in the specification is unclear or ambiguous, ask before implementing. Do not make assumptions about security-critical behavior.
