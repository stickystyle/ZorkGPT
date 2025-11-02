# ABOUTME: Knowledge base condensation utilities using LLM-based compression.
# ABOUTME: Reduces knowledge base size while preserving critical strategic information.

"""
Knowledge base condensation module.

This module provides LLM-based condensation of verbose knowledge bases into more
concise formats. The condensation process focuses on removing redundancy and improving
information density while preserving all critical strategic insights, danger warnings,
and decision-making guidance.
"""

from typing import Optional
from shared_utils import estimate_tokens

# Langfuse observe decorator with graceful fallback
try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


@observe(name="knowledge-condense")
def condense_knowledge_base(
    verbose_knowledge: str,
    client,
    condensation_model: str,
    condensation_sampling: dict,
    log_prompt_callback=None,
    logger=None
) -> Optional[str]:
    """
    Use the condensation_model to condense a knowledge base into a more concise format.

    This step focuses purely on reformatting and removing redundancy without
    adding new strategies or losing critical information. The goal is to achieve
    50-70% reduction in size while maintaining 100% of strategic value.

    Args:
        verbose_knowledge: The full knowledge base content (without map section)
        client: LLM client wrapper instance
        condensation_model: Model identifier for condensation task
        condensation_sampling: Sampling parameters (temperature, top_p, top_k, min_p, max_tokens)
        log_prompt_callback: Optional callback function to log prompts (receives messages list and prefix)
        logger: Optional logger instance for logging

    Returns:
        Condensed knowledge base or original content if condensation failed or didn't reduce size
    """

    if not verbose_knowledge or len(verbose_knowledge) < 1000:
        # Don't condense if content is already short
        return verbose_knowledge

    prompt = f"""You are tasked with condensing this Zork strategy guide into a more concise format while preserving ALL critical information.

**CRITICAL REQUIREMENTS**:
1. **NO NEW STRATEGIES**: Only reformat existing content - never invent or add new strategic advice
2. **PRESERVE ALL KEY INFORMATION**: Every important insight, danger warning, item detail, and strategic pattern must be retained
3. **REMOVE REDUNDANCY**: Eliminate repetitive statements and merge similar advice
4. **MAINTAIN STRUCTURE**: Keep the logical organization but make it more compact
5. **AI-FOCUSED LANGUAGE**: Use direct, actionable instructions for an AI language model
6. **CONSOLIDATE EXAMPLES**: Merge similar examples or scenarios into representative cases

**CURRENT KNOWLEDGE BASE TO CONDENSE**:
{verbose_knowledge}

**CONDENSATION GUIDELINES**:
- Merge repetitive advice into single, comprehensive statements
- Combine similar examples or scenarios into representative cases
- Use bullet points and concise formatting for better readability
- Eliminate verbose explanations while keeping essential details
- Maintain all specific game elements (locations, items, commands, dangers)
- Preserve the strategic frameworks and decision-making patterns
- Keep all unique insights and specialized knowledge

**OUTPUT FORMAT**: Provide a condensed version that is 50-70% of the original length while maintaining 100% of the strategic value.

Focus on creating a guide that is information-dense but highly readable for an AI agent during gameplay.

**IMPORTANT**: Do not add any meta-commentary about the knowledge base structure or organization. Do not include sections like "Updated Knowledge Base Structure" or explanations of how the content is organized. Simply provide the condensed content directly."""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert technical writer specializing in condensing strategic guides for AI systems. Your goal is to maximize information density while preserving completeness and accuracy. Never add new information - only reorganize and consolidate existing content.",
            },
            {"role": "user", "content": prompt},
        ]

        # Log the condensation prompt if callback provided
        if log_prompt_callback:
            log_prompt_callback(messages, "knowledge_condensation")

        response = client.chat.completions.create(
            model=condensation_model,
            messages=messages,
            temperature=condensation_sampling.get('temperature', 0.3),
            top_p=condensation_sampling.get('top_p'),
            top_k=condensation_sampling.get('top_k'),
            min_p=condensation_sampling.get('min_p'),
            max_tokens=condensation_sampling.get('max_tokens', 5000),
            name="StrategyGenerator",
        )

        condensed_content = response.content.strip()

        # Validate that condensation was successful and actually shorter
        if condensed_content and len(condensed_content) < len(verbose_knowledge):
            # Provide both character and token estimates for better feedback
            original_tokens = estimate_tokens(verbose_knowledge)
            condensed_tokens = estimate_tokens(condensed_content)

            if logger:
                logger.info(
                    f"Knowledge condensed: {len(verbose_knowledge)} -> {len(condensed_content)} characters ({len(condensed_content) / len(verbose_knowledge) * 100:.1f}%)",
                    extra={
                        "event_type": "knowledge_update",
                        "details": f"Token estimate: {original_tokens} -> {condensed_tokens} tokens ({condensed_tokens / original_tokens * 100:.1f}%)",
                    },
                )
            return condensed_content
        else:
            if logger:
                logger.warning(
                    "Condensation failed or didn't reduce size - keeping original",
                    extra={"event_type": "knowledge_update"},
                )
            return verbose_knowledge

    except Exception as e:
        if logger:
            logger.error(
                f"Knowledge condensation failed: {e}",
                extra={"event_type": "knowledge_update"},
            )
        return verbose_knowledge  # Return original on failure
