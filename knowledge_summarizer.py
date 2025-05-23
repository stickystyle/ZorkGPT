"""
Knowledge Base Summarizer for ZorkGPT

This module takes raw accumulated knowledge and uses an LLM to create
a concise, strategic game guide that's more efficient and auditable.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from openai import OpenAI


class KnowledgeSummarizer:
    """Summarizes raw knowledge into a concise game guide using an LLM."""
    
    def __init__(self, 
                 model: str = "openai/gpt-4.1-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 4000,
                 client: Optional[OpenAI] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the knowledge summarizer.
        
        Args:
            model: LLM model to use for summarization
            temperature: Low temperature for consistent, focused output
            max_tokens: Maximum tokens for the summary
            client: Optional OpenAI client instance
            base_url: Optional base URL for the client
            api_key: Optional API key for the client
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if client:
            self.client = client
        else:
            # Use same client configuration as main agent
            client_kwargs = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            self.client = OpenAI(**client_kwargs)
        
    def summarize_knowledge_base(self, raw_knowledge: str) -> str:
        """
        Summarize raw knowledge into a strategic game guide.
        
        Args:
            raw_knowledge: The raw markdown knowledge base
            
        Returns:
            Summarized game guide as markdown
        """
        prompt = self._create_summarization_prompt()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Raw Knowledge Base:\n\n{raw_knowledge}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers={"X-Title": "ZorkGPT Knowledge Summarizer"}
            )
            
            summary = response.choices[0].message.content.strip()
            return self._post_process_summary(summary)
            
        except Exception as e:
            print(f"Error during summarization: {e}")
            return self._create_fallback_summary(raw_knowledge)
    
    def _create_summarization_prompt(self) -> str:
        """Create the system prompt for knowledge summarization."""
        return """You are a Zork game strategy expert. Your task is to analyze raw accumulated knowledge from many episode logs and create a strategic guide that promotes exploration while avoiding dangerous pitfalls.

**Your Goal:**
Transform verbose, repetitive raw data into a balanced strategy guide that helps an AI agent explore Zork effectively while avoiding known dangers.

**Critical Guidelines:**
1. **DO NOT create a "golden path"** - avoid prescriptive starting sequences
2. **EMPHASIZE EXPLORATION** - encourage trying different approaches and areas
3. **HIGHLIGHT DANGERS PROMINENTLY** - deaths and combat threats are crucial knowledge
4. **FOCUS ON SURVIVAL FIRST** - avoiding death is more important than optimization
5. **BE DISCOVERY-ORIENTED** - hint at opportunities without spoiling exploration

**Instructions:**
1. **Prioritize safety knowledge** - what kills the agent and how to avoid it
2. **Promote flexible exploration** - multiple valid starting approaches
3. **Highlight treasure opportunities** - both inside and outside the house
4. **Consolidate similar information** - don't repeat the same advice
5. **Use suggestive language** - "Consider...", "You might explore...", "Be aware that..."

**Output Format:**
```markdown
# Zork Strategy Guide

## Exploration Philosophy
[Encourage flexible, discovery-oriented gameplay]

## Critical Dangers & Survival
[Deadly threats, combat situations, and how to avoid death]

## Key Items & Treasure Opportunities
[Important items and where they might be found - both indoor and outdoor areas]

## Exploration Strategies
[Flexible approaches to discovering new areas and items]

## Navigation Tips
[Movement efficiency without creating rigid paths]

## Problem-Solving Insights
[General puzzle-solving approaches and known solutions]

## Learning from Failures
[Common mistakes and inefficient patterns to avoid]
```

**Quality Criteria:**
- NEVER suggest rigid starting sequences - exploration should feel organic
- Always prominently feature deadly threats (trolls, grues, etc.)
- Mention outdoor treasure opportunities to balance indoor focus
- Use encouraging, exploratory language rather than prescriptive commands
- Focus on teaching patterns rather than specific walkthroughs
- Help the agent learn to explore safely, not follow a predetermined path

Transform the raw knowledge into wisdom that makes the agent a better explorer, not a more efficient path-follower."""

    def _post_process_summary(self, summary: str) -> str:
        """Post-process the LLM summary for consistency and formatting."""
        # Add timestamp and metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        processed_summary = f"""# Zork Strategy Guide

*Strategically summarized knowledge base - Last updated: {timestamp}*
*Generated from accumulated episode data using AI analysis*

{summary}

---
*This guide is automatically generated from episode logs and strategically summarized for optimal gameplay. Focus on the most critical strategies first.*
"""
        
        return processed_summary
    
    def _create_fallback_summary(self, raw_knowledge: str) -> str:
        """Create a basic summary if LLM summarization fails."""
        lines = raw_knowledge.split('\n')
        
        # Extract some key information manually
        locations = []
        items = []
        mistakes = []
        
        current_section = None
        for line in lines:
            if line.startswith('### ') and not line.startswith('### Score-') and not line.startswith('### Dangerous'):
                if current_section == 'locations':
                    location_name = line.replace('### ', '').strip()
                    if '*Visited' in lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else False:
                        locations.append(location_name)
                elif current_section == 'items':
                    item_name = line.replace('### ', '').strip()
                    items.append(item_name)
            elif line.startswith('## Location Knowledge'):
                current_section = 'locations'
            elif line.startswith('## Item and Object Knowledge'):
                current_section = 'items'
            elif line.startswith('- Avoid repeating'):
                mistakes.append(line.replace('- ', '').strip())
        
        fallback_summary = f"""# Zork Strategy Guide

*Fallback summary - LLM summarization unavailable*

## Key Locations
{chr(10).join(f"- {loc}" for loc in locations[:10])}

## Important Items  
{chr(10).join(f"- {item}" for item in items[:10])}

## Common Mistakes to Avoid
{chr(10).join(f"- {mistake}" for mistake in mistakes[:5])}

## Note
This is a basic fallback summary. For better strategic insights, ensure the LLM summarization service is available.
"""
        return fallback_summary


def create_strategic_game_guide(raw_knowledge_file: str = "knowledgebase.md",
                               guide_file: str = "zork_strategy_guide.md",
                               model: str = "openai/gpt-4.1-mini",
                               temperature: float = 0.1,
                               base_url: Optional[str] = None,
                               api_key: Optional[str] = None) -> bool:
    """
    Create a strategic game guide from raw knowledge.
    
    Args:
        raw_knowledge_file: Path to raw knowledge base
        guide_file: Path for output strategic guide
        model: LLM model to use
        temperature: Temperature for summarization
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(raw_knowledge_file):
        print(f"Raw knowledge file {raw_knowledge_file} not found")
        return False
    
    print(f"Loading raw knowledge from {raw_knowledge_file}...")
    with open(raw_knowledge_file, 'r', encoding='utf-8') as f:
        raw_knowledge = f.read()
    
    print(f"Raw knowledge loaded: {len(raw_knowledge):,} characters")
    
    print(f"Summarizing with {model} (temperature={temperature})...")
    summarizer = KnowledgeSummarizer(
        model=model, 
        temperature=temperature,
        base_url=base_url,
        api_key=api_key
    )
    
    try:
        strategic_guide = summarizer.summarize_knowledge_base(raw_knowledge)
        
        # Save the strategic guide
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(strategic_guide)
        
        print(f"Strategic guide saved to {guide_file}")
        print(f"Guide length: {len(strategic_guide):,} characters")
        print(f"Compression ratio: {len(raw_knowledge)/len(strategic_guide):.1f}x")
        
        return True
        
    except Exception as e:
        print(f"Failed to create strategic guide: {e}")
        return False


def compare_guide_effectiveness(raw_knowledge_file: str = "knowledgebase.md",
                               guide_file: str = "zork_strategy_guide.md") -> Dict[str, Any]:
    """
    Compare the raw knowledge base with the strategic guide.
    
    Returns:
        Dictionary with comparison metrics
    """
    if not os.path.exists(raw_knowledge_file) or not os.path.exists(guide_file):
        return {"error": "Files not found"}
    
    with open(raw_knowledge_file, 'r') as f:
        raw_content = f.read()
    
    with open(guide_file, 'r') as f:
        guide_content = f.read()
    
    # Calculate metrics
    raw_chars = len(raw_content)
    guide_chars = len(guide_content)
    compression_ratio = raw_chars / guide_chars if guide_chars > 0 else 0
    
    raw_lines = len(raw_content.split('\n'))
    guide_lines = len(guide_content.split('\n'))
    
    # Count sections
    raw_sections = raw_content.count('##')
    guide_sections = guide_content.count('##')
    
    return {
        "raw_knowledge": {
            "characters": raw_chars,
            "lines": raw_lines,
            "sections": raw_sections
        },
        "strategic_guide": {
            "characters": guide_chars,
            "lines": guide_lines,
            "sections": guide_sections
        },
        "compression_ratio": compression_ratio,
        "size_reduction_percent": ((raw_chars - guide_chars) / raw_chars * 100) if raw_chars > 0 else 0
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create strategic game guide from raw knowledge")
    parser.add_argument("--input", default="knowledgebase.md", help="Input raw knowledge file")
    parser.add_argument("--output", default="zork_strategy_guide.md", help="Output strategic guide file")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for summarization")
    parser.add_argument("--compare", action="store_true", help="Compare raw vs. strategic guide")
    
    args = parser.parse_args()
    
    if args.compare:
        if os.path.exists(args.input) and os.path.exists(args.output):
            metrics = compare_guide_effectiveness(args.input, args.output)
            print(f"\n📊 Knowledge Base Comparison:")
            print(f"Raw Knowledge: {metrics['raw_knowledge']['characters']:,} chars, {metrics['raw_knowledge']['lines']} lines")
            print(f"Strategic Guide: {metrics['strategic_guide']['characters']:,} chars, {metrics['strategic_guide']['lines']} lines")
            print(f"Compression: {metrics['compression_ratio']:.1f}x smaller")
            print(f"Size reduction: {metrics['size_reduction_percent']:.1f}%")
        else:
            print("Files not found for comparison")
    else:
        success = create_strategic_game_guide(
            args.input, 
            args.output, 
            args.model, 
            args.temperature
        )
        
        if success:
            print(f"\n✅ Strategic guide creation successful!")
            # Show comparison
            metrics = compare_guide_effectiveness(args.input, args.output)
            if "error" not in metrics:
                print(f"📊 Compression achieved: {metrics['compression_ratio']:.1f}x smaller")
        else:
            print("❌ Strategic guide creation failed") 