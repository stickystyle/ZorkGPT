#!/usr/bin/env python3
"""
Integrated Knowledge Update Script

This script integrates with your existing ZorkGPT setup to update knowledge
and generate strategic guides using the same OpenAI client configuration.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ZorkAgent
from knowledge_extractor import update_knowledge_base
from knowledge_summarizer import KnowledgeSummarizer
import argparse


def update_knowledge_with_agent_config(
    log_file: str = "zork_episode_log.jsonl",
    kb_file: str = "knowledgebase.md",
    guide_file: str = "zork_strategy_guide.md",
    summarize: bool = True,
    verbose: bool = True
) -> bool:
    """
    Update knowledge base using the same configuration as the ZorkAgent.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print("🔄 Integrated Knowledge Update")
            print(f"  Log file: {log_file}")
            print(f"  Knowledge base: {kb_file}")
            if summarize:
                print(f"  Strategic guide: {guide_file}")
        
        # Step 1: Update raw knowledge base
        if verbose:
            print("\n📊 Step 1: Extracting knowledge from episode logs...")
        
        update_knowledge_base(log_file, kb_file)
        
        if verbose:
            print("✅ Raw knowledge base updated!")
        
        # Step 2: Generate strategic guide using agent's OpenAI client
        if summarize:
            if verbose:
                print("\n🧠 Step 2: Generating strategic guide...")
            
            # Create a temporary agent to get the OpenAI client configuration
            try:
                agent = ZorkAgent(
                    agent_model="google/gemini-2.5-flash-preview-05-20",  # Just for getting client config
                    episode_log_file="temp_log.txt",
                    json_log_file="temp_log.jsonl"
                )
                
                # Use the agent's client for summarization
                summarizer = KnowledgeSummarizer(
                    model="google/gemini-2.5-flash-preview-05-20",
                    temperature=0.1,
                    client=agent.client
                )
                
                # Load raw knowledge
                with open(kb_file, 'r', encoding='utf-8') as f:
                    raw_knowledge = f.read()
                
                if verbose:
                    print(f"  Raw knowledge: {len(raw_knowledge):,} characters")
                
                # Generate strategic guide
                strategic_guide = summarizer.summarize_knowledge_base(raw_knowledge)
                
                # Save strategic guide
                with open(guide_file, 'w', encoding='utf-8') as f:
                    f.write(strategic_guide)
                
                if verbose:
                    print(f"✅ Strategic guide generated!")
                    print(f"  Guide length: {len(strategic_guide):,} characters")
                    print(f"  Compression: {len(raw_knowledge)/len(strategic_guide):.1f}x smaller")
                
            except Exception as e:
                if verbose:
                    print(f"⚠️  Strategic guide generation failed: {e}")
                    print("   Continuing with raw knowledge base only...")
                return False
        
        if verbose:
            print(f"\n🎉 Knowledge update completed successfully!")
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Knowledge update failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Integrated knowledge update with ZorkAgent configuration")
    parser.add_argument(
        "--log-file",
        default="zork_episode_log.jsonl",
        help="Episode log file path"
    )
    parser.add_argument(
        "--kb-file", 
        default="knowledgebase.md",
        help="Raw knowledge base file path"
    )
    parser.add_argument(
        "--guide-file",
        default="zork_strategy_guide.md", 
        help="Strategic guide file path"
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip strategic guide generation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    success = update_knowledge_with_agent_config(
        log_file=args.log_file,
        kb_file=args.kb_file,
        guide_file=args.guide_file,
        summarize=not args.no_summarize,
        verbose=not args.quiet
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 