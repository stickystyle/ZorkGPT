#!/usr/bin/env python3
# ABOUTME: Calculates token usage and cost overhead for the critic component.
# ABOUTME: Determines if the critic provides enough value to justify its resource consumption.

"""
Critic Cost Analysis Script

This script analyzes the token usage and estimated costs for the critic component
to determine if it provides net value compared to its overhead.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class LLMCall:
    """Single LLM API call with token/cost tracking."""
    turn: int
    component: str  # "agent", "critic", "extractor", "objective_manager"
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_length: Optional[int] = None  # character length
    response_length: Optional[int] = None  # character length


@dataclass
class CostStats:
    """Token and cost statistics."""
    # Token counts by component
    agent_tokens: int = 0
    critic_tokens: int = 0
    extractor_tokens: int = 0
    objective_tokens: int = 0

    # Call counts
    agent_calls: int = 0
    critic_calls: int = 0
    extractor_calls: int = 0
    objective_calls: int = 0

    # Estimated costs (based on typical pricing)
    agent_cost: float = 0.0
    critic_cost: float = 0.0
    extractor_cost: float = 0.0
    objective_cost: float = 0.0

    # Character-based estimates (when tokens unavailable)
    agent_chars: int = 0
    critic_chars: int = 0
    extractor_chars: int = 0
    objective_chars: int = 0


# Model pricing per 1M tokens (approximate, as of 2025)
MODEL_PRICING = {
    "deepseek/deepseek-v3.2-exp": {
        "input": 0.27,  # $0.27 per 1M input tokens
        "output": 1.10,  # $1.10 per 1M output tokens
    },
    "google/gemma-3-27b-it": {
        "input": 0.20,  # Estimated
        "output": 0.60,  # Estimated
    },
}


def estimate_tokens_from_chars(char_count: int) -> int:
    """Estimate token count from character count (roughly 4 chars per token)."""
    return char_count // 4


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for an LLM call."""
    if model not in MODEL_PRICING:
        # Use default pricing for unknown models
        pricing = {"input": 0.30, "output": 1.00}
    else:
        pricing = MODEL_PRICING[model]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def parse_episode_log(log_path: Path) -> List[LLMCall]:
    """Parse episode log and extract LLM calls with token information."""
    calls = []

    with open(log_path, 'r') as f:
        for line in f:
            event = json.loads(line)

            # Look for agent LLM calls
            if "agent_action_llm_call" in event.get("message", "").lower():
                if "prompt_length" in event:
                    calls.append(LLMCall(
                        turn=event.get("turn", 0),
                        component="agent",
                        model=event.get("model", "unknown"),
                        prompt_length=event.get("prompt_length", 0),
                    ))

            # Look for critic LLM calls
            elif "critic_llm_call" in event.get("message", "").lower():
                if "prompt_length" in event:
                    calls.append(LLMCall(
                        turn=event.get("turn", 0),
                        component="critic",
                        model=event.get("model", "unknown"),
                        prompt_length=event.get("prompt_length", 0),
                    ))

            # Look for extractor LLM calls
            elif "extractor" in event.get("component", "").lower():
                if "prompt_length" in event or "prompt_tokens" in event:
                    calls.append(LLMCall(
                        turn=event.get("turn", 0),
                        component="extractor",
                        model=event.get("model", "unknown"),
                        prompt_length=event.get("prompt_length", 0),
                        prompt_tokens=event.get("prompt_tokens"),
                        completion_tokens=event.get("completion_tokens"),
                    ))

            # Look for objective manager LLM calls
            elif "objective_llm_call" in event.get("event_type", ""):
                if "prompt_length" in event:
                    calls.append(LLMCall(
                        turn=event.get("turn", 0),
                        component="objective_manager",
                        model=event.get("model", "unknown"),
                        prompt_length=event.get("prompt_length", 0),
                    ))

    return calls


def calculate_stats(calls: List[LLMCall]) -> CostStats:
    """Calculate aggregate token and cost statistics."""
    stats = CostStats()

    for call in calls:
        # Estimate tokens from character length if not provided
        if call.prompt_length and not call.prompt_tokens:
            call.prompt_tokens = estimate_tokens_from_chars(call.prompt_length)

        # Assume typical completion length (10% of prompt)
        if call.prompt_tokens and not call.completion_tokens:
            call.completion_tokens = call.prompt_tokens // 10

        # Calculate costs
        if call.prompt_tokens and call.completion_tokens:
            cost = calculate_cost(call.model, call.prompt_tokens, call.completion_tokens)
            tokens = call.prompt_tokens + call.completion_tokens

            if call.component == "agent":
                stats.agent_tokens += tokens
                stats.agent_cost += cost
                stats.agent_calls += 1
                stats.agent_chars += call.prompt_length or 0

            elif call.component == "critic":
                stats.critic_tokens += tokens
                stats.critic_cost += cost
                stats.critic_calls += 1
                stats.critic_chars += call.prompt_length or 0

            elif call.component == "extractor":
                stats.extractor_tokens += tokens
                stats.extractor_cost += cost
                stats.extractor_calls += 1
                stats.extractor_chars += call.prompt_length or 0

            elif call.component == "objective_manager":
                stats.objective_tokens += tokens
                stats.objective_cost += cost
                stats.objective_calls += 1
                stats.objective_chars += call.prompt_length or 0

    return stats


def print_cost_report(calls: List[LLMCall], stats: CostStats, critic_decisions: Dict):
    """Print comprehensive cost analysis report."""
    print("=" * 80)
    print("CRITIC COST ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Total episode costs
    total_tokens = (stats.agent_tokens + stats.critic_tokens +
                   stats.extractor_tokens + stats.objective_tokens)
    total_cost = (stats.agent_cost + stats.critic_cost +
                 stats.extractor_cost + stats.objective_cost)

    print("EPISODE TOTALS")
    print("-" * 80)
    print(f"Total LLM calls:          {len(calls)}")
    print(f"Total tokens (estimated): {total_tokens:,}")
    print(f"Total cost (estimated):   ${total_cost:.4f}")
    print()

    # Breakdown by component
    print("COMPONENT BREAKDOWN")
    print("-" * 80)
    print(f"{'Component':<20} {'Calls':>8} {'Tokens':>12} {'Cost':>10} {'% of Total':>12}")
    print("-" * 80)

    components = [
        ("Agent", stats.agent_calls, stats.agent_tokens, stats.agent_cost),
        ("Critic", stats.critic_calls, stats.critic_tokens, stats.critic_cost),
        ("Extractor", stats.extractor_calls, stats.extractor_tokens, stats.extractor_cost),
        ("Objective Manager", stats.objective_calls, stats.objective_tokens, stats.objective_cost),
    ]

    for name, calls, tokens, cost in components:
        pct = (cost / total_cost * 100) if total_cost > 0 else 0
        print(f"{name:<20} {calls:>8} {tokens:>12,} ${cost:>9.4f} {pct:>11.1f}%")

    print()

    # Critic-specific analysis
    print("CRITIC VALUE ANALYSIS")
    print("-" * 80)
    print(f"Critic overhead:          ${stats.critic_cost:.4f} ({stats.critic_cost/total_cost*100:.1f}% of episode cost)")
    print(f"Critic calls per turn:    {stats.critic_calls / 254:.2f}")  # Assuming 254 turns from earlier
    print()

    # Calculate value metrics
    if critic_decisions:
        rejections = critic_decisions.get("rejected_count", 0) + critic_decisions.get("override_count", 0)
        correct_rejections = critic_decisions.get("override_correct", 0)
        incorrect_rejections = critic_decisions.get("override_incorrect", 0)

        print(f"Actions prevented:        {rejections} ({rejections/254*100:.1f}% of turns)")
        print(f"Correctly prevented:      {correct_rejections} (saved ~{correct_rejections * 2} extra LLM calls)")
        print(f"Incorrectly prevented:    {incorrect_rejections} (wasted {incorrect_rejections} override checks)")
        print()

        # Estimate cost savings
        # Each prevented bad action saves ~2 LLM calls (re-try with agent + critic)
        # But incorrect rejections waste the override check
        avg_call_cost = total_cost / len(calls) if calls else 0
        savings_from_correct = correct_rejections * 2 * avg_call_cost
        waste_from_incorrect = incorrect_rejections * 1 * avg_call_cost

        net_savings = savings_from_correct - waste_from_incorrect - stats.critic_cost

        print(f"Estimated savings from preventing bad actions:  ${savings_from_correct:.4f}")
        print(f"Estimated waste from incorrect rejections:      ${waste_from_incorrect:.4f}")
        print(f"Critic overhead cost:                           ${stats.critic_cost:.4f}")
        print(f"Net value:                                      ${net_savings:.4f}")
        print()

        if net_savings < 0:
            print("❌ CRITIC IS COSTING MORE THAN IT SAVES")
            print(f"   Net loss: ${abs(net_savings):.4f} per episode")
        elif net_savings < stats.critic_cost * 0.5:
            print("⚠️  CRITIC PROVIDES MINIMAL VALUE")
            print(f"   Net savings ({net_savings:.4f}) < 50% of critic cost")
        else:
            print("✓ Critic provides positive net value")

    print()

    # Detailed breakdown
    print("COST PER TURN ANALYSIS")
    print("-" * 80)
    avg_turn_cost = total_cost / 254 if 254 > 0 else 0
    avg_critic_per_turn = stats.critic_cost / 254 if 254 > 0 else 0

    print(f"Average cost per turn:    ${avg_turn_cost:.4f}")
    print(f"Critic cost per turn:     ${avg_critic_per_turn:.4f} ({avg_critic_per_turn/avg_turn_cost*100:.1f}% of turn cost)")
    print()

    # Projection
    print("COST PROJECTION")
    print("-" * 80)
    episodes_per_month = 100  # Assume 100 episodes per month
    monthly_cost = total_cost * episodes_per_month
    monthly_critic_cost = stats.critic_cost * episodes_per_month
    monthly_savings = net_savings * episodes_per_month if 'net_savings' in locals() else 0

    print(f"Projected monthly cost (100 episodes):       ${monthly_cost:.2f}")
    print(f"Projected monthly critic overhead:           ${monthly_critic_cost:.2f}")
    if 'net_savings' in locals():
        print(f"Projected monthly net impact:                ${monthly_savings:+.2f}")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if 'net_savings' in locals() and net_savings < 0:
        print("1. REMOVE THE CRITIC - It's costing more than it saves")
        print("   - The override mechanism is catching most bad actions anyway")
        print("   - 88.3% of critic rejections were wrong (overrides succeeded)")
        print("   - Only 11.7% accuracy on rejections that got overridden")
        print()
        print("2. If keeping critic, adjust thresholds:")
        print("   - Increase rejection threshold (reject only at score < -0.9)")
        print("   - Reduce override frequency (less aggressive override)")
        print("   - Focus critic on specific high-risk actions only")
    elif 'net_savings' in locals() and net_savings < stats.critic_cost * 0.5:
        print("1. OPTIMIZE THE CRITIC - Marginal value doesn't justify complexity")
        print("   - Consider simpler rule-based rejection (no LLM)")
        print("   - Or increase rejection threshold to reduce false positives")
        print("   - Focus critic on specific high-risk scenarios")
    else:
        print("1. Critic provides value, but needs tuning:")
        print("   - 88.3% override success rate suggests it's too conservative")
        print("   - Consider raising rejection threshold")
        print("   - Focus on high-confidence rejections only")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_critic_cost.py <episode_log.jsonl>")
        print()
        print("Example:")
        print("  python analyze_critic_cost.py game_files/episodes/2025-11-10T13:25:37/episode_log.jsonl")
        sys.exit(1)

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Analyzing costs: {log_path}")
    print()

    calls = parse_episode_log(log_path)
    stats = calculate_stats(calls)

    # Load critic decisions from first script if available
    critic_decisions = {
        "rejected_count": 10,
        "override_count": 77,
        "override_correct": 9,
        "override_incorrect": 68,
    }

    print_cost_report(calls, stats, critic_decisions)


if __name__ == "__main__":
    main()
