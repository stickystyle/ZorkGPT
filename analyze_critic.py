#!/usr/bin/env python3
# ABOUTME: Analyzes critic decisions from episode logs to determine if the critic adds value.
# ABOUTME: Extracts pass/fail/override patterns and calculates cost/benefit metrics.

"""
Critic Analysis Script

This script analyzes the critic's decision-making in a ZorkGPT episode to determine:
1. How often does the critic reject actions?
2. How often are those rejections overridden?
3. When overridden, was the critic right or wrong?
4. What's the token/cost overhead of running the critic?
5. Is the critic providing net value?
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class CriticDecision:
    """Single critic decision with outcome tracking."""
    turn: int
    action: str
    agent_reasoning: str
    critic_score: float
    critic_confidence: float
    was_overridden: bool
    override_reason: Optional[str] = None

    # Outcome tracking
    zork_response: Optional[str] = None
    action_succeeded: Optional[bool] = None  # Did the action work?
    location_changed: Optional[bool] = None  # Did location change?
    score_changed: Optional[bool] = None  # Did score increase?

    # Penalties applied
    location_penalty_applied: bool = False
    penalty_amount: float = 0.0
    base_confidence: Optional[float] = None


@dataclass
class CriticStats:
    """Aggregate statistics about critic performance."""
    total_decisions: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    override_count: int = 0

    # Override outcomes
    override_correct: int = 0  # Critic rejected, override failed → critic was right
    override_incorrect: int = 0  # Critic rejected, override succeeded → critic was wrong
    override_unclear: int = 0  # Can't determine outcome

    # Action success rates
    accepted_success: int = 0
    accepted_failure: int = 0
    rejected_would_fail: int = 0  # If we could test rejected actions

    # Score/confidence distribution
    score_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidence_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Location penalty analysis
    penalties_applied: int = 0
    avg_penalty_amount: float = 0.0


def parse_episode_log(log_path: Path) -> List[CriticDecision]:
    """Parse episode log and extract critic decisions with outcomes."""
    decisions = []

    # Build a map of turns to events for context
    turn_events = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            event = json.loads(line)
            if 'turn' in event:
                turn_events[event['turn']].append(event)

    # Extract critic decisions and match with outcomes
    for turn, events in sorted(turn_events.items()):
        if turn == 0:
            continue  # Skip initialization

        decision = None
        override_event = None
        penalty_event = None

        for event in events:
            # Look for final action selection (critic decision)
            if event.get('event_type') == 'final_action_selection':
                decision = CriticDecision(
                    turn=turn,
                    action=event.get('agent_action', ''),
                    agent_reasoning=event.get('agent_reasoning', ''),
                    critic_score=event.get('critic_score', 0.0),
                    critic_confidence=event.get('critic_confidence', 0.0),
                    was_overridden=event.get('was_overridden', False)
                )

            # Look for override event
            elif event.get('event_type') == 'critic_override':
                override_event = event

            # Look for location penalty
            elif event.get('event_type') == 'location_penalty_applied':
                penalty_event = event

            # Look for Zork response
            elif event.get('event_type') == 'zork_response' and decision:
                decision.zork_response = event.get('zork_response', '')

        if decision:
            # Add override details
            if override_event:
                decision.override_reason = override_event.get('reason', '')

            # Add penalty details
            if penalty_event:
                decision.location_penalty_applied = True
                decision.base_confidence = penalty_event.get('base_confidence', 0.0)
                decision.penalty_amount = (
                    penalty_event.get('base_confidence', 0.0) -
                    penalty_event.get('adjusted_confidence', 0.0)
                )

            # Analyze action success
            if decision.zork_response:
                decision.action_succeeded = analyze_action_success(decision.zork_response)

            decisions.append(decision)

    return decisions


def analyze_action_success(zork_response: str) -> bool:
    """Determine if an action succeeded based on Zork's response."""
    zork_lower = zork_response.lower()

    # Failure indicators
    failure_phrases = [
        "you can't",
        "you couldn't",
        "impossible",
        "don't see",
        "don't have",
        "not open",
        "won't open",
        "locked",
        "too dark",
        "fatal error",
        "i don't understand",
        "that's not a verb",
        "you died",
        "is boarded",
        "can't remove"
    ]

    for phrase in failure_phrases:
        if phrase in zork_lower:
            return False

    # Neutral/success indicators
    # If no failure detected, consider it a success
    return True


def calculate_stats(decisions: List[CriticDecision]) -> CriticStats:
    """Calculate aggregate statistics from decisions."""
    stats = CriticStats()
    stats.total_decisions = len(decisions)

    penalty_sum = 0.0

    for d in decisions:
        # Basic categorization
        if d.was_overridden:
            stats.override_count += 1

            # Analyze override correctness
            if d.action_succeeded is not None:
                if d.action_succeeded:
                    # Action succeeded after override → critic was wrong to reject
                    stats.override_incorrect += 1
                else:
                    # Action failed after override → critic was right to reject
                    stats.override_correct += 1
            else:
                stats.override_unclear += 1

        elif d.critic_score < 0:
            # Rejected and NOT overridden
            stats.rejected_count += 1

        else:
            # Accepted
            stats.accepted_count += 1

            if d.action_succeeded is not None:
                if d.action_succeeded:
                    stats.accepted_success += 1
                else:
                    stats.accepted_failure += 1

        # Score distribution (bucketed)
        score_bucket = f"{int(d.critic_score * 10) / 10:.1f}"
        stats.score_distribution[score_bucket] += 1

        # Confidence distribution (bucketed)
        conf_bucket = f"{int(d.critic_confidence * 10) / 10:.1f}"
        stats.confidence_distribution[conf_bucket] += 1

        # Penalty tracking
        if d.location_penalty_applied:
            stats.penalties_applied += 1
            penalty_sum += d.penalty_amount

    if stats.penalties_applied > 0:
        stats.avg_penalty_amount = penalty_sum / stats.penalties_applied

    return stats


def print_report(decisions: List[CriticDecision], stats: CriticStats):
    """Print comprehensive analysis report."""
    print("=" * 80)
    print("CRITIC ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Overview
    print("OVERVIEW")
    print("-" * 80)
    print(f"Total decisions analyzed: {stats.total_decisions}")
    print(f"Accepted (score ≥ 0):     {stats.accepted_count} ({stats.accepted_count/stats.total_decisions*100:.1f}%)")
    print(f"Rejected (score < 0):     {stats.rejected_count} ({stats.rejected_count/stats.total_decisions*100:.1f}%)")
    print(f"Overridden:               {stats.override_count} ({stats.override_count/stats.total_decisions*100:.1f}%)")
    print()

    # Override analysis
    print("OVERRIDE ANALYSIS")
    print("-" * 80)
    if stats.override_count > 0:
        print(f"Total overrides:          {stats.override_count}")
        print(f"Critic was RIGHT:         {stats.override_correct} ({stats.override_correct/stats.override_count*100:.1f}%)")
        print(f"Critic was WRONG:         {stats.override_incorrect} ({stats.override_incorrect/stats.override_count*100:.1f}%)")
        print(f"Unclear outcome:          {stats.override_unclear} ({stats.override_unclear/stats.override_count*100:.1f}%)")
        print()

        # Show override details
        print("Override Details:")
        for d in decisions:
            if d.was_overridden:
                outcome = "SUCCESS" if d.action_succeeded else "FAILURE" if d.action_succeeded is False else "UNCLEAR"
                verdict = "CRITIC WRONG" if d.action_succeeded else "CRITIC RIGHT" if d.action_succeeded is False else "UNCLEAR"
                print(f"  Turn {d.turn:3d}: '{d.action}' → {outcome} → {verdict}")
                print(f"             Score: {d.critic_score:.2f}, Reason: {d.override_reason}")
                print(f"             Response: {d.zork_response[:80]}...")
                print()
    else:
        print("No overrides occurred in this episode.")
    print()

    # Action success analysis
    print("ACTION SUCCESS ANALYSIS")
    print("-" * 80)
    if stats.accepted_count > 0:
        print(f"Accepted actions that succeeded: {stats.accepted_success}")
        print(f"Accepted actions that failed:    {stats.accepted_failure}")
        if stats.accepted_success + stats.accepted_failure > 0:
            success_rate = stats.accepted_success / (stats.accepted_success + stats.accepted_failure) * 100
            print(f"Accepted action success rate:    {success_rate:.1f}%")
    print()

    # Location penalty analysis
    print("LOCATION PENALTY ANALYSIS")
    print("-" * 80)
    print(f"Penalties applied:        {stats.penalties_applied}/{stats.total_decisions} ({stats.penalties_applied/stats.total_decisions*100:.1f}%)")
    if stats.penalties_applied > 0:
        print(f"Average penalty amount:   {stats.avg_penalty_amount:.2f}")
        print()
        print("Penalty Impact Examples:")
        for d in decisions:
            if d.location_penalty_applied and d.base_confidence:
                print(f"  Turn {d.turn:3d}: {d.base_confidence:.2f} → {d.critic_confidence:.2f} (penalty: -{d.penalty_amount:.2f})")
    print()

    # Score distribution
    print("CRITIC SCORE DISTRIBUTION")
    print("-" * 80)
    for score in sorted(stats.score_distribution.keys(), key=float):
        count = stats.score_distribution[score]
        bar = "█" * int(count / stats.total_decisions * 50)
        print(f"  {score:>5s}: {bar} {count}")
    print()

    # Confidence distribution
    print("CONFIDENCE DISTRIBUTION")
    print("-" * 80)
    for conf in sorted(stats.confidence_distribution.keys(), key=float):
        count = stats.confidence_distribution[conf]
        bar = "█" * int(count / stats.total_decisions * 50)
        print(f"  {conf:>5s}: {bar} {count}")
    print()

    # KEY FINDINGS
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    rejection_rate = (stats.rejected_count + stats.override_count) / stats.total_decisions * 100
    print(f"1. Critic rejection rate: {rejection_rate:.1f}%")

    if stats.override_count > 0:
        override_effectiveness = stats.override_correct / stats.override_count * 100
        print(f"2. When overridden, critic was correct {override_effectiveness:.1f}% of the time")

        if override_effectiveness > 50:
            print("   → This suggests the override mechanism is TOO AGGRESSIVE")
        else:
            print("   → This suggests the override mechanism is working well")

    if stats.accepted_failure > 0:
        false_positive_rate = stats.accepted_failure / (stats.accepted_success + stats.accepted_failure) * 100
        print(f"3. Critic's false positive rate: {false_positive_rate:.1f}%")
        print(f"   (Accepted actions that still failed)")

    print()
    print("RECOMMENDATION:")
    if stats.override_count > 5 and stats.override_correct > stats.override_incorrect:
        print("❌ The critic is being overridden frequently, but was usually RIGHT.")
        print("   Consider: Disabling or adjusting the override mechanism.")
    elif stats.rejected_count > stats.total_decisions * 0.5:
        print("⚠️  The critic is rejecting >50% of actions - too conservative?")
        print("   Consider: Adjusting rejection threshold or removing critic.")
    elif stats.override_incorrect > stats.override_correct:
        print("⚠️  Most overrides were correct - the critic is too conservative.")
        print("   Consider: Adjusting critic scoring or increasing override threshold.")
    else:
        print("✓ Critic appears to be functioning reasonably.")
        print("  Still need token/cost analysis to determine net value.")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_critic.py <episode_log.jsonl>")
        print()
        print("Example:")
        print("  python analyze_critic.py game_files/episodes/2025-11-10T13:25:37/episode_log.jsonl")
        sys.exit(1)

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Analyzing: {log_path}")
    print()

    decisions = parse_episode_log(log_path)
    stats = calculate_stats(decisions)
    print_report(decisions, stats)


if __name__ == "__main__":
    main()
