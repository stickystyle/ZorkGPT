"""
Knowledge Base Extractor for ZorkGPT

This module analyzes episode logs to extract and maintain a knowledge base
of useful information about the Zork world that can help future episodes.
"""

import json
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import os
import re


@dataclass
class LocationKnowledge:
    """Knowledge about a specific location."""
    name: str
    exits: Set[str]
    objects: Set[str]
    characters: Set[str]
    successful_actions: Set[str]
    failed_actions: Set[str]
    important_notes: List[str]
    visit_count: int = 0
    

@dataclass
class ItemKnowledge:
    """Knowledge about a specific item or object."""
    name: str
    found_locations: Set[str]
    successful_interactions: Set[str]
    failed_interactions: Set[str]
    properties: Set[str]  # e.g., "takeable", "openable", "readable"
    contents: Set[str]  # for containers
    notes: List[str]


@dataclass
class GameMechanicKnowledge:
    """Knowledge about game mechanics and patterns."""
    puzzle_solutions: Dict[str, str]
    dangerous_actions: Set[str]
    score_gaining_actions: Set[str]
    required_items: Dict[str, Set[str]]  # action -> required items
    action_sequences: List[Tuple[str, ...]]  # successful action sequences
    

class KnowledgeExtractor:
    """Extracts and maintains knowledge from episode logs."""
    
    def __init__(self):
        self.locations: Dict[str, LocationKnowledge] = {}
        self.items: Dict[str, ItemKnowledge] = {}
        self.mechanics: GameMechanicKnowledge = GameMechanicKnowledge(
            puzzle_solutions={},
            dangerous_actions=set(),
            score_gaining_actions=set(),
            required_items=defaultdict(set),
            action_sequences=[]
        )
        self.common_mistakes: List[str] = []
        self.effective_strategies: List[str] = []
        
    def analyze_episode_logs(self, json_log_file: str) -> None:
        """Analyze a JSON log file to extract knowledge."""
        episodes = self._parse_episodes_from_logs(json_log_file)
        
        for episode in episodes:
            self._analyze_episode(episode)
    
    def _parse_episodes_from_logs(self, json_log_file: str) -> List[List[Dict]]:
        """Parse episodes from JSON log file."""
        episodes = []
        current_episode = []
        current_episode_id = None
        
        with open(json_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    episode_id = log_entry.get('episode_id')
                    
                    if episode_id != current_episode_id:
                        if current_episode:
                            episodes.append(current_episode)
                        current_episode = []
                        current_episode_id = episode_id
                    
                    current_episode.append(log_entry)
                    
                except json.JSONDecodeError:
                    continue
        
        if current_episode:
            episodes.append(current_episode)
            
        return episodes
    
    def _analyze_episode(self, episode_logs: List[Dict]) -> None:
        """Analyze a single episode to extract knowledge."""
        episode_data = self._extract_episode_data(episode_logs)
        
        if not episode_data:
            return
            
        # Analyze locations
        self._analyze_locations(episode_data)
        
        # Analyze items and objects
        self._analyze_items(episode_data)
        
        # Analyze game mechanics
        self._analyze_mechanics(episode_data)
        
        # Analyze strategies and mistakes
        self._analyze_strategies_and_mistakes(episode_data)
    
    def _extract_episode_data(self, episode_logs: List[Dict]) -> Optional[Dict]:
        """Extract structured data from episode logs."""
        turns = []
        episode_info = {}
        current_turn = {}
        
        for log_entry in episode_logs:
            event_type = log_entry.get('event_type', '')
            
            if event_type == 'episode_start':
                episode_info = {
                    'episode_id': log_entry.get('episode_id'),
                    'agent_model': log_entry.get('agent_model'),
                    'start_time': log_entry.get('timestamp')
                }
            elif event_type == 'episode_end':
                episode_info.update({
                    'final_score': log_entry.get('zork_score', 0),
                    'max_score': log_entry.get('max_score', 585),
                    'turn_count': log_entry.get('turn_count', 0),
                    'total_reward': log_entry.get('total_reward', 0)
                })
            elif event_type == 'agent_action':
                # Start a new turn when we see an agent action
                if current_turn:
                    turns.append(current_turn)
                current_turn = {'agent_action': log_entry}
            elif event_type in ['critic_evaluation', 'zork_response', 'extracted_info']:
                # Add to current turn
                if current_turn:  # Only if we have a turn started
                    current_turn[event_type] = log_entry
                    
        # Don't forget the last turn
        if current_turn:
            turns.append(current_turn)
        
        if not episode_info:
            return None
            
        return {
            'episode_info': episode_info,
            'turns': turns
        }
    

    
    def _analyze_locations(self, episode_data: Dict) -> None:
        """Analyze location information from episode."""
        for turn in episode_data['turns']:
            extracted_info = turn.get('extracted_info', {})
            
            if not extracted_info:
                continue
                
            location_name = extracted_info.get('extracted_info', {}).get('current_location_name')
            if not location_name:
                continue
                
            # Normalize location name
            location_name = self._normalize_location_name(location_name)
            
            if location_name not in self.locations:
                self.locations[location_name] = LocationKnowledge(
                    name=location_name,
                    exits=set(),
                    objects=set(),
                    characters=set(),
                    successful_actions=set(),
                    failed_actions=set(),
                    important_notes=[]
                )
            
            location = self.locations[location_name]
            location.visit_count += 1
            
            # Update exits, objects, characters
            extracted_data = extracted_info.get('extracted_info', {})
            location.exits.update(extracted_data.get('exits', []))
            location.objects.update(extracted_data.get('visible_objects', []))
            location.characters.update(extracted_data.get('visible_characters', []))
            
            # Analyze action success/failure
            agent_action = turn.get('agent_action', {}).get('agent_action', '')
            critic_score = turn.get('critic_evaluation', {}).get('critic_score', 0)
            zork_response = turn.get('zork_response', {}).get('zork_response', '')
            
            if agent_action:
                if self._is_action_successful(zork_response, critic_score):
                    location.successful_actions.add(agent_action)
                elif self._is_action_failed(zork_response):
                    location.failed_actions.add(agent_action)
    
    def _analyze_items(self, episode_data: Dict) -> None:
        """Analyze item and object information."""
        for turn in episode_data['turns']:
            extracted_info = turn.get('extracted_info', {})
            extracted_data = extracted_info.get('extracted_info', {})
            location_name = extracted_data.get('current_location_name', '')
            location_name = self._normalize_location_name(location_name)
            
            for obj in extracted_data.get('visible_objects', []):
                if obj not in self.items:
                    self.items[obj] = ItemKnowledge(
                        name=obj,
                        found_locations=set(),
                        successful_interactions=set(),
                        failed_interactions=set(),
                        properties=set(),
                        contents=set(),
                        notes=[]
                    )
                
                self.items[obj].found_locations.add(location_name)
                
                # Analyze interactions with this object
                agent_action = turn.get('agent_action', {}).get('agent_action', '')
                if obj.lower() in agent_action.lower():
                    zork_response = turn.get('zork_response', {}).get('zork_response', '')
                    if self._is_action_successful(zork_response):
                        self.items[obj].successful_interactions.add(agent_action)
                    elif self._is_action_failed(zork_response):
                        self.items[obj].failed_interactions.add(agent_action)
    
    def _analyze_mechanics(self, episode_data: Dict) -> None:
        """Analyze game mechanics and patterns."""
        prev_score = 0
        
        for turn in episode_data['turns']:
            agent_action = turn.get('agent_action', {}).get('agent_action', '')
            zork_response = turn.get('zork_response', {}).get('zork_response', '')
            extracted_info = turn.get('extracted_info', {}).get('extracted_info', {})
            
            # Check for score changes
            current_score = self._extract_score_from_response(zork_response)
            if current_score > prev_score and agent_action:
                self.mechanics.score_gaining_actions.add(agent_action)
            prev_score = current_score
            
            # Check for dangerous outcomes (death, injury, combat)
            if self._is_dangerous_outcome(zork_response):
                # Record dangerous action if there was one
                if agent_action:
                    self.mechanics.dangerous_actions.add(agent_action)
                
                # Always mark the location as dangerous regardless of action
                location_name = extracted_info.get('current_location_name')
                if location_name:
                    location_name = self._normalize_location_name(location_name)
                    # Ensure location exists in our tracking
                    if location_name not in self.locations:
                        self.locations[location_name] = LocationKnowledge(
                            name=location_name,
                            exits=set(),
                            objects=set(),
                            characters=set(),
                            successful_actions=set(),
                            failed_actions=set(),
                            important_notes=[]
                        )
                    
                    # Add danger note
                    danger_desc = self._extract_danger_description(zork_response)
                    if agent_action:
                        danger_note = f"DANGER: '{agent_action}' resulted in {danger_desc}"
                    else:
                        danger_note = f"DANGER: {danger_desc} encountered here"
                    
                    if danger_note not in self.locations[location_name].important_notes:
                        self.locations[location_name].important_notes.append(danger_note)
                            
    def _extract_danger_description(self, zork_response: str) -> str:
        """Extract a brief description of what danger occurred."""
        response_lower = zork_response.lower()
        
        if "troll" in response_lower:
            return "troll attack/death"
        elif "grue" in response_lower:
            return "eaten by grue"
        elif "died" in response_lower or "dead" in response_lower:
            return "death"
        elif "killed" in response_lower:
            return "killed"
        else:
            return "dangerous outcome"
    
    def _analyze_strategies_and_mistakes(self, episode_data: Dict) -> None:
        """Analyze effective strategies and common mistakes."""
        episode_info = episode_data['episode_info']
        final_score = episode_info.get('final_score', 0)
        turn_count = episode_info.get('turn_count', 0)
        
        # Analyze high-performing episodes for strategies
        if final_score > 30 or (final_score > 10 and turn_count < 50):
            self._extract_strategies(episode_data)
        
        # Analyze low-performing episodes for mistakes  
        if final_score == 0 or turn_count > 150:
            self._extract_mistakes(episode_data)
    
    def _extract_strategies(self, episode_data: Dict) -> None:
        """Extract effective strategies from successful episodes."""
        action_sequence = []
        for turn in episode_data['turns'][:20]:  # Focus on early game
            agent_action = turn.get('agent_action', {}).get('agent_action', '')
            if agent_action:
                action_sequence.append(agent_action)
        
        if len(action_sequence) > 5:
            # Look for patterns in successful early sequences
            strategy_description = f"Successful early sequence: {' -> '.join(action_sequence[:10])}"
            if strategy_description not in self.effective_strategies:
                self.effective_strategies.append(strategy_description)
    
    def _extract_mistakes(self, episode_data: Dict) -> None:
        """Extract common mistakes from failed episodes."""
        repeated_actions = Counter()
        for turn in episode_data['turns']:
            agent_action = turn.get('agent_action', {}).get('agent_action', '')
            if agent_action:
                repeated_actions[agent_action] += 1
        
        # Find frequently repeated actions
        for action, count in repeated_actions.items():
            if count > 5:
                mistake = f"Avoid repeating '{action}' excessively (repeated {count} times in failed episode)"
                if mistake not in self.common_mistakes:
                    self.common_mistakes.append(mistake)
    
    def _normalize_location_name(self, name: str) -> str:
        """Normalize location names for consistent storage."""
        if not name:
            return "Unknown"
        
        # Clean up common variations
        name = re.sub(r'\s+', ' ', name.strip())
        name = re.sub(r'[_\n\r]+', ' ', name)
        
        # Convert to title case for consistency
        return name.title()
    
    def _is_action_successful(self, zork_response: str, critic_score: float = 0) -> bool:
        """Determine if an action was successful."""
        if not zork_response:
            return False
            
        success_indicators = [
            "taken", "opened", "closed", "unlocked", "your score has just gone up",
            "the door opens", "you are now", "you have", "you can see",
            "you found", "you discover"
        ]
        
        return any(indicator in zork_response.lower() for indicator in success_indicators) or critic_score > 0.5
    
    def _is_action_failed(self, zork_response: str) -> bool:
        """Determine if an action clearly failed."""
        if not zork_response:
            return False
            
        failure_indicators = [
            "i don't understand", "you can't", "that's not possible",
            "there is no", "nothing happens", "you don't see",
            "that doesn't work", "impossible", "you cannot"
        ]
        
        return any(indicator in zork_response.lower() for indicator in failure_indicators)
    
    def _is_dangerous_outcome(self, zork_response: str) -> bool:
        """Check if the response indicates a dangerous outcome."""
        if not zork_response:
            return False
            
        danger_indicators = [
            "you have died", "you are dead", "killed", "eaten",
            "fatal", "crushed", "burned", "drowned", "death",
            "troll", "brandishing", "axe", "bloody", "grue",
            "likely to be eaten", "psychotics", "suicidal",
            "puts you to death", "too much for you", "afraid",
            "installed in the land of the living dead"
        ]
        
        return any(indicator in zork_response.lower() for indicator in danger_indicators)
    
    def _extract_score_from_response(self, zork_response: str) -> int:
        """Extract score from Zork response."""
        if not zork_response:
            return 0
            
        # Look for score patterns
        score_match = re.search(r'score.*?(\d+)', zork_response.lower())
        if score_match:
            return int(score_match.group(1))
            
        return 0
    
    def generate_knowledge_base(self) -> str:
        """Generate a markdown knowledge base from extracted knowledge."""
        kb_content = []
        
        # Header
        kb_content.append("# ZorkGPT Knowledge Base")
        kb_content.append(f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        kb_content.append("This knowledge base contains important information learned from previous episodes to help improve future performance.\n")
        
        # Location Knowledge
        if self.locations:
            kb_content.append("## Location Knowledge\n")
            for location_name, location in sorted(self.locations.items(), key=lambda x: x[1].visit_count, reverse=True):
                if location.visit_count < 2:  # Skip rarely visited locations
                    continue
                    
                kb_content.append(f"### {location_name}")
                kb_content.append(f"*Visited {location.visit_count} times*\n")
                
                if location.exits:
                    kb_content.append(f"**Exits:** {', '.join(sorted(location.exits))}")
                
                if location.objects:
                    kb_content.append(f"**Objects found:** {', '.join(sorted(location.objects))}")
                
                if location.successful_actions:
                    kb_content.append(f"**Successful actions:** {', '.join(sorted(location.successful_actions))}")
                
                if location.failed_actions:
                    kb_content.append(f"**Failed actions (avoid):** {', '.join(sorted(location.failed_actions))}")
                
                if location.important_notes:
                    kb_content.append(f"**Important notes:** {'; '.join(location.important_notes)}")
                
                kb_content.append("")
        
        # Item Knowledge
        if self.items:
            kb_content.append("## Item and Object Knowledge\n")
            for item_name, item in sorted(self.items.items()):
                if len(item.found_locations) < 2 and not item.successful_interactions:
                    continue
                    
                kb_content.append(f"### {item_name}")
                
                if item.found_locations:
                    kb_content.append(f"**Found in:** {', '.join(sorted(item.found_locations))}")
                
                if item.successful_interactions:
                    kb_content.append(f"**Successful interactions:** {', '.join(sorted(item.successful_interactions))}")
                
                if item.failed_interactions:
                    kb_content.append(f"**Failed interactions (avoid):** {', '.join(sorted(item.failed_interactions))}")
                
                kb_content.append("")
        
        # Game Mechanics
        kb_content.append("## Game Mechanics and Patterns\n")
        
        if self.mechanics.score_gaining_actions:
            kb_content.append("### Score-Gaining Actions")
            kb_content.append("These actions have been observed to increase the game score:")
            for action in sorted(self.mechanics.score_gaining_actions):
                kb_content.append(f"- {action}")
            kb_content.append("")
        
        if self.mechanics.dangerous_actions:
            kb_content.append("### Dangerous Actions")
            kb_content.append("These actions have led to death or injury - use with extreme caution:")
            for action in sorted(self.mechanics.dangerous_actions):
                kb_content.append(f"- {action}")
            kb_content.append("")
        
        # Strategies
        if self.effective_strategies:
            kb_content.append("## Effective Strategies\n")
            for strategy in self.effective_strategies[:10]:  # Limit to top 10
                kb_content.append(f"- {strategy}")
            kb_content.append("")
        
        # Common Mistakes
        if self.common_mistakes:
            kb_content.append("## Common Mistakes to Avoid\n")
            for mistake in self.common_mistakes[:10]:  # Limit to top 10
                kb_content.append(f"- {mistake}")
            kb_content.append("")
        
        kb_content.append("---")
        kb_content.append("*This knowledge base is automatically generated from episode logs. Use this information to make better decisions in future episodes.*")
        
        return "\n".join(kb_content)
    
    def save_knowledge_base(self, filename: str = "knowledgebase.md") -> None:
        """Save the knowledge base to a markdown file."""
        kb_content = self.generate_knowledge_base()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(kb_content)
        print(f"Knowledge base saved to {filename}")


def update_knowledge_base(json_log_file: str = "zork_episode_log.jsonl", 
                         knowledge_base_file: str = "knowledgebase.md") -> None:
    """Update the knowledge base with new episode data."""
    if not os.path.exists(json_log_file):
        print(f"Log file {json_log_file} not found")
        return
        
    extractor = KnowledgeExtractor()
    
    # Load existing knowledge base if it exists
    if os.path.exists(knowledge_base_file):
        print(f"Updating existing knowledge base: {knowledge_base_file}")
    else:
        print(f"Creating new knowledge base: {knowledge_base_file}")
    
    print(f"Analyzing episodes from {json_log_file}...")
    extractor.analyze_episode_logs(json_log_file)
    
    print(f"Extracted knowledge for:")
    print(f"  - {len(extractor.locations)} locations")
    print(f"  - {len(extractor.items)} items/objects")
    print(f"  - {len(extractor.mechanics.score_gaining_actions)} score-gaining actions")
    print(f"  - {len(extractor.mechanics.dangerous_actions)} dangerous actions")
    print(f"  - {len(extractor.effective_strategies)} effective strategies")
    print(f"  - {len(extractor.common_mistakes)} common mistakes")
    
    extractor.save_knowledge_base(knowledge_base_file)


if __name__ == "__main__":
    # Example usage
    update_knowledge_base() 