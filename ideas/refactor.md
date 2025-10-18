  # Single-Stage Knowledge Base Generation Refactor Plan

  ## Executive Summary

  This document outlines a complete refactoring of the ZorkGPT knowledge base
     generatio
         + n system from a complex 3-stage process to a simpler, more efficient single-stage
     appr
         + oach while maintaining persistent wisdom integration.

  ## Background & Motivation

  ### Current System Issues
  1. **3-Stage Process** causes information loss through multiple abstraction layers:
     - Stage 1: Quality assessment (LLM call to decide if worth processing)
     - Stage 2: Insight extraction (LLM call to analyze turns)
     - Stage 3: Knowledge base creation (LLM call to format insights)

  2. **Problems**:
     - Each stage abstracts further from raw data
     - "mailbox contains leaflet" → "interact with objects" → "exploration patterns"
     - 3x API costs and latency
     - Complex to debug and maintain
     - Generic advice instead of specific facts

  ### Proposed Solution
  Single-stage generation that:
  - Processes turn data directly into knowledge base format
  - Preserves specific world facts alongside strategic patterns
  - Uses simple heuristics for quality filtering
  - Reduces costs by 66% (1 LLM call vs 3)

  ## File Structure & Key Components

  ### Primary File to Modify
  - **File**: `/Volumes/workingfolder/ZorkGPT/zork_strategy_generator.py`
  - **Class**: `AdaptiveKnowledgeManager`

  ### Methods to Remove
  ```python
  # These methods implement the 3-stage process and will be removed:
  - _assess_knowledge_update_quality()    # Stage 1: Quality assessment
  - _analyze_full_insights()              # Stage 2: Full analysis
  - _analyze_selective_insights()         # Stage 2: Selective analysis
  - _analyze_escape_strategies()          # Stage 2: Escape analysis
  - _determine_update_strategy()          # Strategy selection
  - _merge_insights_with_existing()       # Stage 3: Merging
  - _create_new_knowledge_base()          # Stage 3: KB creation
  ```

  ### Methods to Add
  ```python
  # New simplified methods:
  - _should_update_knowledge()           # Heuristic quality check
  - _generate_knowledge_directly()       # Single-stage generation
  - _format_turn_data_for_prompt()      # Clean data formatting
  - _load_persistent_wisdom()           # Load cross-episode wisdom
  ```

  ### Methods to Modify
  ```python
  # Simplified main entry point:
  - update_knowledge_from_turns()        # Refactored to use single stage
  ```

  ### Methods to Keep Unchanged
  ```python
  # These remain as-is:
  - synthesize_inter_episode_wisdom()    # End-of-episode wisdom synthesis
  - _extract_turn_window_data()          # Turn data extraction
  - _intelligent_knowledge_merge()       # Simplified for merging
  - _preserve_map_section()              # Map preservation
  - _trim_map_section()                  # Map trimming
  ```

  ## Implementation Details

  ### 1. New Quality Check Method

  ```python
  def _should_update_knowledge(self, turn_data: Dict) -> Tuple[bool, str]:
      """
      Determine if turn data warrants a knowledge update using simple heuristics.

      Returns:
          Tuple[bool, str]: (should_update, reason)
      """
      actions = turn_data['actions_and_responses']

      # Always require minimum actions
      if len(actions) < 3:
          return False, "Too few actions (< 3)"

      # Always process death events (high learning value)
      if turn_data.get('death_events'):
          return True, f"Contains {len(turn_data['death_events'])} death event(s)"

      # Process if meaningful progress occurred
      if turn_data.get('score_changes'):
          return True, f"Score changed {len(turn_data['score_changes'])} times"

      if turn_data.get('location_changes'):
          return True, f"Discovered {len(turn_data['location_changes'])} new
     locations"

      # Check action variety (avoid pure repetition)
      unique_actions = set(a['action'] for a in actions)
      action_variety = len(unique_actions) / len(actions)

      if action_variety < 0.3:  # Less than 30% unique actions
          return False, f"Too repetitive ({action_variety:.1%} unique actions)"

      # Check response variety (ensure new information)
      unique_responses = set(a['response'][:50] for a in actions)

      if len(unique_responses) < 2:
          return False, "No new information in responses"

      # Check for meaningful content in responses
      total_response_length = sum(len(a['response']) for a in actions)
      if total_response_length < 100:
          return False, "Responses too short/uninformative"

      return True, f"Varied gameplay ({len(unique_actions)} unique actions)"
  ```

  ### 2. Turn Data Formatter

  ```python
  def _format_turn_data_for_prompt(self, turn_data: Dict) -> str:
      """Format turn data for LLM prompt with clear structure."""

      # Header information
      output = f"""EPISODE: {turn_data['episode_id']}
  TURNS: {turn_data['start_turn']}-{turn_data['end_turn']}
  TOTAL ACTIONS: {len(turn_data['actions_and_responses'])}
  """

      # Gameplay log with truncation for very long responses
      output += "\nGAMEPLAY LOG:\n"
 "\n"

      for action in turn_data['actions_and_responses']:
          response = action['response']
          # Truncate very long responses but preserve key information
          if len(response) > 300:
              response = response[:250] + "... [truncated]"

          output += f"Turn {action['turn']}: {action['action']}\n"
          output += f"Response: {response}\n"
          output += f"Reasoning: {action.get('reasoning', 'N/A')}\n"
          output += f"Critic Score: {action.get('critic_score', 'N/A')}\n\n"

      # Events section
      output += "\nEVENTS:\n"
 "\n"

      # Death events with full details
      if turn_data.get('death_events'):
          output += f"Deaths: {len(turn_data['death_events'])}\n"
          for death in turn_data['death_events']:
              output += f"  - Turn {death['turn']}: {death['reason']}\n"
              output += f"    Fatal action: {death.get('action_taken', 'Unknown')}\n"
              output += f"    Location: {death.get('death_location', 'Unknown')}\n"
              if death.get('death_messages'):
                  output += f"    Messages: {', '.join(death['death_messages'])}\n"
      else:
          output += "Deaths: None\n"

      # Score changes
      if turn_data.get('score_changes'):
          output += f"\nScore Changes: {len(turn_data['score_changes'])}\n"
          for change in turn_data['score_changes']:
              output += f"  - Turn {change['turn']}: {change['from_score']} →
     {change['
         + to_score']}\n"

      # Location changes
      if turn_data.get('location_changes'):
          output += f"\nLocation Changes: {len(turn_data['location_changes'])}\n"
          for change in turn_data['location_changes']:
              output += f"  - Turn {change['turn']}: {change['from_location']} →
     {chang
         + e['to_location']}\n"

      return output
  ```

  ### 3. Persistent Wisdom Loader

  ```python
  def _load_persistent_wisdom(self) -> str:
      """
      Load persistent wisdom from previous episodes.

      Returns:
          str: Formatted persistent wisdom or empty string if not available
      """
      try:
          from config import get_config
          config = get_config()
          persistent_wisdom_file = config.orchestrator.persistent_wisdom_file

          with open(persistent_wisdom_file, "r", encoding="utf-8") as f:
              wisdom = f.read().strip()

          if wisdom:
              return f"""\n**PERSISTENT WISDOM FROM PREVIOUS EPISODES:**
  {'-' * 50}
  {wisdom}
  {'-' * 50}\n"""

      except FileNotFoundError:
          # No persistent wisdom file yet - this is fine for early episodes
          if self.logger:
              self.logger.debug("No persistent wisdom file found (normal for early
     epis
         + odes)")
      except Exception as e:
          if self.logger:
              self.logger.warning(
                  f"Could not load persistent wisdom: {e}",
                  extra={"event_type": "knowledge_update"}
              )

      return ""
  ```

  ### 4. Single-Stage Knowledge Generator

  ```python
  def _generate_knowledge_directly(self, turn_data: Dict, existing_knowledge: str) ->
     s
         + tr:
      """
      Generate knowledge base content in a single LLM call.

      Args:
          turn_data: Extracted turn data
          existing_knowledge: Current knowledge base content (if any)

      Returns:
          str: Complete knowledge base content
      """
      # Format turn data
      formatted_data = self._format_turn_data_for_prompt(turn_data)

      # Load persistent wisdom for context
      persistent_wisdom = self._load_persistent_wisdom()

      # Construct comprehensive prompt
      prompt = f"""Analyze this Zork gameplay data and create/update the knowledge
     base
         + .

  {formatted_data}

  EXISTING KNOWLEDGE BASE:
  {'-' * 50}
  {existing_knowledge if existing_knowledge else "No existing knowledge - this is the
     f
         + irst update"}
  {'-' * 50}

  {persistent_wisdom}

  INSTRUCTIONS:
  Create a comprehensive knowledge base with ALL of the following sections. If a
     sectio
         + n has no new information, keep the existing content for that section.

  ## WORLD KNOWLEDGE
  List ALL specific facts discovered about the game world:
  - **Item Locations**: Exact items and where found (e.g., "mailbox at West of House
     co
         + ntains leaflet")
  - **Room Connections**: Specific navigation paths (e.g., "north from West of House →
         + North of House")
  - **Dangers**: Specific threats and their locations (e.g., "grue in darkness east of
         + North of House")
  - **Object Interactions**: What happens with objects (e.g., "leaflet can be read,
     con
         + tains game introduction")
  - **Puzzle Solutions**: Any puzzles solved and their solutions
  - **Environmental Details**: Properties of locations, special features

  ## STRATEGIC PATTERNS
  Identify patterns from this gameplay session:
  - **Successful Actions**: What specific actions led to progress?
  - **Failed Approaches**: What didn't work and why?
  - **Exploration Strategies**: Effective methods for discovering new areas
  - **Resource Management**: How to use items effectively
  - **Objective Recognition**: How to identify new goals from game responses

  ## DEATH & DANGER ANALYSIS
  {self._format_death_analysis_section(turn_data) if turn_data.get('death_events')
     else
         +  "No deaths occurred in this session."}

  ## COMMAND SYNTAX
  List exact commands that worked:
  - Movement: [specific successful movement commands]
  - Interaction: [specific object interaction commands]
  - Combat: [any combat-related commands]
  - Special: [any special or unusual commands]

  ## LESSONS LEARNED
  Specific insights from this session:
  - **New Discoveries**: What was learned for the first time?
  - **Confirmed Patterns**: What previous knowledge was validated?
  - **Updated Understanding**: What previous assumptions were corrected?
  - **Future Strategies**: What should be tried next based on these learnings?

  ## CROSS-EPISODE INSIGHTS
  How this session relates to persistent wisdom:
  - **Confirmations**: Which persistent patterns were observed again?
  - **Contradictions**: What differed from previous episodes?
  - **Extensions**: What new details extend existing knowledge?

  CRITICAL REQUIREMENTS:
  1. **Be Specific**: Include exact names, locations, and commands
  2. **Preserve Details**: Never generalize specific facts into vague advice
  3. **Additive Updates**: When updating, ADD new facts, don't remove existing ones
  4. **Fact-First**: Prioritize concrete discoveries over abstract strategies
  5. **Complete Sections**: Include all sections even if some have minimal updates

  Remember: The agent needs BOTH specific facts ("mailbox contains leaflet") AND
     strate
         + gic insights ("reading items provides information")."""

      # Add qwen3-30b-a3b optimization if needed
      prompt = r"\no_think " + prompt

      try:
          response = self.client.chat.completions.create(
              model=self.analysis_model,
              messages=[
                  {
                      "role": "system",
                      "content": "You are creating a knowledge base for an AI agent
     pla
         + ying Zork. Focus on preserving specific, actionable facts from the gameplay while
     also
         +  identifying strategic patterns. Never abstract specific discoveries into generic
     advi
         + ce."
                  },
                  {"role": "user", "content": prompt}
              ],
              temperature=self.analysis_sampling.temperature,
              top_p=self.analysis_sampling.top_p,
              top_k=self.analysis_sampling.top_k,
              min_p=self.analysis_sampling.min_p,
              max_tokens=self.analysis_sampling.max_tokens or 3000,
          )

          return response.content.strip()

      except Exception as e:
          if self.logger:
              self.logger.error(
                  f"Knowledge generation failed: {e}",
                  extra={"event_type": "knowledge_update", "error": str(e)}
              )
          # Return existing knowledge on failure
          return existing_knowledge

  def _format_death_analysis_section(self, turn_data: Dict) -> str:
      """Format death events for the knowledge base."""
      if not turn_data.get('death_events'):
          return "No deaths occurred in this session."

      output = f"**{len(turn_data['death_events'])} death(s) occurred:**\n\n"

      for death in turn_data['death_events']:
          output += f"**Death at Turn {death['turn']}**\n"
          output += f"- Cause: {death['reason']}\n"
          output += f"- Fatal Action: {death.get('action_taken', 'Unknown')}\n"
          output += f"- Location: {death.get('death_location', 'Unknown')}\n"
          output += f"- Final Score: {death.get('final_score', 'Unknown')}\n"

          if death.get('death_messages'):
              output += f"- Key Messages: {'; '.join(death['death_messages'])}\n"

          # Include contextual information
          if death.get('death_context'):
              output += f"- Context: {death['death_context']}\n"

          output += "\n"

      return output
  ```

  ### 5. Refactored Main Update Method

  ```python
  def update_knowledge_from_turns(
      self,
      episode_id: str,
      start_turn: int,
      end_turn: int,
      is_final_update: bool = False,
  ) -> bool:
      """
      Update knowledge base from a specific turn range using single-stage generation.

      Args:
          episode_id: Current episode ID
          start_turn: Starting turn number
          end_turn: Ending turn number
          is_final_update: If True, more lenient about quality (episode-end updates)

      Returns:
          bool: True if knowledge was updated, False if skipped
      """
      if self.logger:
          self.logger.info(
              f"Knowledge update requested for turns {start_turn}-{end_turn}",
              extra={
                  "event_type": "knowledge_update_start",
                  "episode_id": episode_id,
                  "turn_range": f"{start_turn}-{end_turn}",
                  "is_final": is_final_update
              }
          )

      # Step 1: Extract turn window data
      turn_data = self._extract_turn_window_data(episode_id, start_turn, end_turn)

      if not turn_data or not turn_data.get('actions_and_responses'):
          if self.logger:
              self.logger.warning(
                  "No turn data found for analysis",
                  extra={
                      "event_type": "knowledge_update_skipped",
                      "reason": "no_data"
                  }
              )
          return False

      # Step 2: Quality check using heuristics
      should_update, reason = self._should_update_knowledge(turn_data)

      # Override for final updates if there's a death or significant content
      if is_final_update and not should_update:
          if turn_data.get('death_events') or len(turn_data['actions_and_responses'])
     >
         + = 5:
              should_update = True
              reason = "Final update with significant content"

      if self.logger:
          self.logger.info(
              f"Knowledge update decision: {'proceed' if should_update else 'skip'} -
     {
         + reason}",
              extra={
                  "event_type": "knowledge_update_decision",
                  "should_update": should_update,
                  "reason": reason
              }
          )

      if not should_update:
          return False

      # Step 3: Load existing knowledge
      existing_knowledge = ""
      try:
          if os.path.exists(self.output_file):
              with open(self.output_file, 'r', encoding='utf-8') as f:
                  existing_knowledge = f.read()

              # Trim map section for LLM processing
              existing_knowledge = self._trim_map_section(existing_knowledge)

      except Exception as e:
          if self.logger:
              self.logger.warning(
                  f"Could not load existing knowledge: {e}",
                  extra={"event_type": "knowledge_update"}
              )

      # Step 4: Generate new knowledge in single pass
      if self.logger:
          self.logger.info(
              "Generating knowledge base update",
              extra={"event_type": "knowledge_generation_start"}
          )

      new_knowledge = self._generate_knowledge_directly(turn_data, existing_knowledge)

      if not new_knowledge or new_knowledge.startswith("SKIP:"):
          if self.logger:
              self.logger.warning(
                  f"Knowledge generation returned skip or empty:
     {new_knowledge[:100]}"
         + ,
                  extra={"event_type": "knowledge_update_skipped"}
              )
          return False

      # Step 5: Preserve map section if it exists
      if existing_knowledge and "## CURRENT WORLD MAP" in existing_knowledge:
          # Extract and preserve the map section
          original_with_map = ""
          try:
              with open(self.output_file, 'r', encoding='utf-8') as f:
                  original_with_map = f.read()
              new_knowledge = self._preserve_map_section(original_with_map,
     new_knowled
         + ge)
          except:
              pass  # Map preservation is non-critical

      # Step 6: Write updated knowledge to file
      try:
          with open(self.output_file, 'w', encoding='utf-8') as f:
              f.write(new_knowledge)

          if self.logger:
              self.logger.info(
                  "Knowledge base updated successfully",
                  extra={
                      "event_type": "knowledge_update_success",
                      "file": self.output_file,
                      "size": len(new_knowledge)
                  }
              )

          # Step 7: Log the prompt if in debug mode
          if hasattr(self, 'log_prompts') and self.log_prompts:
              self._log_prompt_to_file(
                  "knowledge_update",
                  turn_data,
                  new_knowledge
              )

          return True

      except Exception as e:
          if self.logger:
              self.logger.error(
                  f"Failed to write knowledge base: {e}",
                  extra={
                      "event_type": "knowledge_update_error",
                      "error": str(e)
                  }
              )
          return False
  ```

  ## Migration Strategy

  ### Phase 1: Parallel Implementation (Week 1)
  1. Add configuration flag to `AdaptiveKnowledgeManager.__init__()`:
     ```python
     def __init__(self, ..., use_single_stage: bool = False):
         self.use_single_stage = use_single_stage
     ```

  2. Implement new methods alongside existing ones

  3. Add conditional logic in `update_knowledge_from_turns()`:
     ```python
     if self.use_single_stage:
         return self._update_knowledge_single_stage(...)
     else:
         return self._update_knowledge_multi_stage(...)  # Current implementation
     ```

  ### Phase 2: Testing & Validation (Week 2)
  1. Create test script that runs both versions:
     ```python
     # Test script: test_single_stage_knowledge.py
     def compare_knowledge_generation():
         # Create identical test data
         test_episodes = load_test_episodes()

         # Run both versions
         multi_stage_kb = AdaptiveKnowledgeManager(use_single_stage=False)
         single_stage_kb = AdaptiveKnowledgeManager(use_single_stage=True)

         # Compare outputs
         for episode in test_episodes:
             multi_result = multi_stage_kb.update_knowledge_from_turns(...)
             single_result = single_stage_kb.update_knowledge_from_turns(...)

             # Analyze differences
             compare_results(multi_result, single_result)
     ```

  2. Metrics to compare:
     - Specific facts captured
     - Processing time
     - Token usage
     - API costs
     - Knowledge quality

  ### Phase 3: Gradual Rollout (Week 3)
  1. Enable for 10% of updates (random selection)
  2. Monitor logs for issues
  3. Increase to 50% if successful
  4. Full rollout if metrics are positive

  ### Phase 4: Cleanup (Week 4)
  1. Set `use_single_stage=True` as default
  2. Mark old methods as deprecated
  3. Remove old code in next major version

  ## Testing Strategy

  ### Unit Tests
  ```python
  # test_knowledge_generation.py
  class TestSingleStageKnowledge(unittest.TestCase):

      def test_should_update_knowledge_minimum_actions(self):
          """Test quality check with too few actions."""
          turn_data = {'actions_and_responses': [{'action': 'look'}]}
          should_update, reason = self.km._should_update_knowledge(turn_data)
          self.assertFalse(should_update)
          self.assertIn("few actions", reason)

      def test_should_update_knowledge_death_events(self):
          """Test that death events always trigger update."""
          turn_data = {
              'actions_and_responses': [{'action': 'north'}] * 3,
              'death_events': [{'turn': 10, 'reason': 'grue'}]
          }
          should_update, reason = self.km._should_update_knowledge(turn_data)
          self.assertTrue(should_update)
          self.assertIn("death", reason)

      def test_format_turn_data_truncation(self):
          """Test that long responses are truncated properly."""
          long_response = "A" * 500
          turn_data = {
              'actions_and_responses': [{
                  'turn': 1,
                  'action': 'look',
                  'response': long_response
              }]
          }
          formatted = self.km._format_turn_data_for_prompt(turn_data)
          self.assertIn("[truncated]", formatted)
          self.assertLess(len(formatted), 1000)
  ```

  ### Integration Tests
  ```python
  def test_knowledge_preserves_specific_facts():
      """Verify specific facts are preserved, not abstracted."""
      # Create test log with specific facts
      test_log = create_test_log_with_facts()

      # Generate knowledge
      km = AdaptiveKnowledgeManager(use_single_stage=True)
      km.update_knowledge_from_turns("test", 1, 10)

      # Read generated knowledge
      with open("knowledgebase.md", "r") as f:
          knowledge = f.read()

      # Verify specific facts are present
      assert "mailbox at West of House contains leaflet" in knowledge
      assert "north from West of House leads to North of House" in knowledge
      assert "grue in darkness east of North of House" in knowledge

      # Verify it's not just generic advice
      assert knowledge.count("specific") < knowledge.count("mailbox")
  ```

  ## Risk Mitigation

  ### Risk 1: Prompt Too Large
  **Mitigation**:
  - Implement smart truncation in `_format_turn_data_for_prompt()`
  - Limit to most recent 50 actions if over token limit
  - Prioritize death events and score changes

  ### Risk 2: Loss of Quality Control
  **Mitigation**:
  - Comprehensive heuristic checks in `_should_update_knowledge()`
  - Add "SKIP:" prefix handling in prompt
  - Monitor skipped updates in logs

  ### Risk 3: Existing Knowledge Corruption
  **Mitigation**:
  - Always backup existing file before update
  - Validate new knowledge isn't empty
  - Preserve map section separately

  ### Risk 4: Persistent Wisdom Integration
  **Mitigation**:
  - Load wisdom in separate try/except block
  - Continue without wisdom if unavailable
  - Test with and without wisdom file

  ## Configuration Changes

  Add to `config.yaml`:
  ```yaml
  adaptive_knowledge:
    use_single_stage: false  # Set to true to enable new system
    min_actions_for_update: 3
    max_actions_per_prompt: 50
    preserve_backups: true
  ```

  ## Monitoring & Metrics

  Log these metrics for comparison:
  ```python
  self.logger.info(
      "Knowledge update completed",
      extra={
          "event_type": "knowledge_update_metrics",
          "method": "single_stage" if self.use_single_stage else "multi_stage",
          "processing_time": end_time - start_time,
          "turns_processed": len(turn_data['actions_and_responses']),
          "knowledge_size": len(new_knowledge),
          "api_calls": 1 if self.use_single_stage else 3,
          "skipped": not should_update
      }
  )
  ```

  ## Success Criteria

  The refactor is successful if:
  1. **Cost Reduction**: 66% reduction in API calls
  2. **Quality Improvement**: More specific facts in knowledge base
  3. **Performance**: 3x faster knowledge updates
  4. **Reliability**: No increase in error rates
  5. **Specificity**: Concrete facts preserved (not abstracted)

  ## Implementation Checklist

  - [ ] Add configuration flag for single-stage mode
  - [ ] Implement `_should_update_knowledge()` method
  - [ ] Implement `_format_turn_data_for_prompt()` method
  - [ ] Implement `_load_persistent_wisdom()` method
  - [ ] Implement `_generate_knowledge_directly()` method
  - [ ] Refactor `update_knowledge_from_turns()` to use new flow
  - [ ] Add comprehensive unit tests
  - [ ] Add integration tests comparing both approaches
  - [ ] Test with various episode logs (deaths, loops, progress)
  - [ ] Test persistent wisdom integration
  - [ ] Run A/B comparison on real gameplay data
  - [ ] Update documentation
  - [ ] Add monitoring/metrics
  - [ ] Deploy with feature flag
  - [ ] Monitor production metrics
  - [ ] Gradual rollout
  - [ ] Remove old code after validation

  ## Notes for Implementation

  1. **Preserve ALL specific details** - The main goal is to stop losing specific
     facts through abstraction
  2. **Persistent wisdom is context, not primary content** - It should inform but not
     dominate
  3. **Quality filtering should be simple** - Complex LLM-based filtering defeats the
     purpose
  4. **One prompt to rule them all** - The single prompt must be comprehensive but
     clear
  5. **Test with edge cases** - Deaths, loops, first episode, 500+ turn episodes

  This refactor will significantly simplify the codebase while improving the quality
     of knowledge capture and reducing operational costs.
