#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime
from logger import parse_json_logs, render_logs_as_text, format_experiences_for_rl


def calculate_runtime(start_time, end_time):
    """Calculate runtime between two timestamps and return formatted string."""
    if not start_time or not end_time:
        return "Unknown"
    
    try:
        # Parse timestamps (assuming ISO format)
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Calculate duration
        duration = end_dt - start_dt
        total_seconds = int(duration.total_seconds())
        
        # Format as hours:minutes:seconds or minutes:seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="ZorkGPT Log Processing Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for the 'render' command
    render_parser = subparsers.add_parser("render", help="Render JSON logs as human-readable text")
    render_parser.add_argument("input_file", help="JSON log file to parse")
    render_parser.add_argument("-o", "--output", help="Output file (stdout if not specified)")
    render_parser.add_argument("-f", "--filter", help="Filter logs by event_type")
    
    # Parser for the 'rl' command
    rl_parser = subparsers.add_parser("rl", help="Format logs for reinforcement learning")
    rl_parser.add_argument("input_file", help="JSON log file to parse")
    rl_parser.add_argument("-o", "--output", help="Output file (stdout if not specified)")
    rl_parser.add_argument("-e", "--episode", type=int, help="Specific episode to extract (default: all)")
    
    # Parser for the 'stats' command
    stats_parser = subparsers.add_parser("stats", help="Show statistics from logs")
    stats_parser.add_argument("input_file", help="JSON log file to parse")
    
    # Parser for the 'episodes' command
    episodes_parser = subparsers.add_parser("episodes", help="Show episode details in markdown table format")
    episodes_parser.add_argument("input_file", help="JSON log file to parse")
    episodes_parser.add_argument("-o", "--output", help="Output file (stdout if not specified)")
    episodes_parser.add_argument("--limit", type=int, help="Limit the number of episodes shown (default: all)")
    episodes_parser.add_argument("--min-score", type=int, help="Only show episodes with score >= this value")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        logs = parse_json_logs(args.input_file)
        
        if args.command == "render":
            # Filter logs if requested
            if args.filter:
                logs = [log for log in logs if log.get("event_type") == args.filter]
            
            output = render_logs_as_text(logs)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print(output)
                
        elif args.command == "rl":
            # Extract experiences from logs
            experiences = []
            current_episode = 0
            
            for log in logs:
                if log.get("event_type") == "episode_start":
                    current_episode += 1
                    
                if args.episode and current_episode != args.episode:
                    continue
                    
                if log.get("event_type") == "experience":
                    exp = log.get("experience")
                    if exp:
                        experiences.append(exp)
            
            if not experiences:
                print("No experiences found in logs", file=sys.stderr)
                return
                
            formatted_data = format_experiences_for_rl(experiences)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(formatted_data, f, indent=2)
            else:
                print(json.dumps(formatted_data, indent=2))
                
        elif args.command == "stats":
            # Calculate and display statistics
            episode_count = 0
            total_turns = 0
            total_rewards = 0
            max_zork_score = 0
            
            for log in logs:
                event_type = log.get("event_type")
                
                if event_type == "episode_start":
                    episode_count += 1
                elif event_type == "episode_end":
                    total_turns += log.get("turn_count", 0)
                    total_rewards += log.get("total_reward", 0)
                    max_zork_score = max(max_zork_score, log.get("zork_score", 0))
            
            print(f"Statistics for {args.input_file}:")
            print(f"Total episodes: {episode_count}")
            print(f"Average turns per episode: {total_turns / episode_count if episode_count else 0:.2f}")
            print(f"Average reward per episode: {total_rewards / episode_count if episode_count else 0:.2f}")
            print(f"Maximum Zork score achieved: {max_zork_score}")
            
        elif args.command == "episodes":
            # Generate markdown table of episode details
            episodes_data = {}
            
            # Collect episode data
            for log in logs:
                event_type = log.get("event_type")
                episode_id = log.get("episode_id")
                
                if not episode_id:
                    continue
                    
                if episode_id not in episodes_data:
                    episodes_data[episode_id] = {
                        "episode_id": episode_id,
                        "agent_model": "Unknown",
                        "critic_model": "Unknown", 
                        "info_ext_model": "Unknown",
                        "start_time": None,
                        "end_time": None,
                        "runtime": "Unknown",
                        "turns": 0,
                        "score": 0,
                        "max_score": 0,
                        "total_reward": 0.0,
                        "status": "Unknown"
                    }
                
                episode = episodes_data[episode_id]
                
                if event_type == "episode_start":
                    episode["agent_model"] = log.get("agent_model", "Unknown")
                    episode["critic_model"] = log.get("critic_model", "Unknown")
                    episode["info_ext_model"] = log.get("info_ext_model", "Unknown")
                    episode["start_time"] = log.get("timestamp")
                elif event_type == "episode_end":
                    episode["turns"] = log.get("turn_count", 0)
                    episode["score"] = log.get("zork_score", 0)
                    episode["max_score"] = log.get("max_score", 0)
                    episode["total_reward"] = log.get("total_reward", 0.0)
                    episode["status"] = "Completed"
                    episode["end_time"] = log.get("timestamp")
                elif event_type == "game_over":
                    reason = log.get("reason", "")
                    if "died" in reason.lower():
                        episode["status"] = "Died"
                    elif "victory" in reason.lower():
                        episode["status"] = "Victory"
                    else:
                        episode["status"] = "Game Over"
            
            # Calculate runtime for each episode
            for episode in episodes_data.values():
                episode["runtime"] = calculate_runtime(episode["start_time"], episode["end_time"])
            
            # Convert to list and sort by score (descending)
            episodes_list = list(episodes_data.values())
            
            # Exclude episodes with 0 recorded steps/turns
            episodes_list = [ep for ep in episodes_list if ep["turns"] > 0]
            
            episodes_list.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply filters
            if args.min_score is not None:
                episodes_list = [ep for ep in episodes_list if ep["score"] >= args.min_score]
            
            if args.limit:
                episodes_list = episodes_list[:args.limit]
            
            # Generate markdown table
            if not episodes_list:
                output = "No episodes found matching the criteria.\n"
            else:
                output = "# Episode Performance Report\n\n"
                output += "| Episode ID | Score | Turns | Reward | Status | Runtime | Agent Model | Critic Model | Info Ext Model |\n"
                output += "|------------|-------|-------|--------|--------|---------|-------------|--------------|----------------|\n"
                
                for episode in episodes_list:
                    # Truncate model names for better table formatting
                    agent_model = episode["agent_model"][:40] + "..." if len(episode["agent_model"]) > 40 else episode["agent_model"]
                    critic_model = episode["critic_model"][:40] + "..." if len(episode["critic_model"]) > 40 else episode["critic_model"]
                    info_ext_model = episode["info_ext_model"][:40] + "..." if len(episode["info_ext_model"]) > 40 else episode["info_ext_model"]
                    
                    score_display = f"{episode['score']}/{episode['max_score']}" if episode['max_score'] > 0 else str(episode['score'])
                    
                    output += f"| `{episode['episode_id']}` | {score_display} | {episode['turns']} | {episode['total_reward']:.1f} | {episode['status']} | {episode['runtime']} | {agent_model} | {critic_model} | {info_ext_model} |\n"
                
                # Add summary statistics
                if len(episodes_list) > 1:
                    avg_score = sum(ep["score"] for ep in episodes_list) / len(episodes_list)
                    avg_turns = sum(ep["turns"] for ep in episodes_list) / len(episodes_list)
                    avg_reward = sum(ep["total_reward"] for ep in episodes_list) / len(episodes_list)
                    max_score = max(ep["score"] for ep in episodes_list)
                    
                    output += f"\n## Summary\n"
                    output += f"- **Episodes shown:** {len(episodes_list)}\n"
                    output += f"- **Average score:** {avg_score:.1f}\n"
                    output += f"- **Average turns:** {avg_turns:.1f}\n"
                    output += f"- **Average reward:** {avg_reward:.1f}\n"
                    output += f"- **Best score:** {max_score}\n"
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print(output)
    
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Error: File {args.input_file} contains invalid JSON", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main() 