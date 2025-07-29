#!/usr/bin/env python3
"""
🧠 Cortex CLI - Command Line Interface
CLI tool for interacting with the Cortex memory system.
"""

import argparse
import json
import sys
from typing import Dict, Any

from . import (
    store_conversation,
    get_conversation,
    semantic_embeddings,
    self_evolving_context,
    detect_semantic_drift,
    generate_with_context,
    generate_with_evolving_context
)

def store_cmd(args):
    """Store a conversation."""
    try:
        memory_id = store_conversation(
            user_id=args.user_id,
            prompt=args.prompt,
            response=args.response,
            metadata=json.loads(args.metadata) if args.metadata else None
        )
        print(f"✅ Conversation stored with memory ID: {memory_id}")
        return memory_id
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")
        return None

def retrieve_cmd(args):
    """Retrieve a conversation."""
    try:
        conversation = get_conversation(args.memory_id)
        if conversation:
            print(json.dumps(conversation, indent=2))
        else:
            print(f"❌ Memory not found: {args.memory_id}")
    except Exception as e:
        print(f"❌ Error retrieving conversation: {e}")

def search_cmd(args):
    """Search for similar context."""
    try:
        if args.method == "semantic":
            results = semantic_embeddings.find_semantically_similar_context(
                user_id=args.user_id,
                current_prompt=args.prompt,
                limit=args.limit,
                similarity_threshold=args.threshold
            )
        elif args.method == "evolving":
            results = self_evolving_context.find_evolving_context(
                user_id=args.user_id,
                current_prompt=args.prompt,
                limit=args.limit,
                similarity_threshold=args.threshold
            )
        else:
            print(f"❌ Invalid search method: {args.method}")
            return

        print(f"🔍 Found {len(results)} similar contexts:")
        for i, (context, score) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {score:.3f}")
            print(f"   Prompt: {context.get('prompt', 'N/A')}")
            print(f"   Response: {context.get('response', 'N/A')[:100]}...")
            print(f"   Memory ID: {context.get('embedding_id', 'N/A')}")

    except Exception as e:
        print(f"❌ Error searching: {e}")

def generate_cmd(args):
    """Generate response with context."""
    try:
        if args.method == "semantic":
            response = generate_with_context(
                user_id=args.user_id,
                prompt=args.prompt
            )
        elif args.method == "evolving":
            response = generate_with_evolving_context(
                user_id=args.user_id,
                prompt=args.prompt
            )
        else:
            print(f"❌ Invalid generation method: {args.method}")
            return

        print(f"🤖 Generated response:")
        print(response)

    except Exception as e:
        print(f"❌ Error generating response: {e}")

def analytics_cmd(args):
    """Get analytics for a user."""
    try:
        metrics = self_evolving_context.get_performance_metrics(args.user_id)
        print(f"📊 Analytics for user: {args.user_id}")
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"❌ Error getting analytics: {e}")

def drift_cmd(args):
    """Detect semantic drift."""
    try:
        drift_results = detect_semantic_drift(
            user_id=args.user_id,
            time_window_hours=args.hours
        )
        print(f"🔍 Drift analysis for user: {args.user_id}")
        print(json.dumps(drift_results, indent=2))
    except Exception as e:
        print(f"❌ Error detecting drift: {e}")

def prune_cmd(args):
    """Prune low-impact memories."""
    try:
        pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
            user_id=args.user_id,
            threshold=args.threshold
        )
        print(f"🧹 Pruning results for user: {args.user_id}")
        print(json.dumps(pruning_stats, indent=2))
    except Exception as e:
        print(f"❌ Error pruning memories: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🧠 Cortex CLI - Command Line Interface for Cortex Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store a conversation
  cortex store --user-id user123 --prompt "How do I implement auth?" --response "Use JWT tokens..."

  # Search for similar context
  cortex search --user-id user123 --prompt "Authentication best practices" --method semantic

  # Generate response with context
  cortex generate --user-id user123 --prompt "Secure my API" --method evolving

  # Get analytics
  cortex analytics --user-id user123

  # Detect drift
  cortex drift --user-id user123 --hours 24

  # Prune memories
  cortex prune --user-id user123 --threshold 0.3
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Store command
    store_parser = subparsers.add_parser('store', help='Store a conversation')
    store_parser.add_argument('--user-id', required=True, help='User ID')
    store_parser.add_argument('--prompt', required=True, help='User prompt')
    store_parser.add_argument('--response', required=True, help='AI response')
    store_parser.add_argument('--metadata', help='JSON metadata')
    store_parser.set_defaults(func=store_cmd)

    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve a conversation')
    retrieve_parser.add_argument('--memory-id', required=True, help='Memory ID')
    retrieve_parser.set_defaults(func=retrieve_cmd)

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar context')
    search_parser.add_argument('--user-id', required=True, help='User ID')
    search_parser.add_argument('--prompt', required=True, help='Search prompt')
    search_parser.add_argument('--method', choices=['semantic', 'evolving'], default='semantic', help='Search method')
    search_parser.add_argument('--limit', type=int, default=5, help='Number of results')
    search_parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold')
    search_parser.set_defaults(func=search_cmd)

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate response with context')
    generate_parser.add_argument('--user-id', required=True, help='User ID')
    generate_parser.add_argument('--prompt', required=True, help='User prompt')
    generate_parser.add_argument('--method', choices=['semantic', 'evolving'], default='semantic', help='Generation method')
    generate_parser.set_defaults(func=generate_cmd)

    # Analytics command
    analytics_parser = subparsers.add_parser('analytics', help='Get user analytics')
    analytics_parser.add_argument('--user-id', required=True, help='User ID')
    analytics_parser.set_defaults(func=analytics_cmd)

    # Drift command
    drift_parser = subparsers.add_parser('drift', help='Detect semantic drift')
    drift_parser.add_argument('--user-id', required=True, help='User ID')
    drift_parser.add_argument('--hours', type=int, default=24, help='Time window in hours')
    drift_parser.set_defaults(func=drift_cmd)

    # Prune command
    prune_parser = subparsers.add_parser('prune', help='Prune low-impact memories')
    prune_parser.add_argument('--user-id', required=True, help='User ID')
    prune_parser.add_argument('--threshold', type=float, default=0.3, help='Pruning threshold')
    prune_parser.set_defaults(func=prune_cmd)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()