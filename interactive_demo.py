from context_manager import generate_with_context, fetch_context

def interactive_demo():
    """Interactive demo of context-aware generation."""
    
    print("🎭 Interactive Context-Aware AI Demo")
    print("=" * 50)
    print("This AI will remember your previous conversations!")
    print("Type 'quit' to exit, 'context' to see your conversation history\n")
    
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip() or "demo_user"
    print(f"👤 User ID: {user_id}\n")
    
    while True:
        prompt = input("🤖 You: ").strip()
        
        if prompt.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif prompt.lower() == 'context':
            print("\n📋 Your conversation history:")
            context = fetch_context(user_id, "")
            if context:
                print(context)
            else:
                print("No previous conversations found.")
            print()
            continue
        elif not prompt:
            continue
            
        print("🤔 AI is thinking (with context)...")
        response = generate_with_context(prompt, user_id)
        print(f"🤖 AI: {response}\n")

if __name__ == "__main__":
    interactive_demo() 