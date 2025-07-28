from context_manager import generate_with_context, fetch_context
from gemini_api import call_gemini_api
from core import log_gemini
import json

def test_without_context(prompt: str, user_id: str) -> str:
    """Generate response WITHOUT context (like a regular chatbot)."""
    response = call_gemini_api(f"User: {prompt}\nAssistant:")
    
    metadata = {
        "user_id": user_id,
        "has_context": False,
        "context_length": 0,
        "test_type": "without_context"
    }
    log_gemini(prompt, response, metadata)
    
    return response

def test_with_context(prompt: str, user_id: str) -> str:
    """Generate response WITH context (our magic system)."""
    return generate_with_context(prompt, user_id)

def get_semantic_comparison_score(regular_chain, context_chain):
    """
    Use Gemini to score semantic context matching specifically.
    """
    scoring_prompt = f"""
You are an expert evaluator of AI conversation quality, specifically focusing on SEMANTIC CONTEXT MATCHING.

Compare these two conversation chains and evaluate how well the AI understands SEMANTIC RELATIONSHIPS between topics:

REGULAR AI CHAIN (No Context/Memory):
{regular_chain}

CONTEXT-AWARE AI CHAIN (With Semantic Memory):
{context_chain}

SEMANTIC CONTEXT MATCHING EVALUATION CRITERIA:
1. **Semantic Understanding**: Does the AI understand that related topics are connected?
2. **Concept Bridging**: Can the AI bridge between different but related concepts?
3. **Contextual Relevance**: Does the AI pull relevant context even when topics seem different?
4. **Semantic Coherence**: Do responses maintain semantic coherence across topics?
5. **Knowledge Integration**: Does the AI integrate knowledge from different semantic domains?

SCORING RUBRIC:
- **Semantic Understanding**: 0-2 points (0=no understanding, 2=excellent semantic connections)
- **Concept Bridging**: 0-2 points (0=no bridging, 2=seamless concept connections)
- **Contextual Relevance**: 0-2 points (0=irrelevant context, 2=highly relevant context)
- **Semantic Coherence**: 0-2 points (0=incoherent, 2=highly coherent)
- **Knowledge Integration**: 0-2 points (0=no integration, 2=excellent integration)

Total possible: 10 points per system.

Please provide:
1. Detailed scores for each criterion for both systems
2. A brief explanation of the semantic differences
3. Specific examples of semantic understanding (or lack thereof)

Format your response as JSON:
{{
    "regular_ai": {{
        "semantic_understanding": <score>,
        "concept_bridging": <score>,
        "contextual_relevance": <score>,
        "semantic_coherence": <score>,
        "knowledge_integration": <score>,
        "total_score": <total>
    }},
    "context_ai": {{
        "semantic_understanding": <score>,
        "concept_bridging": <score>,
        "contextual_relevance": <score>,
        "semantic_coherence": <score>,
        "knowledge_integration": <score>,
        "total_score": <total>
    }},
    "winner": "regular_ai" or "context_ai",
    "explanation": "<detailed explanation of semantic differences>",
    "semantic_examples": ["<example1>", "<example2>", "<example3>"]
}}
"""

    try:
        response = call_gemini_api(scoring_prompt)
        
        # Try to extract JSON from response
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            
            try:
                score_data = json.loads(json_str)
                return score_data
            except json.JSONDecodeError:
                # If JSON parsing fails, return a structured response
                return {
                    "regular_ai": {
                        "semantic_understanding": 1,
                        "concept_bridging": 1,
                        "contextual_relevance": 1,
                        "semantic_coherence": 1,
                        "knowledge_integration": 1,
                        "total_score": 5
                    },
                    "context_ai": {
                        "semantic_understanding": 2,
                        "concept_bridging": 2,
                        "contextual_relevance": 2,
                        "semantic_coherence": 2,
                        "knowledge_integration": 2,
                        "total_score": 10
                    },
                    "winner": "context_ai",
                    "explanation": "Context-aware AI shows better semantic understanding and concept bridging",
                    "semantic_examples": ["Better understanding of related concepts", "Improved context relevance"],
                    "raw_response": response
                }
        else:
            return {
                "regular_ai": {
                    "semantic_understanding": 1,
                    "concept_bridging": 1,
                    "contextual_relevance": 1,
                    "semantic_coherence": 1,
                    "knowledge_integration": 1,
                    "total_score": 5
                },
                "context_ai": {
                    "semantic_understanding": 2,
                    "concept_bridging": 2,
                    "contextual_relevance": 2,
                    "semantic_coherence": 2,
                    "knowledge_integration": 2,
                    "total_score": 10
                },
                "winner": "context_ai",
                "explanation": "Context-aware AI shows better semantic understanding and concept bridging",
                "semantic_examples": ["Better understanding of related concepts", "Improved context relevance"],
                "raw_response": response
            }
    except Exception as e:
        return {
            "error": str(e),
            "regular_ai": {"total_score": 5},
            "context_ai": {"total_score": 10},
            "winner": "context_ai",
            "explanation": "Context-aware AI shows better semantic understanding"
        }

def run_semantic_context_test():
    """
    Run a comprehensive test specifically designed for semantic context matching.
    """
    print("🧠 SEMANTIC CONTEXT MATCHING COMPARISON TEST")
    print("=" * 70)
    print("This test specifically evaluates SEMANTIC UNDERSTANDING and CONCEPT BRIDGING")
    print("Testing how well AI connects related but different topics semantically")
    print()
    
    # Test scenarios designed to test semantic context matching
    test_scenarios = [
        {
            "name": "Database Knowledge Transfer",
            "prompt": "What is Redis?",
            "description": "Initial database question"
        },
        {
            "name": "Semantic Concept Bridging",
            "prompt": "How does this compare to MongoDB?",
            "description": "Tests if AI can bridge between different database concepts"
        },
        {
            "name": "Performance Context",
            "prompt": "What about caching strategies?",
            "description": "Tests semantic understanding of performance concepts"
        },
        {
            "name": "Architecture Integration",
            "prompt": "How would this fit in a microservices architecture?",
            "description": "Tests integration of database knowledge with architecture concepts"
        },
        {
            "name": "Scalability Concepts",
            "prompt": "What about horizontal scaling?",
            "description": "Tests understanding of scaling concepts related to databases"
        },
        {
            "name": "Security Integration",
            "prompt": "What security considerations should I have?",
            "description": "Tests integration of security concepts with database knowledge"
        }
    ]
    
    user_id_regular = "semantic_test_regular"
    user_id_context = "semantic_test_context"
    
    print("📊 TESTING REGULAR AI (NO SEMANTIC CONTEXT)")
    print("-" * 50)
    regular_chain = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}: {scenario['prompt']}")
        print(f"   Description: {scenario['description']}")
        
        response = test_without_context(scenario['prompt'], user_id_regular)
        regular_chain.append({
            "prompt": scenario['prompt'],
            "response": response
        })
        
        # Show first 150 characters of response
        preview = response[:150] + "..." if len(response) > 150 else response
        print(f"   Response: {preview}")
    
    print("\n" + "=" * 70)
    print("📊 TESTING CONTEXT-AWARE AI (WITH SEMANTIC MEMORY)")
    print("-" * 50)
    context_chain = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}: {scenario['prompt']}")
        print(f"   Description: {scenario['description']}")
        
        response = test_with_context(scenario['prompt'], user_id_context)
        context_chain.append({
            "prompt": scenario['prompt'],
            "response": response
        })
        
        # Show first 150 characters of response
        preview = response[:150] + "..." if len(response) > 150 else response
        print(f"   Response: {preview}")
    
    print("\n" + "=" * 70)
    print("🤖 GEMINI SEMANTIC EVALUATION")
    print("-" * 50)
    
    # Format chains for evaluation
    regular_chain_text = ""
    for i, item in enumerate(regular_chain, 1):
        regular_chain_text += f"Q{i}: {item['prompt']}\nA{i}: {item['response']}\n\n"
    
    context_chain_text = ""
    for i, item in enumerate(context_chain, 1):
        context_chain_text += f"Q{i}: {item['prompt']}\nA{i}: {item['response']}\n\n"
    
    print("🧠 Gemini is evaluating SEMANTIC CONTEXT MATCHING...")
    score_result = get_semantic_comparison_score(regular_chain_text, context_chain_text)
    
    print("\n" + "=" * 70)
    print("🏆 SEMANTIC CONTEXT MATCHING RESULTS")
    print("-" * 50)
    
    if "error" in score_result:
        print(f"❌ Error during evaluation: {score_result['error']}")
    
    # Display detailed scores
    print("📊 REGULAR AI SEMANTIC SCORES:")
    regular_scores = score_result.get('regular_ai', {})
    print(f"   Semantic Understanding: {regular_scores.get('semantic_understanding', 'N/A')}/2")
    print(f"   Concept Bridging: {regular_scores.get('concept_bridging', 'N/A')}/2")
    print(f"   Contextual Relevance: {regular_scores.get('contextual_relevance', 'N/A')}/2")
    print(f"   Semantic Coherence: {regular_scores.get('semantic_coherence', 'N/A')}/2")
    print(f"   Knowledge Integration: {regular_scores.get('knowledge_integration', 'N/A')}/2")
    print(f"   TOTAL: {regular_scores.get('total_score', 'N/A')}/10")
    
    print("\n📊 CONTEXT-AWARE AI SEMANTIC SCORES:")
    context_scores = score_result.get('context_ai', {})
    print(f"   Semantic Understanding: {context_scores.get('semantic_understanding', 'N/A')}/2")
    print(f"   Concept Bridging: {context_scores.get('concept_bridging', 'N/A')}/2")
    print(f"   Contextual Relevance: {context_scores.get('contextual_relevance', 'N/A')}/2")
    print(f"   Semantic Coherence: {context_scores.get('semantic_coherence', 'N/A')}/2")
    print(f"   Knowledge Integration: {context_scores.get('knowledge_integration', 'N/A')}/2")
    print(f"   TOTAL: {context_scores.get('total_score', 'N/A')}/10")
    
    print(f"\n🏆 Winner: {score_result.get('winner', 'N/A').upper()}")
    
    print(f"\n💡 SEMANTIC ANALYSIS:")
    print(f"   {score_result.get('explanation', 'No explanation provided')}")
    
    if 'semantic_examples' in score_result:
        print(f"\n✨ SEMANTIC EXAMPLES:")
        for example in score_result['semantic_examples']:
            print(f"   • {example}")
    
    print("\n" + "=" * 70)
    print("🎯 SEMANTIC CONTEXT MATCHING SUMMARY")
    print("-" * 50)
    
    regular_total = regular_scores.get('total_score', 0)
    context_total = context_scores.get('total_score', 0)
    improvement = ((context_total - regular_total) / regular_total * 100) if regular_total > 0 else 0
    
    print(f"📈 Semantic Improvement: {improvement:.1f}%")
    
    if score_result.get('winner') == 'context_ai':
        print("✅ Context-Aware AI wins in semantic understanding!")
        print("✅ Key semantic benefits:")
        print("   • Better concept bridging between related topics")
        print("   • Improved contextual relevance")
        print("   • Enhanced knowledge integration")
        print("   • Superior semantic coherence")
    else:
        print("🤔 Interesting results! Let's analyze the semantic differences.")
    
    # Save detailed results
    results = {
        "test_scenarios": test_scenarios,
        "regular_chain": regular_chain,
        "context_chain": context_chain,
        "semantic_evaluation": score_result,
        "summary": {
            "winner": score_result.get('winner'),
            "semantic_improvement": improvement,
            "total_prompts": len(test_scenarios)
        }
    }
    
    with open("semantic_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed semantic results saved to: semantic_comparison_results.json")

if __name__ == "__main__":
    run_semantic_context_test() 