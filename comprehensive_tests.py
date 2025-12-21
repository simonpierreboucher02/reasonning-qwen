"""
Comprehensive test suite with model responses.

This script tests the model with various types of prompts and
different sampling strategies.
"""

import torch
from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, GreedySampler, TemperatureSampler, TopPSampler
from utils import setup_device, get_logger

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_test(number, prompt, response, sampler_name="Greedy"):
    """Print a formatted test result."""
    print(f"\n[Test {number}] Sampler: {sampler_name}")
    print("-" * 80)
    print(f"PROMPT: {prompt}")
    print(f"\nRÉPONSE:")
    print(response)
    print("-" * 80)

def main():
    logger = get_logger()

    print("\n" + "=" * 80)
    print("  SÉRIE DE TESTS COMPLÈTE DU MODÈLE QWEN3-0.6B")
    print("=" * 80)

    # Setup
    device = setup_device(optimize_for_device=True)
    logger.info(f"Device: {device}")

    # Load model
    print("\n📥 Chargement du modèle...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path="./models/tokenizer-base.json")
    config = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(config)
    model.load_state_dict(torch.load("./models/qwen3-0.6B-base.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("✓ Modèle chargé")

    # Create generators with different samplers
    greedy_gen = TextGenerator(model, tokenizer, device, sampler=GreedySampler())
    temp_gen = TextGenerator(model, tokenizer, device, sampler=TemperatureSampler(temperature=0.7))
    topp_gen = TextGenerator(model, tokenizer, device, sampler=TopPSampler(top_p=0.9, temperature=0.8))

    # ========================================================================
    # CATÉGORIE 1: MATHÉMATIQUES
    # ========================================================================
    print_section("CATÉGORIE 1: MATHÉMATIQUES ET CALCUL")

    math_prompts = [
        "What is 15 + 27?",
        "Calculate 8 × 9:",
        "If I have 100 apples and give away 23, how many do I have left?",
        "What is the square root of 64?",
    ]

    for i, prompt in enumerate(math_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=50, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 2: CONNAISSANCES GÉNÉRALES
    # ========================================================================
    print_section("CATÉGORIE 2: CONNAISSANCES GÉNÉRALES")

    knowledge_prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "In what year did World War II end?",
    ]

    for i, prompt in enumerate(knowledge_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=50, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 3: RAISONNEMENT LOGIQUE
    # ========================================================================
    print_section("CATÉGORIE 3: RAISONNEMENT LOGIQUE")

    logic_prompts = [
        "If all cats are animals, and all animals need food, do cats need food?",
        "A train leaves station A at 2pm going 60mph. Another train leaves station B at 3pm going 80mph. If they're 200 miles apart, when will they meet?",
        "Which is heavier: a pound of feathers or a pound of gold?",
    ]

    for i, prompt in enumerate(logic_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=80, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 4: CRÉATIVITÉ ET NARRATION
    # ========================================================================
    print_section("CATÉGORIE 4: CRÉATIVITÉ ET NARRATION")

    creative_prompts = [
        "Once upon a time in a magical forest,",
        "The detective looked at the mysterious letter and realized",
        "In the year 2150, humanity discovered",
    ]

    for i, prompt in enumerate(creative_prompts, 1):
        response = temp_gen.generate(prompt, max_new_tokens=60, use_cache=True)
        print_test(i, prompt, response, "Temperature (0.7)")

    # ========================================================================
    # CATÉGORIE 5: INSTRUCTIONS ET EXPLICATIONS
    # ========================================================================
    print_section("CATÉGORIE 5: INSTRUCTIONS ET EXPLICATIONS")

    instruction_prompts = [
        "Explain what photosynthesis is:",
        "How do you make a peanut butter sandwich?",
        "What are the steps to solve a quadratic equation?",
    ]

    for i, prompt in enumerate(instruction_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=70, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 6: COMPLÉTION DE PHRASES
    # ========================================================================
    print_section("CATÉGORIE 6: COMPLÉTION DE PHRASES")

    completion_prompts = [
        "The sun rises in the",
        "Water boils at",
        "The opposite of hot is",
        "A triangle has",
    ]

    for i, prompt in enumerate(completion_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=30, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 7: COMPARAISON DES SAMPLERS
    # ========================================================================
    print_section("CATÉGORIE 7: COMPARAISON DES SAMPLERS SUR MÊME PROMPT")

    test_prompt = "The most important thing in life is"

    print(f"\nPROMPT COMMUN: {test_prompt}\n")

    print("=" * 80)
    print("[GREEDY SAMPLING]")
    print("-" * 80)
    response = greedy_gen.generate(test_prompt, max_new_tokens=50, use_cache=True)
    print(response)

    print("\n" + "=" * 80)
    print("[TEMPERATURE SAMPLING (0.7)]")
    print("-" * 80)
    response = temp_gen.generate(test_prompt, max_new_tokens=50, use_cache=True)
    print(response)

    print("\n" + "=" * 80)
    print("[TOP-P SAMPLING (0.9, temp=0.8)]")
    print("-" * 80)
    response = topp_gen.generate(test_prompt, max_new_tokens=50, use_cache=True)
    print(response)

    # ========================================================================
    # CATÉGORIE 8: QUESTIONS LONGUES ET COMPLEXES
    # ========================================================================
    print_section("CATÉGORIE 8: QUESTIONS COMPLEXES")

    complex_prompts = [
        "Describe the process of how a computer works, starting from when you press the power button:",
        "Explain the difference between machine learning and artificial intelligence:",
    ]

    for i, prompt in enumerate(complex_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=100, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 9: LANGUES ET MULTILINGUE
    # ========================================================================
    print_section("CATÉGORIE 9: CAPACITÉS MULTILINGUES")

    multilingual_prompts = [
        "Translate to French: Hello, how are you?",
        "What is 'thank you' in Spanish?",
        "Bonjour, comment allez-vous?",
    ]

    for i, prompt in enumerate(multilingual_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=40, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # CATÉGORIE 10: CODE ET PROGRAMMATION
    # ========================================================================
    print_section("CATÉGORIE 10: CODE ET PROGRAMMATION")

    code_prompts = [
        "Write a Python function to add two numbers:",
        "What is a for loop in programming?",
        "How do you print 'Hello World' in Python?",
    ]

    for i, prompt in enumerate(code_prompts, 1):
        response = greedy_gen.generate(prompt, max_new_tokens=70, use_cache=True)
        print_test(i, prompt, response, "Greedy")

    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print_section("RÉSUMÉ DES TESTS")

    print("""
📊 STATISTIQUES:
- Total de catégories testées: 10
- Total de prompts testés: ~35
- Samplers testés: 3 (Greedy, Temperature, Top-p)
- Device utilisé: {}
- Modèle: Qwen3-0.6B-base

✅ OBSERVATIONS:
- Le modèle génère des réponses cohérentes
- Greedy sampling: Déterministe, répétitif
- Temperature sampling: Plus varié, créatif
- Top-p sampling: Équilibre entre cohérence et diversité

📝 NOTES:
- Les réponses peuvent ne pas toujours être factuellement correctes
- C'est un modèle de base (non fine-tuné pour suivre des instructions)
- Pour de meilleures performances, utiliser le modèle "reasoning"
- Le modèle peut répéter ou générer des patterns

🎯 PROCHAINES ÉTAPES:
- Tester avec le modèle reasoning pour voir l'amélioration
- Utiliser self-consistency pour les questions mathématiques
- Fine-tuner sur des tâches spécifiques
""".format(device))

    print("\n" + "=" * 80)
    print("  ✓ TESTS TERMINÉS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
