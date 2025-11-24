"""Test script for command agent translations"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.command_agent import CommandAgent

def test_translations():
    """Test various translation cases"""
    print("Loading command agent...")
    agent = CommandAgent()
    print("Command agent loaded!\n")
    
    test_cases = [
        ("show me current directory", "pwd"),
        ("go back", "cd -"),
        ("list all files in the Dataset directory", "ls Dataset"),
        ("print all files in the Dataset directory", "ls Dataset"),
        ("go to Dataset directory", "cd Dataset"),
        ("list all files in the model directory", "ls model"),
        ("go to model directory", "cd model"),
        ("list files in Dataset directory", "ls Dataset"),
    ]
    
    print("=" * 70)
    print("Testing Command Translations")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = agent.translate(input_text)
        result_normalized = result.lower().strip()
        expected_normalized = expected.lower().strip()
        
        if result_normalized == expected_normalized:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"\n{status}")
        print(f"  Input:    {input_text}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        
        if status == "✗ FAIL":
            print(f"  → Mismatch!")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = test_translations()
    sys.exit(0 if success else 1)

