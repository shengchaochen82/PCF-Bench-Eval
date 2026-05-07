"""Code evaluation metrics including advanced metrics (Pass@k, CodeBLEU, etc.).

Based on industry best practices:
- HumanEval/MBPP style pass@k metrics
- CodeBLEU for code similarity
- Execution-based evaluation
- AST-based structural comparison
"""

import ast
import re
import subprocess
import tempfile
import os
import math
import difflib
import time
from typing import Dict, Tuple, List, Any, Optional


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks."""
    # Try to find ```python code block
    pattern = r'```python\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find ``` code block
    pattern = r'```\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Return as is if no code block found
    return text


def check_syntax(code: str) -> Tuple[bool, str]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_executability(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Check if code can be executed without errors."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Try to execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        success = (result.returncode == 0)
        message = result.stderr if not success else ""
        
        return success, message
    
    except subprocess.TimeoutExpired:
        return False, f"Execution timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def check_code_completeness(code: str, required_components: Dict[str, list]) -> Dict[str, bool]:
    """Check if code contains required components."""
    code_lower = code.lower()
    
    results = {}
    for component_name, keywords in required_components.items():
        # Check if any of the keywords are present
        has_component = any(kw.lower() in code_lower for kw in keywords)
        results[component_name] = has_component
    
    return results


def check_imports(code: str, required_imports: list) -> Dict[str, bool]:
    """Check if code has required imports."""
    results = {}
    for imp in required_imports:
        # Check various import patterns
        patterns = [
            f'import {imp}',
            f'from {imp}',
            f'import {imp} as',
        ]
        has_import = any(pattern in code for pattern in patterns)
        results[imp] = has_import
    
    return results


def analyze_code_structure(code: str) -> Dict[str, int]:
    """Analyze code structure (functions, classes, etc.)."""
    try:
        tree = ast.parse(code)
        
        num_functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        num_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        num_imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        num_assignments = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assign))
        num_loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        num_conditionals = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
        
        # Cyclomatic complexity approximation
        complexity = 1 + num_loops + num_conditionals
        
        return {
            "num_functions": num_functions,
            "num_classes": num_classes,
            "num_imports": num_imports,
            "num_assignments": num_assignments,
            "num_lines": len(code.split('\n')),
            "num_loops": num_loops,
            "num_conditionals": num_conditionals,
            "cyclomatic_complexity": complexity
        }
    except:
        return {
            "num_functions": 0,
            "num_classes": 0,
            "num_imports": 0,
            "num_assignments": 0,
            "num_lines": len(code.split('\n')),
            "num_loops": 0,
            "num_conditionals": 0,
            "cyclomatic_complexity": 0
        }


# ============================================================================
# Advanced Code Evaluation Metrics
# ============================================================================

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k in pass@k
        
    Returns:
        pass@k score
        
    Reference: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def extract_identifiers(code: str) -> set:
    """Extract all identifiers (variable/function names) from code."""
    try:
        tree = ast.parse(code)
        identifiers = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)
                
        return identifiers
    except:
        return set()


def compute_ast_similarity(code1: str, code2: str) -> float:
    """
    Compute AST structural similarity between two code snippets.
    
    Args:
        code1: First code snippet
        code2: Second code snippet
        
    Returns:
        Similarity score [0, 1]
    """
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        # Extract node types
        nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
        nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
        
        # Compute Jaccard similarity
        set1 = set(nodes1)
        set2 = set(nodes2)
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
        
    except SyntaxError:
        return 0.0


def compute_code_bleu(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute CodeBLEU score (simplified version).
    
    CodeBLEU = α * BLEU + β * BLEU_weight + γ * Match_ast + δ * Match_df
    
    Args:
        reference: Ground truth code
        hypothesis: Generated code
        
    Returns:
        Dictionary with CodeBLEU components
    """
    # 1. Token-level BLEU (unigram and bigram) with clipped counts
    from collections import Counter
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Unigram precision (clipped)
    hyp_uni_counts = Counter(hyp_tokens)
    ref_uni_counts = Counter(ref_tokens)
    uni_clipped = sum(min(hyp_uni_counts[t], ref_uni_counts[t]) for t in hyp_uni_counts)
    unigram_precision = uni_clipped / sum(hyp_uni_counts.values()) if hyp_uni_counts else 0.0

    # Bigram precision (clipped)
    hyp_bigrams_list = list(zip(hyp_tokens[:-1], hyp_tokens[1:]))
    ref_bigrams_list = list(zip(ref_tokens[:-1], ref_tokens[1:]))
    hyp_bi_counts = Counter(hyp_bigrams_list)
    ref_bi_counts = Counter(ref_bigrams_list)
    bi_clipped = sum(min(hyp_bi_counts[b], ref_bi_counts[b]) for b in hyp_bi_counts)
    bigram_precision = bi_clipped / sum(hyp_bi_counts.values()) if hyp_bi_counts else 0.0

    # Geometric mean for BLEU component
    bleu_score = math.sqrt(unigram_precision * bigram_precision) if unigram_precision > 0 and bigram_precision > 0 else 0.0
    
    # 2. AST match
    ast_match = compute_ast_similarity(reference, hypothesis)
    
    # 3. Identifier overlap (simplified dataflow)
    ref_identifiers = extract_identifiers(reference)
    hyp_identifiers = extract_identifiers(hypothesis)
    identifier_overlap = len(ref_identifiers & hyp_identifiers) / len(ref_identifiers | hyp_identifiers) if (ref_identifiers | hyp_identifiers) else 0.0
    
    # Weighted combination (standard weights)
    code_bleu = 0.25 * bleu_score + 0.25 * unigram_precision + 0.25 * ast_match + 0.25 * identifier_overlap
    
    return {
        "code_bleu": code_bleu,
        "bleu_score": bleu_score,
        "ast_match": ast_match,
        "identifier_overlap": identifier_overlap,
        "unigram_precision": unigram_precision,
        "bigram_precision": bigram_precision
    }


def compute_edit_distance(code1: str, code2: str, normalized: bool = True) -> float:
    """
    Compute normalized edit distance between two code snippets.
    
    Args:
        code1: First code snippet
        code2: Second code snippet
        normalized: If True, normalize by max length
        
    Returns:
        Edit distance (normalized if specified, as similarity 0-1)
    """
    lines1 = code1.split('\n')
    lines2 = code2.split('\n')
    
    # Use difflib for sequence matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    # Get similarity ratio directly
    if normalized:
        return matcher.ratio()
    else:
        # Compute edit operations count
        opcodes = matcher.get_opcodes()
        edit_distance = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in opcodes if tag != 'equal')
        return float(edit_distance)


def execute_code_with_timeout(
    code: str,
    timeout: int = 5,
    capture_output: bool = True
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute code with timeout and detailed execution info.
    
    Args:
        code: Code to execute
        timeout: Execution timeout in seconds
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        (success, error_message, execution_info)
    """
    execution_info = {
        "executed": False,
        "execution_time": 0.0,
        "return_code": None,
        "stdout": "",
        "stderr": ""
    }
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute code
            result = subprocess.run(
                ['python', temp_file],
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            execution_info["execution_time"] = execution_time
            execution_info["executed"] = True
            execution_info["return_code"] = result.returncode
            
            if capture_output:
                execution_info["stdout"] = result.stdout[:1000] if result.stdout else ""
                execution_info["stderr"] = result.stderr[:1000] if result.stderr else ""
            
            if result.returncode != 0:
                return False, result.stderr[:500] if result.stderr else "Non-zero exit code", execution_info
            
            return True, "", execution_info
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except subprocess.TimeoutExpired:
        return False, f"Execution timeout after {timeout}s", execution_info
    except Exception as e:
        return False, str(e), execution_info


def compute_functional_correctness(
    generated_codes: List[str],
    reference_code: Optional[str] = None,
    timeout: int = 5,
    k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Compute functional correctness metrics including pass@k.
    
    Args:
        generated_codes: List of generated code samples
        reference_code: Optional ground truth reference code
        timeout: Execution timeout per sample
        k_values: k values for pass@k computation (default: [1, 5, 10])
        
    Returns:
        Dictionary with functional correctness metrics
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    n = len(generated_codes)
    correct_count = 0
    execution_results = []
    
    for code in generated_codes:
        success, error, exec_info = execute_code_with_timeout(code, timeout)
        execution_results.append({
            "success": success,
            "error": error,
            "exec_info": exec_info
        })
        if success:
            correct_count += 1
    
    # Compute pass@k for different k values
    pass_at_k_scores = {}
    for k in k_values:
        if k <= n:
            pass_at_k_scores[f"pass@{k}"] = compute_pass_at_k(n, correct_count, k)
    
    result = {
        "execution_success_rate": correct_count / n if n > 0 else 0.0,
        "num_correct": correct_count,
        "num_total": n,
        **pass_at_k_scores,
    }
    
    # If reference code provided, compute similarity metrics
    if reference_code:
        similarities = []
        for code in generated_codes:
            bleu_result = compute_code_bleu(reference_code, code)
            ast_sim = compute_ast_similarity(reference_code, code)
            edit_sim = compute_edit_distance(reference_code, code, normalized=True)
            similarities.append({
                "code_bleu": bleu_result["code_bleu"],
                "ast_similarity": ast_sim,
                "edit_similarity": edit_sim
            })
        
        result["avg_code_bleu"] = sum(s["code_bleu"] for s in similarities) / len(similarities)
        result["avg_ast_similarity"] = sum(s["ast_similarity"] for s in similarities) / len(similarities)
        result["avg_edit_similarity"] = sum(s["edit_similarity"] for s in similarities) / len(similarities)
    
    return result


def evaluate_code_comprehensive(
    generated_code: str,
    reference_code: Optional[str] = None,
    required_imports: Optional[List[str]] = None,
    required_components: Optional[Dict[str, List[str]]] = None,
    execute: bool = False,
    timeout: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive code evaluation combining all metrics.
    
    Args:
        generated_code: Generated code to evaluate
        reference_code: Optional ground truth code
        required_imports: Required import modules
        required_components: Required code components
        execute: Whether to execute the code
        timeout: Execution timeout
        
    Returns:
        Comprehensive evaluation results
    """
    results = {}
    
    # 1. Syntax check
    syntax_ok, syntax_error = check_syntax(generated_code)
    results["syntax_correct"] = syntax_ok
    results["syntax_error"] = syntax_error
    
    # 2. Import check
    if required_imports:
        import_results = check_imports(generated_code, required_imports)
        results["imports"] = import_results
        results["import_score"] = sum(import_results.values()) / len(import_results)
    
    # 3. Completeness check
    if required_components:
        completeness_results = check_code_completeness(generated_code, required_components)
        results["completeness"] = completeness_results
        results["completeness_score"] = sum(completeness_results.values()) / len(completeness_results)
    
    # 4. Structure analysis
    structure = analyze_code_structure(generated_code)
    results["structure"] = structure
    
    # 5. Comparison with reference (if provided)
    if reference_code and syntax_ok:
        bleu_result = compute_code_bleu(reference_code, generated_code)
        results["code_bleu"] = bleu_result["code_bleu"]
        results["ast_similarity"] = compute_ast_similarity(reference_code, generated_code)
        results["edit_similarity"] = compute_edit_distance(reference_code, generated_code)
        results["bleu_details"] = bleu_result
    
    # 6. Execution (if enabled)
    if execute and syntax_ok:
        exec_ok, exec_error, exec_info = execute_code_with_timeout(generated_code, timeout)
        results["executable"] = exec_ok
        results["exec_error"] = exec_error if not exec_ok else ""
        results["exec_info"] = exec_info
    else:
        results["executable"] = False
        results["exec_error"] = "Execution disabled" if not execute else "Syntax error"
        results["exec_info"] = {}
    
    # 7. Compute overall quality score
    quality_components = []
    if syntax_ok:
        quality_components.append(1.0)
    else:
        quality_components.append(0.0)
    
    if "import_score" in results:
        quality_components.append(results["import_score"])
    if "completeness_score" in results:
        quality_components.append(results["completeness_score"])
    if "code_bleu" in results:
        quality_components.append(results["code_bleu"])
    
    results["quality_score"] = sum(quality_components) / len(quality_components) if quality_components else 0.0
    
    return results
