"""Complete version of code generation tasks for PCF benchmarking."""

from typing import Any, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.metrics.code import (
    extract_code_from_markdown,
    check_syntax,
    check_executability,
    check_code_completeness,
    check_imports,
    analyze_code_structure
)
from .base import BaseTask


class SimulationCodeGenerationTask(BaseTask):
    """Generate complete Meep simulation code with full validation."""
    
    def get_task_name(self) -> str:
        return "simulation_code_generation_complete"
    
    def get_task_description(self) -> str:
        return "Generate executable and complete Meep simulation code from PCF description"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "params" not in data:
            return None
        
        family = data["params"]["family"]
        params = data["params"]["params"]
        
        # Detailed prompt for complete code generation
        prompt = f"""Generate a complete, executable Python script using Meep to simulate this {family} PCF structure.

Parameters: {params}

The script must include:
1. All necessary imports (meep, numpy, matplotlib if needed)
2. Material definitions (silica, air, etc.)
3. Geometry setup with proper PCF structure
4. Simulation cell and boundary conditions
5. Light source configuration
6. Simulation run with appropriate runtime
7. Field extraction and result saving
8. Proper error handling

Provide ONLY the complete Python code in a ```python code block, ready to execute."""
        
        pred_code = self.model.query(prompt, image_path=data["struct_image"], max_tokens=2048)
        
        # Extract code from markdown
        pred_code = extract_code_from_markdown(pred_code)
        
        # 1. Syntax check
        syntax_correct, syntax_error = check_syntax(pred_code)
        
        # 2. Import check
        required_imports = ['meep', 'numpy']
        import_results = check_imports(pred_code, required_imports)
        
        # 3. Completeness check
        required_components = {
            'geometry': ['geometry', 'Cylinder', 'Block', 'GeometricObject'],
            'simulation': ['Simulation', 'sim', 'run'],
            'source': ['Source', 'GaussianSource', 'ContinuousSource'],
            'material': ['Medium', 'material', 'epsilon'],
            'output': ['output', 'get_', 'field', 'flux']
        }
        completeness_results = check_code_completeness(pred_code, required_components)
        
        # 4. Code structure analysis
        structure = analyze_code_structure(pred_code)
        
        # 5. Executability check (optional, can be slow)
        # executable, exec_error = check_executability(pred_code, timeout=10)
        executable = False  # Skip for speed, can enable if needed
        exec_error = "Skipped for performance"
        
        # Compute overall scores
        import_score = sum(import_results.values()) / len(import_results) if import_results else 0.0
        completeness_score = sum(completeness_results.values()) / len(completeness_results) if completeness_results else 0.0
        
        # Overall quality score
        quality_score = (
            0.3 * (1.0 if syntax_correct else 0.0) +
            0.3 * import_score +
            0.4 * completeness_score
        )
        
        return {
            "sample_id": sample_id,
            "family": family,
            # Syntax
            "syntax_correct": syntax_correct,
            "syntax_error": syntax_error if not syntax_correct else "",
            # Imports
            "has_meep": import_results.get('meep', False),
            "has_numpy": import_results.get('numpy', False),
            "import_score": import_score,
            # Completeness
            "has_geometry": completeness_results.get('geometry', False),
            "has_simulation": completeness_results.get('simulation', False),
            "has_source": completeness_results.get('source', False),
            "has_material": completeness_results.get('material', False),
            "has_output": completeness_results.get('output', False),
            "completeness_score": completeness_score,
            # Structure
            "num_functions": structure.get('num_functions', 0),
            "num_imports": structure.get('num_imports', 0),
            "num_lines": structure.get('num_lines', 0),
            # Execution
            "executable": executable,
            "exec_error": exec_error if not executable else "",
            # Overall
            "quality_score": quality_score,
            "code_length": len(pred_code)
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        
        # Syntax metrics
        syntax_rate = sum(r["syntax_correct"] for r in valid_results) / len(valid_results)
        
        # Import metrics
        meep_rate = sum(r["has_meep"] for r in valid_results) / len(valid_results)
        numpy_rate = sum(r["has_numpy"] for r in valid_results) / len(valid_results)
        avg_import_score = np.mean([r["import_score"] for r in valid_results])
        
        # Completeness metrics
        geometry_rate = sum(r["has_geometry"] for r in valid_results) / len(valid_results)
        simulation_rate = sum(r["has_simulation"] for r in valid_results) / len(valid_results)
        source_rate = sum(r["has_source"] for r in valid_results) / len(valid_results)
        material_rate = sum(r["has_material"] for r in valid_results) / len(valid_results)
        output_rate = sum(r["has_output"] for r in valid_results) / len(valid_results)
        avg_completeness = np.mean([r["completeness_score"] for r in valid_results])
        
        # Structure metrics
        avg_functions = np.mean([r["num_functions"] for r in valid_results])
        avg_imports = np.mean([r["num_imports"] for r in valid_results])
        avg_lines = np.mean([r["num_lines"] for r in valid_results])
        
        # Overall quality
        avg_quality = np.mean([r["quality_score"] for r in valid_results])
        
        return {
            # Syntax
            "syntax_correctness_rate": syntax_rate,
            # Imports
            "meep_import_rate": meep_rate,
            "numpy_import_rate": numpy_rate,
            "avg_import_score": avg_import_score,
            # Completeness
            "geometry_rate": geometry_rate,
            "simulation_rate": simulation_rate,
            "source_rate": source_rate,
            "material_rate": material_rate,
            "output_rate": output_rate,
            "avg_completeness_score": avg_completeness,
            # Structure
            "avg_num_functions": avg_functions,
            "avg_num_imports": avg_imports,
            "avg_num_lines": avg_lines,
            # Overall
            "avg_quality_score": avg_quality,
            "num_samples": len(valid_results)
        }


class ParameterModificationTask(BaseTask):
    """Generate code to modify PCF parameters based on design suggestions."""
    
    def get_task_name(self) -> str:
        return "parameter_modification_complete"
    
    def get_task_description(self) -> str:
        return "Generate complete code to modify PCF parameters based on expert suggestions"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level3" not in data["annotation"]:
            return None
        
        # Extract suggestion from level3
        level3 = data["annotation"]["level3"]
        suggestion = ""
        for conv in level3.get("conversations", []):
            value = conv.get("value", "")
            if "action" in value.lower():
                suggestion = value
                break
        
        if not suggestion:
            return None
        
        # Get current parameters
        current_params = data["params"]["params"]
        
        prompt = f"""Based on this expert design suggestion:
"{suggestion}"

Current parameters: {current_params}

Generate Python code that:
1. Defines the current parameters as a dictionary
2. Implements the suggested modifications
3. Validates the new parameters
4. Prints the changes

Provide complete, executable Python code in a ```python code block."""
        
        pred_code = self.model.query(prompt, image_path=data["struct_image"], max_tokens=1024)
        pred_code = extract_code_from_markdown(pred_code)
        
        # Check syntax
        syntax_correct, syntax_error = check_syntax(pred_code)
        
        # Check if parameters are mentioned
        has_param_dict = 'params' in pred_code or 'parameters' in pred_code
        has_assignment = '=' in pred_code
        has_validation = any(kw in pred_code.lower() for kw in ['if', 'assert', 'check', 'valid'])
        has_output = any(kw in pred_code for kw in ['print', 'return', 'output'])
        
        # Analyze structure
        structure = analyze_code_structure(pred_code)
        
        completeness = sum([has_param_dict, has_assignment, has_validation, has_output]) / 4.0
        
        return {
            "sample_id": sample_id,
            "syntax_correct": syntax_correct,
            "syntax_error": syntax_error if not syntax_correct else "",
            "has_param_dict": has_param_dict,
            "has_assignment": has_assignment,
            "has_validation": has_validation,
            "has_output": has_output,
            "completeness": completeness,
            "num_lines": structure.get('num_lines', 0),
            "code_length": len(pred_code)
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        
        syntax_rate = sum(r["syntax_correct"] for r in valid_results) / len(valid_results)
        param_dict_rate = sum(r["has_param_dict"] for r in valid_results) / len(valid_results)
        assignment_rate = sum(r["has_assignment"] for r in valid_results) / len(valid_results)
        validation_rate = sum(r["has_validation"] for r in valid_results) / len(valid_results)
        output_rate = sum(r["has_output"] for r in valid_results) / len(valid_results)
        avg_completeness = np.mean([r["completeness"] for r in valid_results])
        avg_lines = np.mean([r["num_lines"] for r in valid_results])
        
        return {
            "syntax_correctness_rate": syntax_rate,
            "param_dict_rate": param_dict_rate,
            "assignment_rate": assignment_rate,
            "validation_rate": validation_rate,
            "output_rate": output_rate,
            "avg_completeness": avg_completeness,
            "avg_num_lines": avg_lines,
            "num_samples": len(valid_results)
        }


class CodeRefactoringTask(BaseTask):
    """Refactor and optimize existing PCF simulation code."""
    
    def get_task_name(self) -> str:
        return "code_refactoring"
    
    def get_task_description(self) -> str:
        return "Refactor and optimize PCF simulation code for better performance"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level4" not in data["annotation"]:
            return None
        
        level4 = data["annotation"]["level4"]
        original_code = level4.get("code", "")
        
        if not original_code:
            return None
        
        prompt = f"""Refactor and optimize this PCF simulation code:

```python
{original_code}
```

Improvements to make:
1. Add proper error handling
2. Improve code structure and readability
3. Add comments and documentation
4. Optimize performance where possible
5. Follow Python best practices

Provide the refactored code in a ```python code block."""
        
        pred_code = self.model.query(prompt, image_path=data["struct_image"], max_tokens=2048)
        pred_code = extract_code_from_markdown(pred_code)
        
        # Compare structures
        orig_structure = analyze_code_structure(original_code)
        new_structure = analyze_code_structure(pred_code)
        
        # Check improvements
        syntax_correct, _ = check_syntax(pred_code)
        has_comments = '#' in pred_code
        has_docstrings = '"""' in pred_code or "'''" in pred_code
        has_error_handling = 'try' in pred_code or 'except' in pred_code
        
        # Check if code is more structured
        improved_structure = new_structure['num_functions'] >= orig_structure['num_functions']
        
        quality_score = sum([
            syntax_correct,
            has_comments,
            has_docstrings,
            has_error_handling,
            improved_structure
        ]) / 5.0
        
        return {
            "sample_id": sample_id,
            "syntax_correct": syntax_correct,
            "has_comments": has_comments,
            "has_docstrings": has_docstrings,
            "has_error_handling": has_error_handling,
            "improved_structure": improved_structure,
            "orig_lines": orig_structure['num_lines'],
            "new_lines": new_structure['num_lines'],
            "quality_score": quality_score
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        
        return {
            "syntax_correctness_rate": sum(r["syntax_correct"] for r in valid_results) / len(valid_results),
            "comment_rate": sum(r["has_comments"] for r in valid_results) / len(valid_results),
            "docstring_rate": sum(r["has_docstrings"] for r in valid_results) / len(valid_results),
            "error_handling_rate": sum(r["has_error_handling"] for r in valid_results) / len(valid_results),
            "structure_improvement_rate": sum(r["improved_structure"] for r in valid_results) / len(valid_results),
            "avg_quality_score": np.mean([r["quality_score"] for r in valid_results]),
            "num_samples": len(valid_results)
        }
