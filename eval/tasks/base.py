"""Base task class for PCF benchmarking."""

from abc import ABC, abstractmethod
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

console = Console()
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
import base64


class BaseTask(ABC):
    """Base class for all evaluation tasks."""
    
    def __init__(self, data_dir: str, model_interface: Any):
        """
        Initialize task.
        
        Args:
            data_dir: Path to PCF samples directory
            model_interface: Model interface for inference
        """
        self.data_dir = data_dir
        self.model = model_interface
        self.results: List[Dict[str, Any]] = []
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Return task name."""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Return task description."""
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        """
        Evaluate a single sample.
        
        Args:
            sample_dir: Path to sample directory
            sample_id: Sample ID
            
        Returns:
            Evaluation result dictionary
        """
        pass
    
    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics from all results.
        
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def load_sample_data(self, sample_dir: str) -> Dict[str, Any]:
        """Load all data for a sample."""
        from tools.io_utils import read_json
        
        data = {}
        
        # Load params
        params_path = os.path.join(sample_dir, "params.json")
        if os.path.exists(params_path):
            data["params"] = read_json(params_path)
        
        # Load sim results
        sim_path = os.path.join(sample_dir, "sim.json")
        if os.path.exists(sim_path):
            data["sim"] = read_json(sim_path)
        
        # Load annotations
        ann_path = os.path.join(sample_dir, "annotation.json")
        if os.path.exists(ann_path):
            data["annotation"] = read_json(ann_path)
        
        # Get image paths
        images_dir = os.path.join(sample_dir, "images")
        data["struct_image"] = os.path.join(images_dir, "struct.png")
        data["field_1310_image"] = os.path.join(images_dir, "field_1310.png")
        data["field_1550_image"] = os.path.join(images_dir, "field_1550.png")
        
        # Legacy compatibility: use 1550nm as default field image
        data["field_image"] = data["field_1550_image"]
        
        # Verify image existence
        data["available_images"] = {
            "struct": os.path.exists(data["struct_image"]),
            "field_1310": os.path.exists(data["field_1310_image"]),
            "field_1550": os.path.exists(data["field_1550_image"]),
        }
        
        return data
    
    @staticmethod
    def concatenate_images(
        image_paths: List[str],
        direction: str = "horizontal",
        labels: Optional[List[str]] = None,
        spacing: int = 10,
        label_height: int = 30
    ) -> str:
        """
        Concatenate multiple images into a single image.
        
        Following best practices from multi-image VLM benchmarks:
        - Horizontal concatenation for comparing similar views
        - Add labels to distinguish images
        - Return base64 encoded image or temp file path
        
        Args:
            image_paths: List of paths to images
            direction: 'horizontal' or 'vertical'
            labels: Optional labels for each image (e.g., ['Structure', 'Field@1310nm', 'Field@1550nm'])
            spacing: Pixels between images
            label_height: Height for label text area
            
        Returns:
            Path to concatenated image (temporary file)
        """
        # Load images
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                images.append(img)
            else:
                # Create placeholder for missing images
                placeholder = Image.new('RGB', (256, 256), color=(200, 200, 200))
                images.append(placeholder)
        
        if not images:
            raise ValueError("No valid images to concatenate")
        
        # Calculate dimensions
        if direction == "horizontal":
            total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
            max_height = max(img.height for img in images)
            if labels:
                max_height += label_height
            
            # Create canvas
            canvas = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
            
            # Paste images
            x_offset = 0
            for i, img in enumerate(images):
                y_offset = label_height if labels else 0
                canvas.paste(img, (x_offset, y_offset))
                
                # Add label if provided
                if labels and i < len(labels):
                    try:
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(canvas)
                        # Use default font
                        font = ImageFont.load_default()
                        text_bbox = draw.textbbox((0, 0), labels[i], font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_x = x_offset + (img.width - text_width) // 2
                        draw.text((text_x, 5), labels[i], fill=(0, 0, 0), font=font)
                    except Exception:
                        pass  # Skip labels if font issues
                
                x_offset += img.width + spacing
        else:  # vertical
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
            if labels:
                total_height += label_height * len(images)
            
            canvas = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
            
            y_offset = 0
            for i, img in enumerate(images):
                if labels and i < len(labels):
                    try:
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(canvas)
                        font = ImageFont.load_default()
                        draw.text((5, y_offset + 5), labels[i], fill=(0, 0, 0), font=font)
                    except Exception:
                        pass
                    y_offset += label_height
                
                x_offset = (max_width - img.width) // 2
                canvas.paste(img, (x_offset, y_offset))
                y_offset += img.height + spacing
        
        # Save to temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        canvas.save(temp_file.name)
        return temp_file.name
    
    def get_multi_image_prompt(
        self,
        base_prompt: str,
        image_types: List[str]
    ) -> str:
        """
        Format prompt with image type markers.
        
        Best practice: Use explicit markers to indicate which parts of a 
        concatenated image correspond to what.
        
        Args:
            base_prompt: The base instruction prompt
            image_types: List of image types in order (e.g., ['struct', 'field_1310', 'field_1550'])
            
        Returns:
            Formatted prompt with image markers
        """
        type_labels = {
            'struct': '<img:structure>',
            'field_1310': '<img:field@1310nm>',
            'field_1550': '<img:field@1550nm>',
        }
        
        markers = [type_labels.get(t, f'<img:{t}>') for t in image_types]
        image_desc = "The image shows (from left to right): " + ", ".join(markers)
        
        return f"{image_desc}\n\n{base_prompt}"
    
    def prepare_task_images(
        self,
        data: Dict[str, Any],
        image_config: str = "struct_only"
    ) -> tuple[str, str]:
        """
        Prepare images for a task based on configuration.
        
        Args:
            data: Sample data from load_sample_data()
            image_config: One of:
                - 'struct_only': Only structure image
                - 'field_1310_only': Only field at 1310nm
                - 'field_1550_only': Only field at 1550nm  
                - 'struct_field_1550': Structure + Field 1550nm
                - 'struct_field_1310': Structure + Field 1310nm
                - 'both_fields': Both field images (1310nm + 1550nm)
                - 'all_three': Structure + both field images
                
        Returns:
            (image_path, prompt_prefix) tuple
        """
        configs = {
            'struct_only': (
                [data['struct_image']],
                ['Structure'],
                ['struct']
            ),
            'field_1310_only': (
                [data['field_1310_image']],
                ['Field@1310nm'],
                ['field_1310']
            ),
            'field_1550_only': (
                [data['field_1550_image']],
                ['Field@1550nm'],
                ['field_1550']
            ),
            'struct_field_1550': (
                [data['struct_image'], data['field_1550_image']],
                ['Structure', 'Field@1550nm'],
                ['struct', 'field_1550']
            ),
            'struct_field_1310': (
                [data['struct_image'], data['field_1310_image']],
                ['Structure', 'Field@1310nm'],
                ['struct', 'field_1310']
            ),
            'both_fields': (
                [data['field_1310_image'], data['field_1550_image']],
                ['Field@1310nm', 'Field@1550nm'],
                ['field_1310', 'field_1550']
            ),
            'all_three': (
                [data['struct_image'], data['field_1310_image'], data['field_1550_image']],
                ['Structure', 'Field@1310nm', 'Field@1550nm'],
                ['struct', 'field_1310', 'field_1550']
            ),
        }
        
        if image_config not in configs:
            raise ValueError(f"Unknown image_config: {image_config}")
        
        paths, labels, types = configs[image_config]
        
        # Single image: return directly
        if len(paths) == 1:
            return paths[0], ""
        
        # Multiple images: concatenate and add prompt prefix
        concat_path = self.concatenate_images(paths, labels=labels)
        prompt_prefix = self.get_multi_image_prompt("", types)
        
        return concat_path, prompt_prefix
    
    def run(self, max_samples: Optional[int] = None, num_workers: int = 1,
            max_retry_rounds: int = 3) -> Dict[str, Any]:
        """
        Run evaluation on all samples with automatic retry for error-skipped samples.

        Args:
            max_samples: Maximum number of samples to evaluate.
            num_workers: Number of parallel workers for sample evaluation.
                         Use 1 for sequential (default), >1 for concurrent API calls.
            max_retry_rounds: Additional passes to retry samples that errored
                              out in earlier rounds (default: 3). Samples that
                              legitimately return None (missing data) are NOT
                              retried; only transient API/network failures are.

        Returns:
            Dictionary containing results and metrics.
        """
        self.results = []
        self._errored_ids: List[str] = []  # track errors per round

        # Sample IDs: prefer the runner-supplied list (parquet-resolved),
        # otherwise fall back to a flat scan of the materialized cache dir.
        sample_dirs: Optional[List[str]] = getattr(self, "sample_ids", None)
        if sample_dirs is None:
            sample_dirs = sorted([
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
                and d.startswith("PCF_")
            ])

        if max_samples:
            sample_dirs = sample_dirs[:max_samples]

        # Display task header
        task_name = self.get_task_name()
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Task: {task_name}[/bold cyan]")
        console.print(f"[dim]Description: {self.get_task_description()}[/dim]")
        console.print(f"[yellow]Samples: {len(sample_dirs)} | Workers: {num_workers}[/yellow]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        # Initial pass
        remaining = list(sample_dirs)
        round_idx = 0
        while remaining:
            if round_idx > 0:
                console.print(
                    f"[yellow]↻ Retry round {round_idx}/{max_retry_rounds}:"
                    f" re-evaluating {len(remaining)} errored sample(s)...[/yellow]"
                )
            self._errored_ids = []
            if num_workers > 1:
                self._run_parallel(remaining, num_workers)
            else:
                self._run_sequential(remaining)

            # Decide what to retry in the next round: only samples that threw
            # an exception (tracked in self._errored_ids). Samples that
            # returned None from evaluate_sample() are legitimate skips.
            remaining = list(self._errored_ids)
            round_idx += 1
            if not remaining or round_idx > max_retry_rounds:
                break

        if remaining:
            console.print(
                f"[red]⚠ {len(remaining)} sample(s) still failed after "
                f"{max_retry_rounds} retries; excluded from metrics.[/red]"
            )

        metrics = self.compute_metrics()
        
        # Display metrics summary
        valid = len([r for r in self.results if r is not None])
        skipped = len(sample_dirs) - len(self.results)
        console.print(f"\n[green]✓[/green] [bold]Completed:[/bold] {valid} valid / {len(sample_dirs)} total ({skipped} skipped)")
        if metrics:
            console.print(f"[dim]Metrics:[/dim]")
            for key, value in metrics.items():
                if isinstance(value, float):
                    console.print(f"  [cyan]{key}:[/cyan] {value:.4f}")
                else:
                    console.print(f"  [cyan]{key}:[/cyan] {value}")
        else:
            console.print(f"[yellow]⚠ No valid results — all samples were skipped (missing required data)[/yellow]")
        
        return {
            "task_name": self.get_task_name(),
            "task_description": self.get_task_description(),
            "num_samples": len(self.results),
            "metrics": metrics,
            "results": self.results
        }
    
    def _run_sequential(self, sample_dirs: List[str]) -> None:
        """Evaluate samples sequentially with progress bar."""
        errors: List[str] = []
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {self.get_task_name()}",
                total=len(sample_dirs)
            )
            for sample_id in sample_dirs:
                sample_dir = os.path.join(self.data_dir, sample_id)
                try:
                    result = self.evaluate_sample(sample_dir, sample_id)
                    self.results.append(result)
                except Exception as e:
                    errors.append(f"{sample_id}: {e}")
                    self._errored_ids.append(sample_id)
                progress.update(task, advance=1)
        # Print errors after progress bar is done (no interleaving)
        if errors:
            console.print(f"[yellow]  {len(errors)} error(s) during evaluation:[/yellow]")
            for err in errors[:5]:
                console.print(f"  [red]✗[/red] {err}")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")
    
    def _run_parallel(self, sample_dirs: List[str], num_workers: int) -> None:
        """Evaluate samples in parallel with thread-safe progress bar."""
        import threading
        errors: List[str] = []
        results_lock = threading.Lock()
        errors_lock = threading.Lock()
        
        def _eval_one(sample_id: str) -> Optional[Dict[str, Any]]:
            sample_dir = os.path.join(self.data_dir, sample_id)
            try:
                return self.evaluate_sample(sample_dir, sample_id)
            except Exception as e:
                with errors_lock:
                    errors.append(f"{sample_id}: {e}")
                    self._errored_ids.append(sample_id)
                return None
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]{self.get_task_name()} (parallel, {num_workers}w)",
                total=len(sample_dirs)
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_eval_one, sid): sid for sid in sample_dirs}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            self.results.append(result)
                    progress.update(task, advance=1)
        # Print errors after progress bar is done (no interleaving)
        if errors:
            console.print(f"[yellow]  {len(errors)} error(s) during evaluation:[/yellow]")
            for err in errors[:5]:
                console.print(f"  [red]✗[/red] {err}")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")
