from .base import BaseTask
from .geometry_perception import (
    FamilyClassificationTask,
    ParameterPredictionTask,
    StructureCountingTask,
    TopologyRecognitionTask
)
from .physics_understanding import (
    LossPredictionTask,
    ApplicationClassificationTask,
    ModeClassificationTask,
    MechanismReasoningTask
)
from .text_generation import (
    GeometryDescriptionTask,
    ConversationQATask,
    ReportGenerationTask
)
from .code_generation_complete import (
    SimulationCodeGenerationTask,
    ParameterModificationTask,
    CodeRefactoringTask
)
from .multimodal_reasoning import (
    WavelengthComparisonTask,
    MechanismDiagnosisTask,
    EvidenceBasedExplanationTask
)
from .inverse_design import (
    ParameterSelectionTask,
    DesignDirectionTask,
    ClosedLoopImprovementTask
)
from .mas_pipeline import MASPipelineTask

__all__ = [
    "BaseTask",
    # Geometry Perception
    "FamilyClassificationTask",
    "ParameterPredictionTask",
    "StructureCountingTask",
    "TopologyRecognitionTask",
    # Physics Understanding
    "LossPredictionTask",
    "ApplicationClassificationTask",
    "ModeClassificationTask",
    "MechanismReasoningTask",
    # Text Generation
    "GeometryDescriptionTask",
    "ConversationQATask",
    "ReportGenerationTask",
    # Code Generation
    "SimulationCodeGenerationTask",
    "ParameterModificationTask",
    "CodeRefactoringTask",
    # Multi-modal Physical Reasoning (NEW)
    "WavelengthComparisonTask",
    "MechanismDiagnosisTask",
    "EvidenceBasedExplanationTask",
    # Inverse Design (NEW)
    "ParameterSelectionTask",
    "DesignDirectionTask",
    "ClosedLoopImprovementTask",
    # Multi-Agent System Pipeline
    "MASPipelineTask",
]
