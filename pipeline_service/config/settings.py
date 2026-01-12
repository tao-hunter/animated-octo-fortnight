from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent

class Settings(BaseSettings):
    api_title: str = "3D Generation pipeline Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Generated files settings
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # Trellis settings
    trellis_model_id: str = Field(default="jetx/trellis-image-large", env="TRELLIS_MODEL_ID")
    trellis_sparse_structure_steps: int = Field(default=8, env="TRELLIS_SPARSE_STRUCTURE_STEPS")
    trellis_sparse_structure_cfg_strength: float = Field(default=5.75, env="TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH")
    trellis_slat_steps: int = Field(default=20, env="TRELLIS_SLAT_STEPS")
    trellis_slat_cfg_strength: float = Field(default=2.4, env="TRELLIS_SLAT_CFG_STRENGTH")
    trellis_num_oversamples: int = Field(default=3, env="TRELLIS_NUM_OVERSAMPLES")
    compression: bool = Field(default=False, env="COMPRESSION")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(default="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",env="QWEN_EDIT_BASE_MODEL_PATH")
    qwen_edit_model_path: str = Field(default="Qwen/Qwen-Image-Edit-2511",env="QWEN_EDIT_MODEL_PATH")
    qwen_edit_lora_repo: str = Field(default="lightx2v/Qwen-Image-Edit-2511-Lightning",env="QWEN_EDIT_LORA_REPO")
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=4, env="NUM_INFERENCE_STEPS")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(default=config_dir.joinpath("qwen_edit_prompt.json"), env="QWEN_EDIT_PROMPT_PATH")

    # Backgorund removal settings
    background_removal_model_id: str = Field(default="ZhengPeng7/BiRefNet", env="BACKGROUND_REMOVAL_MODEL_ID")
    input_image_size: tuple[int, int] = Field(default=(1024, 1024), env="INPUT_IMAGE_SIZE") # (height, width)
    output_image_size: tuple[int, int] = Field(default=(518, 518), env="OUTPUT_IMAGE_SIZE") # (height, width)
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")
    # Mask post-processing (helps avoid "double objects" from clutter/shadows)
    keep_largest_component: bool = Field(default=True, env="KEEP_LARGEST_COMPONENT")
    mask_gamma: float = Field(default=1.25, env="MASK_GAMMA")

    # Object-only preprocessing
    # Neutral background color used after segmentation (improves stability on cluttered photos)
    object_bg_color: tuple[int, int, int] = Field(default=(128, 128, 128), env="OBJECT_BG_COLOR")
    # Optional second neutral background to encourage background-invariance
    object_bg_color_alt: Optional[tuple[int, int, int]] = Field(default=(112, 120, 132), env="OBJECT_BG_COLOR_ALT")
    # Faster, identity-preserving "multi-view": multiple crops of the same segmented object
    use_multicrop_views: bool = Field(default=True, env="USE_MULTICROP_VIEWS")
    multicrop_padding_factors: tuple[float, float, float] = Field(default=(1.05, 1.20, 1.35), env="MULTICROP_PADDING_FACTORS")
    # Optional mild tone variant for robustness to glare/lighting; keeps identity.
    use_tone_variant: bool = Field(default=True, env="USE_TONE_VARIANT")
    tone_contrast: float = Field(default=0.92, env="TONE_CONTRAST")
    tone_saturation: float = Field(default=0.96, env="TONE_SATURATION")
    tone_brightness: float = Field(default=1.0, env="TONE_BRIGHTNESS")

    # Optional detail-boost variant (helps fine textures like sprinkles, grain, engravings)
    use_detail_variant: bool = Field(default=True, env="USE_DETAIL_VARIANT")
    detail_radius: int = Field(default=2, env="DETAIL_RADIUS")
    detail_percent: int = Field(default=120, env="DETAIL_PERCENT")
    detail_threshold: int = Field(default=3, env="DETAIL_THRESHOLD")

    # Auto-tighten/loosen crop based on alpha coverage (keeps object scale stable)
    auto_tighten_crops: bool = Field(default=True, env="AUTO_TIGHTEN_CROPS")
    target_alpha_coverage: float = Field(default=0.45, env="TARGET_ALPHA_COVERAGE")
    alpha_coverage_tolerance: float = Field(default=0.18, env="ALPHA_COVERAGE_TOLERANCE")
    # Optional (slow / may introduce identity drift). Keep off for 30s budget.
    use_qwen_views: bool = Field(default=False, env="USE_QWEN_VIEWS")
    
    vllm_url: str = "http://localhost:8095/v1"
    vllm_api_key: str = "local"
    vllm_model_name: str = "THUDM/GLM-4.1V-9B-Thinking"
    
    # Maximum number of shape candidates to generate in Stage 1
    max_candidates: int = Field(default=3, env="MAX_CANDIDATES")
    
    # Candidate selection based on voxel count ranges
    # Format: [(max_voxels, num_candidates), ...]
    # Reads as: "if voxels <= max_voxels, use num_candidates"
    candidate_ranges: list[tuple[int, int]] = Field(default=[
        (25000, 3),    # 0-25k voxels → 3 candidates
        (50000, 2),    # 25k-40k voxels → 2 candidates
        (10000, 1),    # 40k-50k voxels → 1 candidate
        # Above 50k → 1 candidate (uses last value)
    ], env="CANDIDATE_RANGES")
    
    # Timeout for generation + judging (seconds)
    generation_timeout: int = Field(default=25, env="GENERATION_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]

