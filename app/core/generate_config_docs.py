"""
Configuration Documentation Generator

This script generates comprehensive documentation from the ConfigSchema.
It creates markdown documentation including name, type, default, range, 
description, and impact for each configuration option.

Requirements: 8.1, 8.2, 8.5
"""

import sys
from pathlib import Path
from typing import Any
from config_schema import ConfigSchema, ConfigOption

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ConfigDocGenerator:
    """Generates documentation from ConfigSchema."""
    
    # Impact descriptions for configuration options
    IMPACT_DESCRIPTIONS = {
        'capture.fps': 'Higher FPS provides smoother capture but increases CPU/GPU usage and memory consumption. Lower FPS reduces resource usage but may miss fast-changing content.',
        'capture.quality': 'Higher quality produces clearer text for OCR but increases processing time and memory usage. Lower quality is faster but may reduce OCR accuracy.',
        'capture.timeout_ms': 'Longer timeout allows more time for capture but may delay the pipeline. Shorter timeout improves responsiveness but may cause capture failures.',
        'capture.method': 'Auto selects the best method for your system. DirectX is fastest on Windows with GPU. Screenshot is most compatible but slower.',
        
        'ocr.engine': 'EasyOCR provides good accuracy for multiple languages. PaddleOCR is faster but less accurate. Tesseract is lightweight. Manga OCR specializes in Japanese text.',
        'ocr.confidence_threshold': 'Higher threshold reduces false positives but may miss valid text. Lower threshold captures more text but may include noise.',
        'ocr.max_retries': 'More retries improve reliability but increase latency on failures. Fewer retries fail faster but may miss recoverable errors.',
        'ocr.languages': 'More languages increase flexibility but slow down OCR processing. Limit to needed languages for best performance.',
        
        'translation.engine': 'MarianMT works offline and is free but less accurate. Google/DeepL/Azure require API keys and internet but provide better quality.',
        'translation.source_language': 'Must match the language of text being captured. Incorrect source language will reduce translation accuracy and produce poor results.',
        'translation.target_language': 'The language you want text translated into. Must be supported by the selected translation engine. Incorrect target language will reduce translation quality and cause failures.',
        'translation.batch_size': 'Larger batches improve throughput but increase latency and memory usage. Smaller batches reduce latency but lower throughput.',
        'translation.quality_level': 'Higher values favor translation quality over speed. Lower values prioritize speed. Maps to engine quality hints (low/medium/high).',
        'translation.fallback_enabled': 'When enabled, the system tries alternative engines if the primary engine fails. Disabling may cause translation failures when the primary engine is unavailable.',
        'translation.preserve_formatting': 'When enabled, engines attempt to retain original formatting (bold, italic, line breaks). Disabling may improve speed for engines that support it.',
        'translation.quality_filter_enabled': 'When enabled, low-quality translations are filtered out before saving to the smart dictionary. Disabling allows all translations to be saved.',
        'translation.quality_filter_mode': 'Balanced mode (0) allows most reasonable translations. Strict mode (1) raises thresholds, saving only high-confidence translations to the dictionary.',
        'cache.translation_cache_size': 'Larger cache reduces API calls and improves speed for repeated text. Smaller cache uses less memory but may require more API calls.',
        'cache.translation_cache_ttl': 'Longer TTL keeps cached translations available longer, reducing repeat API calls. Shorter TTL ensures fresher translations at the cost of more API requests.',
        
        'pipeline.queue_size': 'Larger queues absorb burst latency spikes but increase memory usage. Smaller queues reduce memory but may drop frames under load.',
        'pipeline.max_workers': 'More workers improve parallelism but increase CPU and memory usage. Fewer workers reduce resource usage but may bottleneck throughput.',
        
        'performance.enable_gpu': 'GPU acceleration significantly speeds up OCR and translation but requires CUDA-capable GPU. Disable if GPU unavailable or causing issues.',
        'performance.enable_frame_skip': 'Frame skipping improves performance by processing fewer frames. Disable for maximum accuracy at cost of higher resource usage.',
        'performance.enable_translation_cache': 'Caching avoids re-translating identical text, saving time and API costs. Disable only for testing or if memory is constrained.',
        'performance.enable_smart_dictionary': 'Smart dictionary improves translation quality for technical terms and proper nouns. Minimal performance impact.',
        
        'translation.google_api_key': 'Required for Google Translate API. Obtain from Google Cloud Console. Stored encrypted using Windows DPAPI.',
        'translation.deepl_api_key': 'Required for DeepL API. Obtain from DeepL website. Stored encrypted using Windows DPAPI.',
        'translation.azure_api_key': 'Required for Azure Translator. Obtain from Azure Portal. Stored encrypted using Windows DPAPI.',
        'translation.azure_region': 'Azure service region for your Translator resource. Must match your Azure subscription region to enable proper API access.',
        
        'urls.libretranslate_api': 'Base URL for LibreTranslate API. Change to use self-hosted instance for better performance or alternative provider.',
        'urls.azure_translator_endpoint': 'Azure Translator API endpoint. Change to use private endpoint for improved security. Typically does not need adjustment.',
        'urls.pytorch_cpu_index': 'PyTorch package index for CPU-only installation. Change to use faster mirror or custom repository. Used during dependency installation.',
        'urls.pytorch_cuda118_index': 'PyTorch package index for CUDA 11.8. Change to use faster mirror. Used during GPU-enabled installation.',
        'urls.pytorch_cuda121_index': 'PyTorch package index for CUDA 12.1. Change to use faster mirror. Used during GPU-enabled installation.',
        'urls.pytorch_cuda124_index': 'PyTorch package index for CUDA 12.4. Change to use faster mirror. Used during GPU-enabled installation.',
        'urls.tesseract_data': 'Repository URL for Tesseract language data files. Change to use faster mirror or local repository. Used during OCR engine setup.',
        
        'timeouts.capture_stage_seconds': 'Maximum time allowed for screen capture. Increase if capture frequently times out. Decrease to fail faster.',
        'timeouts.preprocessing_stage_seconds': 'Maximum time for image preprocessing. Increase for complex images. Decrease to fail faster.',
        'timeouts.ocr_stage_seconds': 'Maximum time for OCR processing. Increase for large images or complex text. Decrease to fail faster.',
        'timeouts.validation_stage_seconds': 'Maximum time for text validation. Increase for complex validation rules. Decrease to improve responsiveness.',
        'timeouts.translation_stage_seconds': 'Maximum time for translation. Increase for large batches or slow APIs. Decrease to fail faster.',
        'timeouts.overlay_stage_seconds': 'Maximum time for overlay rendering. Increase for complex overlays. Decrease to improve responsiveness.',
        'timeouts.pipeline_stop_seconds': 'Maximum time to wait for pipeline to stop gracefully. Increase if pipeline stop frequently times out.',
        'timeouts.thread_join_seconds': 'Maximum time to wait for threads to finish. Increase if seeing thread join warnings.',
        'timeouts.subprocess_ready_seconds': 'Maximum time to wait for subprocess initialization. Increase on slow systems.',
        'timeouts.subprocess_stop_seconds': 'Maximum time to wait for subprocess to stop. Increase if subprocess stop frequently times out.',
        'timeouts.subprocess_terminate_seconds': 'Maximum time to wait for subprocess termination. Increase on slow systems. Decrease to improve shutdown speed.',
        'timeouts.translation_pool_seconds': 'Maximum time for translation pool operations. Increase for large translation batches.',
        'timeouts.health_monitor_seconds': 'Maximum time for health monitor to stop. Increase on slow systems. Decrease to improve shutdown speed.',
        'timeouts.nvidia_smi_seconds': 'Maximum time for nvidia-smi GPU detection. Increase on systems with slow GPU initialization.',
        'timeouts.pip_uninstall_seconds': 'Maximum time for pip uninstall operations. Increase on slow systems or large packages.',
        'timeouts.pip_install_seconds': 'Maximum time for pip install operations. Increase on slow systems or when installing large packages.',
        
        'paths.user_data_dir': 'Directory for user-specific data. Change to use custom location for better organization. Must have write permissions.',
        'paths.models_dir': 'Directory for AI model storage. Change to use custom location or faster storage. Requires significant disk space.',
        'paths.cache_dir': 'Directory for cache files. Change to use custom location or faster storage for improved performance.',
        'paths.temp_dir': 'Directory for temporary files. Change to use custom location or faster storage. Files are automatically cleaned up.',
        'paths.config_file': 'Path to configuration file. Change to use custom location for better organization. Must have write permissions.',
        'paths.image_processing_presets_dir': 'Directory for user-defined image processing presets. Each preset is stored as an individual JSON file.',
        
        'overlay.disappear_timeout_seconds': 'Time before translation overlay disappears. Increase to read translations longer. Decrease for less screen clutter.',
        'overlay.save_interval': 'Number of translations before auto-save. Lower values save more frequently but may impact performance.',
        
        'thresholds.queue_get_timeout_seconds': 'Timeout for internal queue operations. Increase for slower systems. Decrease to improve responsiveness.',
        'thresholds.worker_scale_timeout_seconds': 'Timeout for worker thread scaling. Increase on slow systems. Decrease to improve responsiveness.',
        'thresholds.batch_coordinator_timeout_seconds': 'Timeout for batch coordinator operations. Increase for large batches. Decrease to improve responsiveness.',
        'thresholds.async_wait_timeout_seconds': 'Timeout for async operations. Increase on slow systems. Decrease to improve responsiveness.',
        'thresholds.resource_task_timeout_seconds': 'Timeout for resource allocation. Increase under heavy load. Decrease to improve responsiveness.',
        'thresholds.background_queue_timeout_seconds': 'Timeout for background queue operations. Increase on slow systems. Decrease to improve responsiveness.',
        'thresholds.subprocess_message_timeout_seconds': 'Timeout for subprocess messaging. Increase on slow systems. Decrease to improve responsiveness.',
        
        'retries.model_download_max': 'Number of retry attempts for model downloads. Increase on unreliable networks. Decrease to fail faster.',
        'retries.translation_max': 'Number of retry attempts for translation API calls. Increase for unreliable APIs. Decrease to fail faster.',
        'retries.api_request_max': 'Number of retry attempts for general API requests. Increase for unreliable networks. Decrease to fail faster.',
    }
    
    def __init__(self, schema: ConfigSchema):
        """Initialize generator with schema."""
        self.schema = schema
    
    def generate_markdown(self) -> str:
        """
        Generate complete markdown documentation.
        
        Returns:
            Markdown formatted documentation string
        """
        lines = []
        lines.append("# OptikR Configuration Reference")
        lines.append("")
        lines.append("This document provides a complete reference for all OptikR configuration options.")
        lines.append("")
        lines.append("## Table of Contents")
        lines.append("")
        
        # Group options by category
        categories = self._group_by_category()
        
        # Generate table of contents
        for category in sorted(categories.keys()):
            lines.append(f"- [{category}](#{category.lower().replace(' ', '-')})")
        lines.append("")
        
        # Generate sections for each category
        for category in sorted(categories.keys()):
            lines.append(f"## {category}")
            lines.append("")
            
            options = categories[category]
            for option in sorted(options, key=lambda o: o.name):
                lines.extend(self._format_option(option))
                lines.append("")
        
        return "\n".join(lines)
    
    def _group_by_category(self) -> dict[str, list[ConfigOption]]:
        """Group configuration options by category."""
        categories: dict[str, list[ConfigOption]] = {}
        
        for option in self.schema.get_all_options().values():
            # Extract category from option name (first part before dot)
            category_key = option.name.split('.')[0]
            category_name = category_key.replace('_', ' ').title()
            
            if category_name not in categories:
                categories[category_name] = []
            categories[category_name].append(option)
        
        return categories
    
    def _format_option(self, option: ConfigOption) -> list[str]:
        """Format a single configuration option as markdown."""
        lines = []
        
        # Option name as heading
        lines.append(f"### `{option.name}`")
        lines.append("")
        
        # Description
        lines.append(f"**Description:** {option.description}")
        lines.append("")
        
        # Type
        type_name = option.type.__name__
        lines.append(f"**Type:** `{type_name}`")
        lines.append("")
        
        # Default value
        default_str = self._format_value(option.default, option.type)
        lines.append(f"**Default:** `{default_str}`")
        lines.append("")
        
        # Valid range/choices
        if option.choices:
            choices_str = ", ".join(f"`{c}`" for c in option.choices)
            lines.append(f"**Valid Values:** {choices_str}")
            lines.append("")
        elif option.min_value is not None or option.max_value is not None:
            range_parts = []
            if option.min_value is not None:
                range_parts.append(f"min: `{option.min_value}`")
            if option.max_value is not None:
                range_parts.append(f"max: `{option.max_value}`")
            lines.append(f"**Valid Range:** {', '.join(range_parts)}")
            lines.append("")
        
        # Sensitive flag
        if option.sensitive:
            lines.append("**Security:** This value is encrypted using Windows DPAPI before storage.")
            lines.append("")
        
        # Impact description
        impact = self.IMPACT_DESCRIPTIONS.get(option.name)
        if impact:
            lines.append(f"**Impact:** {impact}")
            lines.append("")
        
        return lines
    
    def _format_value(self, value: Any, value_type: type) -> str:
        """Format a value for display."""
        if value_type == str:
            return f'"{value}"' if value else '""'
        elif value_type == list:
            if not value:
                return "[]"
            return f"[{', '.join(repr(v) for v in value)}]"
        elif value_type == bool:
            return str(value)
        else:
            return str(value)
    
    def generate_to_file(self, output_path: Path) -> None:
        """
        Generate documentation and write to file.
        
        Args:
            output_path: Path to output markdown file
        """
        content = self.generate_markdown()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
        print(f"Configuration documentation generated: {output_path}")


def main():
    """Main entry point for documentation generation."""
    # Initialize schema
    schema = ConfigSchema()
    
    # Create generator
    generator = ConfigDocGenerator(schema)
    
    # Generate documentation
    output_path = Path(__file__).parent.parent.parent / 'docs' / 'CONFIGURATION_REFERENCE.md'
    generator.generate_to_file(output_path)
    
    print(f"Generated documentation for {len(schema.get_all_options())} configuration options")


if __name__ == '__main__':
    main()
