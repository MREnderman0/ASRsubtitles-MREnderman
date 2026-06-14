import os
import torch
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from typing import Optional
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc
from core.resource_limits import choose_device, config_float, config_int, configure_torch_runtime
from core.utils.models import *

class PreloadedSeparator(Separator):
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        configure_torch_runtime(torch)
        device = choose_device("resource_limits.demucs.device", allow_mps=True, torch_module=torch)
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

def demucs_audio():
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    
    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    segment = config_int("resource_limits.demucs.segment", 0) or None
    jobs = config_int("resource_limits.demucs.jobs", 0)
    overlap = config_float("resource_limits.demucs.overlap", 0.25) or 0.25
    separator = PreloadedSeparator(model=model, shifts=1, overlap=overlap, segment=segment, jobs=jobs)
    
    console.print("🎵 Separating audio...")
    try:
        _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)
    except RuntimeError as exc:
        if segment is None or "shape" not in str(exc) or "invalid for input of size" not in str(exc):
            raise
        rprint("[yellow]Demucs failed with configured segment; retrying with model default segment.[/yellow]")
        del separator
        gc.collect()
        separator = PreloadedSeparator(model=model, shifts=1, overlap=overlap, segment=None, jobs=jobs)
        _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)
    
    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2, 
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}
    
    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)
    
    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)
    
    # Clean up memory
    del outputs, background, model, separator
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()
