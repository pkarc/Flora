"""EXPERIMENTAL: TempSeg through vLLM, in the worker's own process (no HTTP, no JPEG round trip).

Run it from a separate environment (vLLM brings its own torch):

  python3.12 -m venv .venv-vllm && . .venv-vllm/bin/activate
  pip install vllm av "numpy<3" "Pillow>=11,<13" huggingface_hub
  python captioner/segment_worker.py --watch /tmp/w_vllm --backend vllm --once --settle 0 --save-raw

It reuses the reference preprocessing (frame regularisation, prompt, chat template) and hands vLLM the prepared
frames as an array with the same frame metadata, so the model should see the same input. Whether it does is for
captioner/compare_runs.py to say: compare input_tokens per window first, then the raw outputs, then the segments.
Never tested against a real vLLM install when written; expect to adjust to your vLLM version's multimodal API.
"""
import os

import numpy as np
from PIL import Image

from batch_utils.bootstrap import production as a
from batch_utils.config import MODELS
from batch_utils.models import InputErrorForModel, processor, runtime_event, snapshot, validate_manifest
from batch_utils.preprocessing import instruction_suffix

MAX_NEW_TOKENS = 1024


class VLLMBackend:

    def __init__(self, gpu_memory_utilization=0.85, max_model_len=16384, lora_dtype='auto', enforce_eager=False):
        self.gpu_memory_utilization, self.max_model_len = gpu_memory_utilization, max_model_len
        self.lora_dtype, self.enforce_eager = lora_dtype, enforce_eager
        self.last_input_tokens = []

    def load(self):
        try:
            import vllm
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as error:
            raise RuntimeError(f'vLLM is not installed in this environment: {error}') from error
        from huggingface_hub import snapshot_download
        adapter = snapshot('tempseg')
        validate_manifest(adapter)
        spec = MODELS['tempseg_base']
        base = snapshot_download(spec['repo'], revision=spec['revision'], token=os.environ.get('HF_TOKEN'), allow_patterns=['*.safetensors', '*.json', '*.jinja'])
        self.proc = processor('tempseg', adapter)
        kwargs = dict(model=base, tokenizer=str(adapter), enable_lora=True, max_lora_rank=32, max_loras=1, dtype='bfloat16', max_model_len=self.max_model_len,
                      limit_mm_per_prompt={'video': 1}, gpu_memory_utilization=self.gpu_memory_utilization, seed=0, enforce_eager=self.enforce_eager)
        if self.lora_dtype != 'auto':
            kwargs['lora_dtype'] = self.lora_dtype
        runtime_event('vllm_load_start', vllm=vllm.__version__, base=spec['revision'], lora_dtype=self.lora_dtype, eager=self.enforce_eager)
        self.llm = LLM(**kwargs)
        self.lora = LoRARequest('tempseg', 1, str(adapter))
        from vllm import SamplingParams
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        runtime_event('vllm_ready', vllm=vllm.__version__)

    def prepare(self, window, frames, pool, roles, instruction):
        """CPU only. Same regularisation, prompt and template as batch_utils.preprocessing.temporal_inputs."""
        from flora_runtime.exact_frame_serving import regularized_resize_dims
        profile = window.window_profile
        if len(frames) != profile.slot_count:
            raise ValueError('Incorrect native temporal slot count')
        regularized = []
        for image in frames:
            image = image.convert('RGB')
            height, width = regularized_resize_dims(height=image.height, width=image.width, min_pixels=256, max_pixels=40000 if len(roles) == 1 else 160000)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            regularized.append(image)
        prompt = a.temporal_prompt(window, pool) + instruction_suffix(instruction)
        text = self.proc.apply_chat_template([dict(role='user', content=prompt.replace('<video>', self.proc.video_token))], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        array = np.stack([np.asarray(image) for image in regularized])
        fps = float(profile.slot_fps)
        metadata = dict(total_num_frames=len(regularized), fps=fps, duration=len(regularized) / fps, frames_indices=list(range(len(regularized))), video_backend='flora', do_sample_frames=False)
        return dict(prompt=text, video=(array, metadata))

    def infer_batch(self, windows, prepared):
        requests = [{'prompt': p['prompt'], 'multi_modal_data': {'video': p['video']}} for p in prepared]
        outputs = self.llm.generate(requests, self.params, lora_request=self.lora, use_tqdm=False)
        texts, tokens = [], []
        for window, output in zip(windows, outputs):
            tokens.append(len(output.prompt_token_ids))
            completion = output.outputs[0]
            text = completion.text.strip()
            if completion.finish_reason != 'stop' or not text:
                raise InputErrorForModel(f'vLLM did not return a complete output for window {window.window_index} (finish_reason={completion.finish_reason}).')
            texts.append(text)
        self.last_input_tokens = tokens
        return texts

    def infer(self, window, prepared):
        return self.infer_batch([window], [prepared])[0]
