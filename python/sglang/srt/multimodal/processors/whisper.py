from typing import Any, Dict, List, Optional, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        super().__init__(
            hf_config, server_args, _processor, transport_mode, *args, **kwargs
        )

    async def process_mm_data_async(
        self,
        audio_data: Optional[List[Union[str, bytes]]] = None,
        input_text: Union[str, List[int]] = "",
        **kwargs,
    ) -> Optional[Dict[str, Any]]:

        if isinstance(input_text, list) and isinstance(input_text[0], int):
            input_ids = input_text
        else:
            input_ids = self._processor.tokenizer(input_text)["input_ids"]

        output = {"input_ids": input_ids, "mm_items": []}

        # Handle audio input if provided
        if audio_data and len(audio_data) > 0:
            audios = [load_audio(audio) for audio in audio_data]
            assert len(audios) == 1

            input_features = self._processor.feature_extractor(
                audios[0],
                pad_to_multiple_of=320,
                sampling_rate=16000,
                padding="longest",
                return_tensors="pt",
            )["input_features"][0]

            output["mm_items"] = [
                MultimodalDataItem(
                    feature=input_features,
                    modality=Modality.AUDIO,
                )
            ]
            # For Whisper, the encoder reduces sequence length by half due to conv2 stride=2
            output["num_image_tokens"] = input_features.shape[1] // 2
        else:
            # For text-only input, create a small dummy audio feature
            # This satisfies the model's requirement for mm_inputs to be non-None
            # Use small random values instead of zeros to avoid numerical issues
            n_mels = 80  # Standard Whisper mel feature dimension
            seq_len = 16  # Small but reasonable sequence length to avoid div-by-zero
            dummy_audio_feature = (
                torch.randn((n_mels, seq_len), dtype=torch.float32) * 0.001
            )

            output["mm_items"] = [
                MultimodalDataItem(
                    feature=dummy_audio_feature,
                    modality=Modality.AUDIO,
                )
            ]
            # For text-only inputs, use 0 tokens so no encoder processing occurs
            output["num_image_tokens"] = 0

        return output
