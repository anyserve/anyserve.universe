"""Minimal Triton Python backend model demonstrating dynamic batching.

Scores a 4-feature vector with a sigmoid linear model and returns both the
score and the dynamic batch size Triton formed for this execution.
"""
import numpy as np
import triton_python_backend_utils as pb_utils

WEIGHTS = np.array([0.8, -0.4, 1.2, 0.6], dtype=np.float32)
BIAS = np.float32(-0.5)


class TritonPythonModel:
    def execute(self, requests):
        batches = [
            pb_utils.get_input_tensor_by_name(req, "FEATURES").as_numpy()
            for req in requests
        ]
        stacked = np.concatenate(batches, axis=0)
        total_batch = np.int32(stacked.shape[0])
        logits = stacked @ WEIGHTS + BIAS
        scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        responses = []
        offset = 0
        for batch in batches:
            n = batch.shape[0]
            out_score = pb_utils.Tensor("SCORE", scores[offset : offset + n].reshape(n, 1))
            out_batch = pb_utils.Tensor("BATCH_SIZE", np.full((n, 1), total_batch, dtype=np.int32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_score, out_batch]))
            offset += n
        return responses
