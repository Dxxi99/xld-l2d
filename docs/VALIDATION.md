# Pixel Validation

모든 픽셀에 대해 다음을 계산한다.

- class confidence
- boundary continuity (edge/line)
- neighbor consistency (color / label)
- occlusion consistency (앞/뒤 관계)
- layer consistency (Hair_Front ⊄ Face)

Accept 조건 (모두 만족):
- confidence >= threshold_class
- boundary continuity >= threshold_boundary
- neighbor delta <= threshold_neighbor

Reject된 픽셀 → Error Pixel Mask → Dilate → 재추론 → 재검증.
