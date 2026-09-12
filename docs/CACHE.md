# Cache

캐시 키 (정규화):

    hash(
      context_block_bytes ||
      model_id ||
      model_version ||
      params_canonical ||
      neighbor_labels_canonical ||
      pipeline_version
    )

정책:
- Tile 단위로 AI 결과 캐시
- 픽셀 단위로 유효성 관리 (PixelValidityMask)
- 원본/파라미터/모델 변경 시 invalidate
- Warm 모델 LRU, Cold 모델 언로드
