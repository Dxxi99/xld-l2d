# Pixel Unit

## Atomic Unit

1 pixel. 모든 마스크, 검증, 캐시 유효성, Undo/Redo는 픽셀 단위.

## Inference Unit

AI는 픽셀 하나를 입력으로 받지 않는다.
문맥 블록(Context Block, tile)을 입력으로 받고, 픽셀 단위 출력을 낸다.

Context Block 크기: 8/16/32 (모델 receptive field에 맞춤)

## Decision Unit

Accept / Reject / Re-inference는 픽셀 단위.
1차 전역 패스 후 Error Pixel Mask를 만들고,
Dilate 해서 Error Tile List를 만든 뒤,
해당 Tile만 재추론한다.

## Rule

- AI 입력: 문맥 블록 (병렬, GPU)
- AI 출력: 픽셀 단위 (병렬, GPU)
- 격리/캐시/복원: 픽셀 단위
