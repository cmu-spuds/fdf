---
dataset_info:
- config_name: all
  features:
  - name: image
    dtype: image
  - name: landmarks
    struct:
    - name: nose
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_eye
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_eye
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_ear
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_ear
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_shoulder
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_shoulder
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
  - name: bbox
    struct:
    - name: x_min
      dtype: uint16
    - name: y_min
      dtype: uint16
    - name: x_max
      dtype: uint16
    - name: y_max
      dtype: uint16
  - name: metadata
    struct:
    - name: license
      dtype: string
    - name: photo_url
      dtype: string
  splits:
  - name: train
    num_bytes: 76145388
    num_examples: 241982
  - name: validation
    num_bytes: 2033307
    num_examples: 6531
  download_size: 25383711039
  dataset_size: 78178695
- config_name: fdf256
  features:
  - name: image
    dtype: image
  - name: landmarks
    struct:
    - name: nose
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_eye
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_eye
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_ear
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_ear
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: r_shoulder
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: l_shoulder
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
  - name: bbox
    struct:
    - name: x_min
      dtype: uint16
    - name: y_min
      dtype: uint16
    - name: x_max
      dtype: uint16
    - name: y_max
      dtype: uint16
  - name: metadata
    struct:
    - name: license
      dtype: string
    - name: photo_url
      dtype: string
  splits:
  - name: train
    num_bytes: 76145388
    num_examples: 241982
  - name: validation
    num_bytes: 2033307
    num_examples: 6531
  download_size: 25383711039
  dataset_size: 78178695
---
