# 3D AVATA Appearance

dlib facial landmark 기반 얼굴 파라미터 예측
![다운로드](https://github.com/soo811/3D-Avatar-Apprearance/assets/91643983/4d48bebb-28d6-49aa-b3bd-ba83bf463c83)

*디렉토리*
- DataSet : 학습데이터로 사용한 데이터 셋
- ParameterPrediction : 얼굴 파라미터 예측을 위한 모델
    - dlib : dlib을 이용한 모델
    - mediapipe : google mediapipe 이용 모델
- HairSegment : 헤어, 눈썹 분류를 위한 모델
- models : 예측모델에 사용한 모델 저장본
    - model2 : head size fix 해제한 후 학습한 모델(정확도가 더 낮음)
- Final : 가장 최근 수정 모델(헤어, 눈썹, 얼굴 파라미터 한번에 예측 가능) + 보고서
