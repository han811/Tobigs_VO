# Tobigs_VO
##PyTorch Implementation of Deep Global- Relative Networks for End-to-End 6-DoF Visual Localization and Odometry
##dataset부분은

###사용법 
1. 구글드라이브에서 KITTI Dataset을 다운받는다.
2. preprocess.py를 돌립니다.
3. preprocess.py에서 나오는 image평균값과 표준편차가 있는데, 이걸 params.py에 인자로 다시 넣어줍니다.
3. main.py를 돌립니다.
(데이터를 계속 read하는 형태이기 때문에 저장하고 싶으면 generate data를 사용하되, 값을 조금씩 바꿔서 써야함)

