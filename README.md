## Simple-Mask-Classify

프로젝트 기간 : 2020.09.07~2020.09.18

프로젝트 내용 : 얼굴 이미지를 입력 받고, 마스크를 썼는지 안썼는지 분류하는 프로그램입니다.

개발환경 : Python3.6, Tensorflow1.10, C++, Rex (AI Computing Embedded Computer)


</br>



------

### Mask-classify-python



**Python과 TensorFlow를 활용해 model을 만들고 학습시킵니다.**

Train image : 5000

Epoch : 100

Mini batch size : 10

Lr : 1e-3

Optimizer : adam

</br>  


**모델의 train, test accuracy**

 

<img src="https://user-images.githubusercontent.com/5088280/102866028-f78ee200-4479-11eb-87e6-38eac1a15fd7.png" alt="image"  />  






**python으로 test input을 넣은 결과**

![image](https://user-images.githubusercontent.com/5088280/102866058-05dcfe00-447a-11eb-9ab5-8eddf0a2818c.png)

그 후 모델을 freeze 시켰고. netron 으로 모델의 구조를 확인할 수 있었습니다.

<img src="https://user-images.githubusercontent.com/5088280/102866138-2442f980-447a-11eb-9bab-4da37693d8ff.png" alt="image" style="zoom:67%;" />  

</br>



------

### Rex-mask-classify


freeze 시킨 pb 파일을 Rex Ai Computing board에서 사용할 수 있는 형태인 RKNN 모델 형태로 변환합니다.

그리고 보드에서 연결된 USB CAM의 이미지 버퍼를 가져와 모델에 입력하고 결과를 출력하는 C++ 프로그램을 개발했습니다.

![image](https://user-images.githubusercontent.com/5088280/102866466-a6cbb900-447a-11eb-8140-7ec0dc5831d6.png)
