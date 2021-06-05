# Visualizing CNN Layer Outputs
2021 hufs deep learning course

## CNN을 이용한 차량 번호판 인식 
> 참고 사이트 : https://www.kaggle.com/sarthakvajpayee/license-plate-recognition-using-cnn/data?select=indian_license_plate.xml

## 파일 설명
* data : 학습시킬 때 사용하는 이미지
* test_data : 테스트 할 때 사용하는 이미지
* cnn_visualization.ipynb : 학습시키는 코드 및 수행결과
## 1. Model
![image](https://user-images.githubusercontent.com/53362054/120901753-acf3d180-c677-11eb-8d70-fefdae682e33.png)
## 2. Predict result
![image](https://user-images.githubusercontent.com/53362054/120901851-2d1a3700-c678-11eb-80ea-e103be928375.png)

## 3. Visualizing을 위한 모델 정의 (activation_model)
``` python
# CNN Visualization
layer_cnt = 5
layer_names = [layer.name for layer in model.layers[:layer_cnt]]
layer_outputs = [layer.output for layer in model.layers[:layer_cnt]]
# print(layer_names)
# print(layer_outputs)
activation_model = tf.keras.Model(inputs=model.input,outputs=layer_outputs)
activations = activation_model.predict(test_generator)
```

## 4. 결과
### conv2d_1
![image](https://user-images.githubusercontent.com/53362054/120901924-7b2f3a80-c678-11eb-820d-8df7d3f61b47.png)

### conv2d_2
![image](https://user-images.githubusercontent.com/53362054/120901972-b29de700-c678-11eb-8de6-8913faa9fd48.png)
### conv2d_3
![image](https://user-images.githubusercontent.com/53362054/120901979-be89a900-c678-11eb-8afb-55cf568bbb56.png)
### conv2d_4
![image](https://user-images.githubusercontent.com/53362054/120901989-cc3f2e80-c678-11eb-90a3-317c8c91366a.png)

### maxpool_1
![image](https://user-images.githubusercontent.com/53362054/120902000-d95c1d80-c678-11eb-9479-c6eeeb0dba7a.png)

> 참고사항 : https://keras.io/api/preprocessing/image/
