from rknn.api import RKNN

#create RKNN object
rknn = RKNN(verbose=True)

#model config
rknn.config(channel_mean_value='103.94 116.78 123.68 58.82',
            reorder_channel='0 1 2',
            need_horizontal_merge=True) 
# input data가 3 channel 일때만 이 config 사용
# input이 channel1이면 무시

#Load model
print('--> Loading model') #로딩하는 중임을 알려주는 출력
ret = rknn.load_tensorflow(tf_pb='./freeze.pb', inputs=['Placeholder'], outputs=['Softmax'],input_size_list=[[2500]]) # w*h(w, h, 1)
#오류 발생시 오류 메시지 출력
if ret !=0:
  print('Load failed!')
  exit(ret)
  
#Load 완료시 'done'출력
print('Load done')

#Build model
print('--> Buliding model') # 빌드하려는 중

ret = rknn.build(do_quantization=False)

#Bulid 오류 발생시 오류 메시지 출력
if ret !=0:
  print('Build failed')
  exit(ret)
#Build 완료시 'Build done' 출력
print('Build done')

#Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn('./mask.rknn') # 다음과 같은 이름으로 저장됩니다.

if ret !=0:
  print('Export failed')
  exit(ret)
#Export 완료시 'Export done' 출력
print('Export done')