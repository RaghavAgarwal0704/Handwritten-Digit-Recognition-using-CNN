from keras.models import model_from_json
import numpy as np

json_file = open('C:/Users/agarw/Desktop/digit/dig.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights
loaded_model.load_weights("C:/Users/agarw/Desktop/digit/dig.h5")
print("model loaded")

truey = []
predy = []


x_test = np.load("C:/Users/agarw/Desktop/digit/x_test.npy")
y_test = np.load("C:/Users/agarw/Desktop/digit/y_test.npy")


yhat = loaded_model.predict(x_test)
yh = yhat.tolist()
yt = y_test.tolist()
count = 0

for i in range(len(y_test)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy) == yt[i].index(yyt)):
        count += 1
acc = (count/len(y_test))*100


# saving values for confusion matrix
np.save("C:/Users/agarw/Desktop/digit/truey", truey)
np.save("C:/Users/agarw/Desktop/digit/predy", predy)
print("predicted and true label saved")
print("accuracy "+str(acc)+"%")
