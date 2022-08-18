from utils import *
from sklearn.model_selection import train_test_split

#Loading Data

path = 'myData'
data = importDataInfo(path)

# Balance data
data = balanceData(data, display=False)

imagesPath, steering = loadData(path, data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total # Training images: ', len(xTrain))
print('Total # Validation images: ', len(xVal))


# Model training
model = create_model()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 100, trainFlag=True), steps_per_epoch=300, epochs=10,
		  validation_data=batchGen(xVal, yVal, 100, trainFlag=False), validation_steps=200)

# Saving and Plotting Model params
model.save('nvidia_model.h5')
print('Model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legent(['Training', 'Validation'])
#plt.ylin([0,1]) # Sets the y values from 0 to 1  
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
