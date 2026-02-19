# Change work directory
import os
os.chdir("/Volumes/YZWD/Machine_Learnings/Projects_ML/Project5/Breast_Cancer/")
# The folder should be structured as
#Breast_Cancer
#├── test
#│   ├── benign
#│   └── malignant
#└── train
#│   ├── benign
#│   └── malignant
# Calculate the time
import time
t0 = time.time()
###############################################################################
def plot_image(folder):
	from matplotlib import pyplot
	from matplotlib.image import imread
	import numpy as np
	# define location of dataset
	images=os.listdir(path=folder)
    # Return random integers from low (inclusive) to high (exclusive).
	np.random.seed(123)
	image_idx=np.random.randint(len(images),size=9)
	# plot first few images
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# load image pixels
		filename=folder+images[image_idx[i]]
		image = imread(filename)
		# plot raw pixel data
		pyplot.imshow(image)
	# show the figure
	pyplot.show()
###############################################################################

# plot images from benign breast cancer tumors
plot_image('train/benign/')
# plot images from malignant breast cancer tumors
plot_image('train/malignant/')


# tensorflow.keras.preprocessing.image.ImageDataGenerator
# generate batches of tensor image data with real-time data augmentation.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create data generators
# photos in the training dataset will be augmented with small (10%)
# random horizontal and vertical shifts and random horizontal flips that create a mirror image of a photo.
datagen = ImageDataGenerator(rescale=1./255,  # scale the pixel values to the range of 0-1.
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=2,  # set range for random zoom
rotation_range = 90,
horizontal_flip=True,  # randomly flip images
vertical_flip=True,  # randomly flip images
)

###############################################################################
# define three Block VGG Model
def define_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import SGD
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
###############################################################################

# define model
model = define_model()

train_it = datagen.flow_from_directory('./train/',
	class_mode='binary', batch_size=64, target_size=(200, 200))
test_it = datagen.flow_from_directory('./test/',
	class_mode='binary', batch_size=64, target_size=(200, 200))
history = model.fit(train_it, steps_per_epoch=len(train_it),
  	validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
# evaluate model
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('The accuracy of VGG3 model is: %.3f' % (acc * 100.0))

###############################################################################
# plot diagnostic learning curves
def summarize_diagnostics(history,outplot_file):
    	from matplotlib import pyplot
    	# plot loss
    	pyplot.subplot(211)
    	pyplot.title('Cross Entropy Loss')
    	pyplot.plot(history.history['loss'], color='blue', label='train')
    	pyplot.plot(history.history['val_loss'], color='orange', label='test')
    	# plot accuracy
    	pyplot.subplot(212)
    	pyplot.title('Classification Accuracy')
    	pyplot.plot(history.history['accuracy'], color='blue', label='train')
    	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    	# save plot to file
    	pyplot.savefig(outplot_file + '_plot.png')
    	pyplot.close()
###############################################################################

summarize_diagnostics(history,'VGG3')
# save model
model.save('VGG3_model.h5')

###############################################################################
# define VGG model for transfer learning
def define_VGG16_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import SGD
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
###############################################################################
# define model
model_VGG16 = define_VGG16_model()

train_it = datagen.flow_from_directory('./train/',
	class_mode='binary', batch_size=64, target_size=(200, 200))
test_it = datagen.flow_from_directory('./test/',
	class_mode='binary', batch_size=64, target_size=(200, 200))
history_VGG16 = model_VGG16.fit(train_it, steps_per_epoch=len(train_it),
  	validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
# evaluate model
_, acc = model_VGG16.evaluate(test_it, steps=len(test_it), verbose=0)
summarize_diagnostics(history_VGG16,'VGG16')
# save model
model_VGG16.save('VGG16_model.h5')

print('The accuracy of VGG16 model is: %.3f' % (acc * 100.0))
t1 = time.time()
total_time = t1-t0
print('The practice takes %s seconds' % total_time)

