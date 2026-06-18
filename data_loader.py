from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset/chest_xray/train"
val_dir = "dataset/chest_xray/val"

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

print("Classes:", train_generator.class_indices)