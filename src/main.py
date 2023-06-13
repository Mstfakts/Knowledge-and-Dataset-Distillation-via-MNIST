import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.dataset.mnist import MnistDataset
from src.model.teacher_model import TeacherModel
from src.model.student_model import StudentModel
from src.utils.constants import PATHS, MODEL_PARAMS
from src.utils.dataset_operations import average_images, select_images

# Some parameters to compare the results
accuracies = list()
misclassifications = list()

# Some basics
remaining_tests = lambda acc_, num_test_: int((1 - acc_) * num_test_)

# Adjust the GPU settings
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Upload dataset and split it
mnist_dataset = MnistDataset()
mnist_dataset.upload_mnist(normalize=True, dtype="float32")
train_gen, test_gen = mnist_dataset.mnist_generator
(X_train, y_train), (X_test, y_test) = mnist_dataset.mnist_numpy

num_train = train_gen.x.shape[0]
num_test = test_gen.x.shape[0]

"""
Teacher Model Section
"""
print("\n\n")
print("##############################")
print("### Teacher Model Section ###")
print("##############################")
teacher = TeacherModel(T=3.5)  # T is set as Softmax Temperature

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

teacher.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=["accuracy"])

teacher.fit(train_gen,
            batch_size=MODEL_PARAMS.BATCH_SIZE,
            epochs=1,
            verbose=1,
            validation_data=test_gen)

y_pred_teacher = np.argmax(teacher.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred_teacher)
misclassification = remaining_tests(acc, num_test)
print("Accuracy for Teacher Model: ", acc)
print(f"Number of samples misclassified: {misclassification}")
accuracies.append(acc)
misclassifications.append(misclassification)

y_train_pred_teacher = teacher.predict(X_train)

if MODEL_PARAMS.IS_SAVE:
    teacher.save(PATHS.TEACHER_PATH)

del teacher
"""
Small Model Section
"""
print("\n\n")
print("###########################")
print("### Small Model Section ###")
print("###########################")
small_model = StudentModel(T=1.0)
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

small_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=["accuracy"])

scores = np.zeros(MODEL_PARAMS.CV)
kf = KFold(n_splits=MODEL_PARAMS.CV)
print("For Small Model;")

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

    train_gen_kf = mnist_dataset.convert_generator(X_train_kf, y_train_kf)
    small_model.fit(train_gen_kf,
                    batch_size=MODEL_PARAMS.BATCH_SIZE,
                    epochs=3,
                    verbose=0)
    y_pred = small_model.predict(X_test_kf)
    y_pred = np.argmax(y_pred, axis=1)

    score = accuracy_score(y_test_kf, y_pred)
    scores[i] = score

    print(f"Fold: {i + 1}, accuracy: {score}")

print("Average accuracy: ", scores.mean())

small_model.fit(train_gen,
                batch_size=MODEL_PARAMS.BATCH_SIZE,
                epochs=3,
                verbose=0,
                validation_data=test_gen)
y_pred_small = np.argmax(small_model.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred_small)
misclassification = remaining_tests(acc, num_test)
print("Accuracy for Small Model:", acc)
print(f"Number of samples misclassified: {misclassification}")
accuracies.append(acc)
misclassifications.append(misclassification)

if MODEL_PARAMS.IS_SAVE:
    small_model.save(PATHS.SMALL_PATH)

del small_model
"""
Distillation Section
"""
print("\n\n")
print("#############################")
print("### Student Model Section ###")
print("#############################")
student_model = StudentModel(T=3.5)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()

student_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=["accuracy"])

scores = np.zeros(MODEL_PARAMS.CV)
kf = KFold(n_splits=MODEL_PARAMS.CV)
print("For Student Model;")
for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = y_train_pred_teacher[train_index], y_train_pred_teacher[test_index]
    train_gen_kf = mnist_dataset.convert_generator(X_train_kf, y_train_kf)
    student_model.fit(train_gen_kf,
                      batch_size=MODEL_PARAMS.BATCH_SIZE,
                      epochs=3,
                      verbose=0)
    y_pred = student_model.predict(X_test_kf)
    y_pred = np.argmax(y_pred, axis=1)

    score = accuracy_score(np.argmax(y_test_kf, axis=1), y_pred)
    scores[i] = score

    print(f"Fold: {i + 1}, accuracy: {score}")

print("Average accuracy: ", scores.mean())

student_model.fit(X_train,
                  y_train_pred_teacher,
                  batch_size=MODEL_PARAMS.BATCH_SIZE,
                  epochs=3,
                  verbose=0)
y_pred_student = np.argmax(student_model(X_test), axis=1)

acc = accuracy_score(y_test, y_pred_student)
misclassification = remaining_tests(acc, num_test)
print("Accuracy for Student Model:", acc)
print(f"Number of samples misclassified: {misclassification}")
accuracies.append(acc)
misclassifications.append(misclassification)

if MODEL_PARAMS.IS_SAVE:
    student_model.save(PATHS.STUDENT_PATH)

del student_model
"""
Avg Model Section
"""
print("\n\n")
print("###########################")
print("### Avg Dataset Section ###")
print("###########################")

# For this section, the averaged MNIST dataset is used.

n_samples = [20, 100, 500, 2500]

for n_sample in n_samples:
    print("\n")
    print(f"For n_sample {n_sample};")
    X_train_avg, y_train_avg = average_images(X_train, y_train, num_samples=n_sample)
    train_gen_kf = mnist_dataset.convert_generator(X_train_avg, y_train_avg)
    avg = TeacherModel(T=3.5)  # T is set as Softmax Temperature

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    avg.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"])

    avg.fit(train_gen_kf,
            batch_size=MODEL_PARAMS.BATCH_SIZE,
            epochs=1,
            verbose=1)

    y_pred_avg = np.argmax(avg.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred_avg)
    misclassification = remaining_tests(acc, num_test)
    print("Accuracy for Avg Model: ", acc)
    print(f"Number of samples misclassified: {misclassification}")
    accuracies.append(acc)
    misclassifications.append(misclassification)

    if MODEL_PARAMS.IS_SAVE:
        avg.save(PATHS.AVG_PATH+str(n_sample))

    del avg

"""
Selection Model Section
"""
print("\n\n")
print("###########################")
print("### Selection ###")
print("###########################")

# For this section, the averaged MNIST dataset is used.

n_samples = [20, 100, 500, 2500]

for n_sample in n_samples:
    print("\n")
    print(f"For n_sample {n_sample};")
    X_train_selection, y_train_selection = select_images(X_train, y_train, num_samples=n_sample)
    train_gen_kf = mnist_dataset.convert_generator(X_train_selection, y_train_selection)
    selection = TeacherModel(T=3.5)  # T is set as Softmax Temperature

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    selection.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"])

    selection.fit(train_gen_kf,
            batch_size=MODEL_PARAMS.BATCH_SIZE,
            epochs=1,
            verbose=1)

    y_pred_selection = np.argmax(selection.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred_selection)
    misclassification = remaining_tests(acc, num_test)
    print("Accuracy for Avg Model: ", acc)
    print(f"Number of samples misclassified: {misclassification}")
    accuracies.append(acc)
    misclassifications.append(misclassification)

    if MODEL_PARAMS.IS_SAVE:
        selection.save(PATHS.SELECTION_PATH+str(n_sample))

    del selection
"""
Result Section
"""
print("\n\n")
print("######################")
print("### Result Section ###")
print("######################")

result_df = pd.DataFrame([accuracies, misclassifications],
                         columns=[
                             'Teacher Model',
                             'Small Model',
                             'Student Model',
                             'Avg Model-1',
                             'Avg Model-100',
                             'Avg Model-500',
                             'Avg Model-2000',
                              'Selection Model-1',
                             'Selection Model-100',
                             'Selection Model-500',
                             'Selection Model-2000'
                         ],
                         index=['Accuracy      : ',
                                'Misclassified : '])
print(result_df.to_string())
