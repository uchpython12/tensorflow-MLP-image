import tensorflow as tf
import matplotlib.pyplot as plt
labelString = ['apple', 'aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
               'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
               'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
               'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard',
               'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
               'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
               'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
               'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
               'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
               'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
]
print(len(labelString))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 顯示其中的圖形
s=8
num=0
fig, axes = plt.subplots(s, s,  sharex=True, sharey=True)
for i in range(s):
    for j  in range(s):
        label=y_train[num][0]
        axes[i, j].imshow(x_train[num])
        axes[i, j].set_title('%d,%d,%s' % (num, label, labelString[label]))
        num=num+1
plt.show()
