import numpy as n

fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
mapping = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

# sp_label_list = list(mapping.keys())
# superclasses = ['flowers', 'vehicles 1', 'large carnivores', 'household furniture', 'household electrical devices', 'insects', 'people', 'large natural outdoor scenes', 'aquatic mammals', 'fruit and vegetables']
# superclasses = ['fruit and vegetables', 'food containers', 'aquatic mammals']
# sp_labels_map = {}
# for sp_cls in superclasses:
#     sp_new_idx = superclasses.index(sp_cls)
#     sp_old_idx = sp_label_list.index(sp_cls)
#     sp_labels_map[sp_old_idx] = sp_new_idx
# for key,val in sp_labels_map.items():
#     print('{}:{}'.format(key,val))
    
# superclasses = ['fruit and vegetables', 'food containers', 'aquatic mammals']
# subclass = []
# for sp_cls in superclasses:
#     subclass_list = mapping[sp_cls]
#     subclass += [fine_labels.index(sub_cls) for sub_cls in subclass_list]
# print(subclass)


# sp_label_list = list(mapping.keys())
# cv_score = [70.572,58.85,62.132,55.94,79.112,65.726,65.094,66.748,64.777,51.376,69.03,61.811,54.728,52.082,68.196,53.568,54.629,59.662,63.046,59.382]
# sort_idx = np.argsort(cv_score)
# superclasses = [sp_label_list[idx] for idx in sort_idx[-10:]]
# # print([sp_label_list.index(label) for label in superclasses])
# print(superclasses)

superclass_labels = list(mapping.keys())
remove_fine_labels = [4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
remove_mapping = {0: 4, 1: 73, 2: 54, 3: 10, 4: 51, 5: 40, 6: 84, 7: 18, 8: 3, 9: 12, 10: 33, 11: 38, 12: 64, 13: 45, 14: 2, 15: 44, 16: 80, 17: 96, 18: 13, 19: 81} #(superclass_idx: subclass_idx)
select_superclasses = superclass_labels
# select_superclasses = ['flowers', 'vehicles 1', 'large carnivores', 'household furniture', 'household electrical devices', 'insects', 'people', 'large natural outdoor scenes', 'aquatic mammals', 'fruit and vegetables']
# select_superclasses = ['fruit and vegetables', 'food containers', 'aquatic mammals']
selected_fine_labels = []
for sp_label in select_superclasses:
    sp_idx = superclass_labels.index(sp_label)
    for sb_label in mapping[sp_label]:
        sb_idx = fine_labels.index(sb_label)
        if sb_idx != remove_mapping[sp_idx]:
            selected_fine_labels.append(sb_idx)
            break
    selected_fine_labels.append(remove_mapping[sp_idx])
assert len(selected_fine_labels) == len(select_superclasses)*2, len(selected_fine_labels)
print(selected_fine_labels)