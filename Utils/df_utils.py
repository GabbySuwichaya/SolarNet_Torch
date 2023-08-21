import pdb, os 
import numpy as np
import matplotlib.pyplot as plt

def edit_root_folder(file_name, new_root):
    paths = file_name.split("/")
    children_paths = paths[1:]
    print(children_paths)
    rm_root = os.path.join(new_root, children_paths[0],children_paths[1])
    return rm_root


def get_image_and_label(df_,  new_root, batch_size=0, min_index=0, max_index=None):
    if (max_index is not None):
        last_index = int(min_index + np.floor((max_index-min_index)/batch_size)*batch_size)
    else:
        last_index = int(min_index + np.floor((len(df_)-min_index)/batch_size)*batch_size)
    
    df_        = df_[min_index:last_index].copy()
     
    image_list = []
    label_list = [] 
    for lind_id, (img_, target_) in enumerate(zip(df_['Image'], df_['Target'])):
        print("[%d]: ...." % lind_id)
        img_0 = edit_root_folder(img_[0], new_root)
        img_1 = edit_root_folder(img_[1], new_root)

        print(".... %s ==> %s" %(img_[0], img_0))
        print(".... %s ==> %s" %(img_[1], img_1))
        image_list.append([img_0, img_1])  
        label_list.append(target_)

    return  image_list, label_list 



def show_trainimage_from_X(X, image_name):
    if X.dim() > 3:    
        X_0 = X[0,:3,:,:].numpy()
        X_1 = X[0,3:6,:,:].numpy() 
    else:
        X_0 = X[:3,:,:].numpy()
        X_1 = X[3:6,:,:].numpy() 

    plt.figure(figsize=(10, 5))

    # Display the first image
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(X_0, (1, 2, 0)))  # Transpose to convert from (C, H, W) to (H, W, C)
    plt.title("Image 1")
    plt.axis('off')

    # Display the second image
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(X_1, (1, 2, 0)))  # Transpose to convert from (C, H, W) to (H, W, C)
    plt.title("Image 2")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(image_name)