def  get_font_color(value):
    if value < 50:
        return "black"
    else:
        return "white"

def generate_confusion_matrix_plot(true_labels, predicted_labels, categories):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)


    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cm_norm = np.asarray(cm)
    
    
    #print("adas")

    cm_norm_mean = cm_norm.sum(axis=1,dtype=np.float32)

    divisor = np.tile(cm_norm_mean, (cm.shape[0], 1)).T 
    print(divisor)
    cm_norm = (cm_norm / divisor ) * 100
    print(cm_norm[3,:]) 

    res = ax.imshow(cm_norm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=100)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    #
    width = len(cm)
    height = len(cm[0])
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str("%.1f" % (cm_norm[x][y])), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=get_font_color(cm_norm[x][y]))

    # add genres as ticks
    alphabet = map(lambda x: x[:13],  sorted(categories))
    
    xticks_range = np.asarray(range(width)) 
    
    #xticks_range = [0,1,2,3,4,5,6,7]
    
    plt.xticks(xticks_range, alphabet, rotation=65)

    plt.yticks(range(height), alphabet[:height])
    return plt
