import collections
import matplotlib.pyplot as plt

def get_data(file_name):
    a = open(file_name, 'r')
    train_loss = collections.defaultdict(float)
    train_acc = collections.defaultdict(float)
    val_loss = collections.defaultdict(float)
    val_acc = collections.defaultdict(float)
    test_loss = 0
    test_acc = 0
    count = 0
    for line in a:
        line = line.split()
        if line[1] == 'train':
            train_loss[int(line[0])] += float(line[2])
            train_acc[int(line[0])] += float(line[3])
        elif line[1] == 'val':
            val_loss[int(line[0])] += float(line[2])
            val_acc[int(line[0])] += float(line[3])
        elif line[1] == 'test':
            test_loss += float(line[2])
            test_acc += float(line[3])
            count += 1
        else:
            print("error")
    a.close()
    for key in train_loss.keys():
        train_loss[key] /= count
        train_acc[key] /= count
        val_loss[key] /= count
        val_acc[key]  /= count
    test_loss /= count
    test_acc /= count
    
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


def get_plot(file_name):
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = get_data(file_name)
    
    #plt.plot(train_loss.keys(), train_loss.values(), label = 'train_loss')
    #print(train_acc.values())
    plt.plot(train_acc.keys(), train_acc.values(), label = 'train_acc')
    #plt.plot(val_loss.keys(), val_loss.values(), label = 'val_loss')
    plt.plot(val_acc.keys(), val_acc.values(), label = 'val_acc')
    plt.ylim(0, 1) 
    plt.xlim(0, 300) 
    plt.legend()
    plt.show()